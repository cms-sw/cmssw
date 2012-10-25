/// \class SiStripLorentzAngleCalibration
///
/// Calibration of Lorentz angle (and strip deco mode backplane corrections?)
/// for the tracker, integrated in the alignment algorithms.
///
/// Note that not all algorithms support this...
///
///  \author    : Gero Flucke
///  date       : August 2012
///  $Revision: 1.5 $
///  $Date: 2012/09/20 13:17:24 $
///  (last update by $Author: flucke $)

#include "Alignment/CommonAlignmentAlgorithm/interface/IntegratedCalibrationBase.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "TTree.h"
#include "TFile.h"
#include "TString.h"

// #include <iostream>
#include <vector>
#include <map>
#include <sstream>
#include <cstdio>

class SiStripLorentzAngleCalibration : public IntegratedCalibrationBase
{
public:
  /// Constructor
  explicit SiStripLorentzAngleCalibration(const edm::ParameterSet &cfg);
  
  /// Destructor
  virtual ~SiStripLorentzAngleCalibration();

  /// How many parameters does this calibration define?
  virtual unsigned int numParameters() const;

  // /// Return all derivatives,
  // /// default implementation uses other derivatives(..) method,
  // /// but can be overwritten in derived class for efficiency.
  // virtual std::vector<double> derivatives(const TransientTrackingRecHit &hit,
  // 					  const TrajectoryStateOnSurface &tsos,
  // 					  const edm::EventSetup &setup,
  // 					  const EventInfo &eventInfo) const;

  /// Return non-zero derivatives for x- and y-measurements with their indices by reference.
  /// Return value is their number.
  virtual unsigned int derivatives(std::vector<ValuesIndexPair> &outDerivInds,
				   const TransientTrackingRecHit &hit,
				   const TrajectoryStateOnSurface &tsos,
				   const edm::EventSetup &setup,
				   const EventInfo &eventInfo) const;

  /// Setting the determined parameter identified by index,
  /// should return false if out-of-bounds, true otherwise.
  virtual bool setParameter(unsigned int index, double value);

  /// Setting the determined parameter uncertainty identified by index,
  /// returns false if out-of-bounds, true otherwise.
  virtual bool setParameterError(unsigned int index, double error);

  /// Return current value of parameter identified by index.
  /// Should return 0. if index out-of-bounds.
  virtual double getParameter(unsigned int index) const;

  /// Return current value of parameter identified by index.
  /// Returns 0. if index out-of-bounds or if errors undetermined.
  virtual double getParameterError(unsigned int index) const;

  // /// Call at beginning of job:
  // virtual void beginOfJob(const AlignableTracker *tracker,
  // 			  const AlignableMuon *muon,
  // 			  const AlignableExtras *extras);

  /// Called at end of a the job of the AlignmentProducer.
  /// Write out determined parameters.
  virtual void endOfJob();

private:
  /// If called the first time, fill 'siStripLorentzAngleInput_',
  /// later check that LorentzAngle has not changed.
  bool checkLorentzAngleInput(const edm::EventSetup &setup, const EventInfo &eventInfo);
  /// Input LorentzAngle values:
  /// - either from EventSetup of first call to derivatives(..)
  /// - or created from files of passed by configuration (i.e. from parallel processing)
  const SiStripLorentzAngle* getLorentzAnglesInput();
  /// in non-peak mode the effective thickness is reduced...
  double effectiveThickness(const GeomDet *det, int16_t mode, const edm::EventSetup &setup) const;
  /// take care that map of back plane fractions is properly filled
  void checkBackPlaneFractionMap(const edm::EventSetup &setup);

  /// Determined parameter value for this detId (detId not treated => 0.)
  /// and the given run.
  double getParameterForDetId(unsigned int detId, edm::RunNumber_t run) const;
  /// Index of parameter for given detId (detId not treated => < 0)
  /// and the given run.
  int getParameterIndexFromDetId(unsigned int detId, edm::RunNumber_t run) const;
  /// Total number of IOVs.
  unsigned int numIovs() const;
  /// First run of iov (0 if iovNum not treated).
  edm::RunNumber_t firstRunOfIOV(unsigned int iovNum) const;

  void writeTree(const SiStripLorentzAngle *lorentzAngle, const char *treeName) const;
  SiStripLorentzAngle* createFromTree(const char *fileName, const char *treeName) const;

  const std::string readoutModeName_;
  int16_t readoutMode_;
  const bool saveToDB_;
  const std::string recordNameDBwrite_;
  const std::string outFileName_;
  const std::vector<std::string> mergeFileNames_;

  edm::ESWatcher<SiStripLorentzAngleRcd> watchLorentzAngleRcd_;
  edm::ESWatcher<SiStripConfObjectRcd>   watchBackPlaneRcd_;

  // const AlignableTracker *alignableTracker_;
  SiStripLorentzAngle *siStripLorentzAngleInput_;
  std::map<SiStripDetId::ModuleGeometry, double> backPlaneFractionMap_;

  std::vector<double> parameters_;
  std::vector<double> paramUncertainties_;
};

//======================================================================
//======================================================================
//======================================================================

SiStripLorentzAngleCalibration::SiStripLorentzAngleCalibration(const edm::ParameterSet &cfg)
  : IntegratedCalibrationBase(cfg),
    readoutModeName_(cfg.getParameter<std::string>("readoutMode")),
    saveToDB_(cfg.getParameter<bool>("saveToDB")),
    recordNameDBwrite_(cfg.getParameter<std::string>("recordNameDBwrite")),
    outFileName_(cfg.getParameter<std::string>("treeFile")),
    mergeFileNames_(cfg.getParameter<std::vector<std::string> >("mergeTreeFiles")),
    //    alignableTracker_(0),
    siStripLorentzAngleInput_(0)
{
  // SiStripLatency::singleReadOutMode() returns
  // 1: all in peak, 0: all in deco, -1: mixed state
  // (in principle one could treat even mixed state APV by APV...)
  if (readoutModeName_ == "peak") {
    readoutMode_ = 1;
  } else if (readoutModeName_ == "deconvolution") {
    readoutMode_ = 0;
  } else {
    throw cms::Exception("BadConfig")
	  << "SiStripLorentzAngleCalibration:\n" << "Unknown mode '" 
	  << readoutModeName_ << "', should be 'peak' or 'deconvolution' .\n";
  }

  // FIXME: Which granularity, leading to how many parameters?
  parameters_.resize(2, 0.); // currently two parameters (TIB, TOB), start value 0.
  paramUncertainties_.resize(2, 0.); // dito for errors

  edm::LogInfo("Alignment") << "@SUB=SiStripLorentzAngleCalibration" << "Created with name "
                            << this->name() << " for readout mode '" << readoutModeName_
			    << "',\n" << this->numParameters() << " parameters to be determined."
                            << "\nsaveToDB = " << saveToDB_
                            << "\n outFileName = " << outFileName_
                            << "\n N(merge files) = " << mergeFileNames_.size();
  if (mergeFileNames_.size()) {
    edm::LogInfo("Alignment") << "@SUB=SiStripLorentzAngleCalibration"
                              << "First file to merge: " << mergeFileNames_[0];
  }
}
  
//======================================================================
SiStripLorentzAngleCalibration::~SiStripLorentzAngleCalibration()
{
  //  std::cout << "Destroy SiStripLorentzAngleCalibration named " << this->name() << std::endl;
  delete siStripLorentzAngleInput_;
}

//======================================================================
unsigned int SiStripLorentzAngleCalibration::numParameters() const
{
  return parameters_.size();
}

//======================================================================
unsigned int
SiStripLorentzAngleCalibration::derivatives(std::vector<ValuesIndexPair> &outDerivInds,
					    const TransientTrackingRecHit &hit,
					    const TrajectoryStateOnSurface &tsos,
					    const edm::EventSetup &setup,
					    const EventInfo &eventInfo) const
{
  // ugly const-cast:
  // But it is either only first initialisation or throwing an exception...
  const_cast<SiStripLorentzAngleCalibration*>(this)->checkLorentzAngleInput(setup, eventInfo);

  outDerivInds.clear();

  edm::ESHandle<SiStripLatency> latency;  
  setup.get<SiStripLatencyRcd>().get(latency);
  const int16_t mode = latency->singleReadOutMode();
  if(mode == readoutMode_) {
    if (hit.det()) { // otherwise 'constraint hit' or whatever
      
      const int index = this->getParameterIndexFromDetId(hit.det()->geographicalId(),
							 eventInfo.eventId_.run());
      if (index >= 0) { // otherwise not treated
        edm::ESHandle<MagneticField> magneticField;
        setup.get<IdealMagneticFieldRecord>().get(magneticField);
        const GlobalVector bField(magneticField->inTesla(hit.det()->surface().position()));
        const LocalVector bFieldLocal(hit.det()->surface().toLocal(bField));
        //std::cout << "SiStripLorentzAngleCalibration derivatives " << readoutModeName_ << std::endl;
        const double dZ = this->effectiveThickness(hit.det(), mode, setup);
        // shift due to LA: dx = tan(LA) * dz/2 = mobility * B_y * dz/2,
        // '-' since we have derivative of the residual r = trk -hit
        const double xDerivative = bFieldLocal.y() * dZ * -0.5; // parameter is mobility!
        if (xDerivative) { // If field is zero, this is zero: do not return it
          const Values derivs(xDerivative, 0.); // yDerivative = 0.
          outDerivInds.push_back(ValuesIndexPair(derivs, index));
        }
      }
    } else {
      edm::LogWarning("Alignment") << "@SUB=SiStripLorentzAngleCalibration::derivatives2"
                                   << "Hit without GeomDet, skip!";
    }
  } else if (mode != 0 && mode != 1) { // warn only if unknown/mixed mode  
    edm::LogWarning("Alignment") << "@SUB=SiStripLorentzAngleCalibration::derivatives3"
                                 << "Readout mode is " << mode << ", but looking for "
                                 << readoutMode_ << " (" << readoutModeName_ << ").";
  }
  
  return outDerivInds.size();
}

//======================================================================
bool SiStripLorentzAngleCalibration::setParameter(unsigned int index, double value)
{
  if (index >= parameters_.size()) {
    return false;
  } else {
    parameters_[index] = value;
    return true;
  }
}

//======================================================================
bool SiStripLorentzAngleCalibration::setParameterError(unsigned int index, double error)
{
  if (index >= paramUncertainties_.size()) {
    return false;
  } else {
    paramUncertainties_[index] = error;
    return true;
  }
}

//======================================================================
double SiStripLorentzAngleCalibration::getParameter(unsigned int index) const
{
  //   if (index >= parameters_.size()) {
  //     return 0.;
  //   } else {
  //     return parameters_[index];
  //   }
  return (index >= parameters_.size() ? 0. : parameters_[index]);
}

//======================================================================
double SiStripLorentzAngleCalibration::getParameterError(unsigned int index) const
{
  //   if (index >= paramUncertainties_.size()) {
  //     return 0.;
  //   } else {
  //     return paramUncertainties_[index];
  //   }
  return (index >= paramUncertainties_.size() ? 0. : paramUncertainties_[index]);
}

// //======================================================================
// void SiStripLorentzAngleCalibration::beginOfJob(const AlignableTracker *tracker,
//                                                 const AlignableMuon */*muon*/,
//                                                 const AlignableExtras */*extras*/)
// {
//   alignableTracker_ = tracker;
// }


//======================================================================
void SiStripLorentzAngleCalibration::endOfJob()
{
  // loginfo output
  std::ostringstream out;
  out << "Parameter results for readout mode '" << readoutModeName_ << "'\n";
  for (unsigned int iPar = 0; iPar < parameters_.size(); ++iPar) {
    out << iPar << ": " << parameters_[iPar] << " +- " << paramUncertainties_[iPar] << "\n";
  }
  edm::LogInfo("Alignment") << "@SUB=SiStripLorentzAngleCalibration::endOfJob" << out.str();

  // now write 'input' tree
  const SiStripLorentzAngle *input = this->getLorentzAnglesInput(); // never NULL
  const std::string treeName(this->name() + '_' + readoutModeName_ + '_');
  this->writeTree(input, (treeName + "input").c_str());

  if (input->getLorentzAngles().empty()) {
    edm::LogError("Alignment") << "@SUB=SiStripLorentzAngleCalibration::endOfJob"
			       << "Input Lorentz angle map is empty ('"
			       << readoutModeName_ << "' mode), skip writing output!";
    return;
  }

  for (unsigned int iIOV = 0; iIOV < this->numIovs(); ++iIOV) {
    cond::Time_t firstRunOfIOV = this->firstRunOfIOV(iIOV);
    SiStripLorentzAngle *output = new SiStripLorentzAngle;
    // Loop on map of values from input and add (possible) parameter results
    for (auto iterIdValue = input->getLorentzAngles().begin();
	 iterIdValue != input->getLorentzAngles().end(); ++iterIdValue) {
      // type of (*iterIdValue) is pair<unsigned int, float>
      const unsigned int detId = iterIdValue->first; // key of map is DetId
      // Nasty: putLorentzAngle(..) takes float by reference - not even const reference!
      float value = iterIdValue->second + this->getParameterForDetId(detId, firstRunOfIOV);
      output->putLorentzAngle(detId, value); // put result in output
    }

    // Write this even for mille jobs?
    this->writeTree(output, (treeName + Form("result_%lld", firstRunOfIOV)).c_str());

    if (saveToDB_) { // If requested, write out to DB 
      edm::Service<cond::service::PoolDBOutputService> dbService;
      if (dbService.isAvailable()) {
	dbService->writeOne(output, firstRunOfIOV, recordNameDBwrite_.c_str());
	// no 'delete output;': writeOne(..) took over ownership
      } else {
	delete output;
	edm::LogError("BadConfig") << "@SUB=SiStripLorentzAngleCalibration::endOfJob"
				   << "No PoolDBOutputService available, but saveToDB true!";
      }
    } else {
      delete output;
    }
  } // end loop on IOVs
}

//======================================================================
bool SiStripLorentzAngleCalibration::checkLorentzAngleInput(const edm::EventSetup &setup,
							    const EventInfo &eventInfo)
{
  edm::ESHandle<SiStripLorentzAngle> lorentzAngleHandle;
  if (!siStripLorentzAngleInput_) {
    setup.get<SiStripLorentzAngleRcd>().get(readoutModeName_, lorentzAngleHandle);
    siStripLorentzAngleInput_ = new SiStripLorentzAngle(*lorentzAngleHandle);
  } else {
    if (watchLorentzAngleRcd_.check(setup)) { // new IOV of input - but how to check peak vs deco?
      setup.get<SiStripLorentzAngleRcd>().get(readoutModeName_, lorentzAngleHandle);
      if (lorentzAngleHandle->getLorentzAngles() // but only bad if non-identical values
	  != siStripLorentzAngleInput_->getLorentzAngles()) { // (comparing maps)
	// FIXME: Could different maps have same content, but different order?
	//        Or 'floating point comparison' problem?
	throw cms::Exception("BadInput")
	  << "SiStripLorentzAngleCalibration::checkLorentzAngleInput:\n"
	  << "Content of SiStripLorentzAngle changed at run " << eventInfo.eventId_.run()
	  << ", but algorithm expects constant input!\n";
	return false; // not reached...
      }
    }
  }
  
  return true;
}

//======================================================================
double SiStripLorentzAngleCalibration::effectiveThickness(const GeomDet *det,
							  int16_t mode,
							  const edm::EventSetup &setup) const
{
  if (!det) return 0.;
  double dZ = det->surface().bounds().thickness(); // it is a float only...

  // readoutMode_ == 1: "peak", == 0: "deconvolution"
  // FIXME: SiStripConfObject also stores values for peak?
  if (mode != 1) { // local reco always applies except for peak
    // cons_cast since method may initialise values
    const_cast<SiStripLorentzAngleCalibration*>(this)->checkBackPlaneFractionMap(setup);
    const SiStripDetId id(det->geographicalId());
    auto iter = backPlaneFractionMap_.find(id.moduleGeometry());
    if (iter != backPlaneFractionMap_.end()) {
      //std::cout << "apply " << iter->second << " for subdet " << id.subDetector() 
      //          << " with thickness " << dZ << std::endl;
      dZ *= (1. - iter->second);
    } else {
      edm::LogError("Alignment") << "@SUB=SiStripLorentzAngleCalibration::effectiveThickness"
                                 << "Unknown SiStrip module type " << id.moduleGeometry();
    }
  }

  return dZ;
} 

//======================================================================
void SiStripLorentzAngleCalibration::checkBackPlaneFractionMap(const edm::EventSetup &setup)
{
  // Filled and valid? => Just return!
  // FIXME: Why called twice?? Should we better do
  // if (!(backPlaneFractionMap_.empty() || watchBackPlaneRcd_.check(setup))) return;
  if (!backPlaneFractionMap_.empty() && !watchBackPlaneRcd_.check(setup)) return;

  // All module types, see StripCPE constructor:
  std::vector<std::pair<std::string, SiStripDetId::ModuleGeometry> > nameTypes;
  nameTypes.push_back(std::make_pair("IB1",SiStripDetId::IB1));
  nameTypes.push_back(std::make_pair("IB2",SiStripDetId::IB2));
  nameTypes.push_back(std::make_pair("OB1",SiStripDetId::OB1));
  nameTypes.push_back(std::make_pair("OB2",SiStripDetId::OB2));
  nameTypes.push_back(std::make_pair("W1A",SiStripDetId::W1A));
  nameTypes.push_back(std::make_pair("W2A",SiStripDetId::W2A));
  nameTypes.push_back(std::make_pair("W3A",SiStripDetId::W3A));
  nameTypes.push_back(std::make_pair("W1B",SiStripDetId::W1B));
  nameTypes.push_back(std::make_pair("W2B",SiStripDetId::W2B));
  nameTypes.push_back(std::make_pair("W3B",SiStripDetId::W3B));
  nameTypes.push_back(std::make_pair("W4" ,SiStripDetId::W4 ));
  nameTypes.push_back(std::make_pair("W5" ,SiStripDetId::W5 ));
  nameTypes.push_back(std::make_pair("W6" ,SiStripDetId::W6 ));
  nameTypes.push_back(std::make_pair("W7" ,SiStripDetId::W7 ));

  edm::ESHandle<SiStripConfObject> stripConfObj;
  setup.get<SiStripConfObjectRcd>().get(stripConfObj);

  backPlaneFractionMap_.clear(); // Just to be sure!
  for (auto nameTypeIt = nameTypes.begin(); nameTypeIt != nameTypes.end(); ++nameTypeIt) {
    // See StripCPE constructor:
    const std::string modeS(readoutModeName_ == "peak" ? "Peak" : "Deco");
    const std::string shiftS("shift_" + nameTypeIt->first + modeS);
    if (stripConfObj->isParameter(shiftS)) {
      backPlaneFractionMap_[nameTypeIt->second] = stripConfObj->get<double>(shiftS);
      std::cout << "backPlaneFraction for " << nameTypeIt->first << ": " 
                << backPlaneFractionMap_[nameTypeIt->second] << std::endl;
    } else {
      std::cout << "No " << shiftS << " in SiStripConfObject!?!";
    }
  }
}

//======================================================================
const SiStripLorentzAngle* SiStripLorentzAngleCalibration::getLorentzAnglesInput()
{
  // For parallel processing in Millepede II, create SiStripLorentzAngle
  // from info stored in files of parallel jobs and check that they are identical.
  // If this job has run on data, still check that LA is identical to the ones
  // from mergeFileNames_.
  const std::string treeName(((this->name() + '_') += readoutModeName_) += "_input");
  for (auto iFile = mergeFileNames_.begin(); iFile != mergeFileNames_.end(); ++iFile) {
    SiStripLorentzAngle* la = this->createFromTree(iFile->c_str(), treeName.c_str());
    // siStripLorentzAngleInput_ could be non-null from previous file of this loop
    // or from checkLorentzAngleInput(..) when running on data in this job as well
    if (!siStripLorentzAngleInput_ || siStripLorentzAngleInput_->getLorentzAngles().empty()) {
      delete siStripLorentzAngleInput_; // NULL or empty
      siStripLorentzAngleInput_ = la;
    } else {
      // FIXME: about comparison of maps see comments in checkLorentzAngleInput
      if (la && !la->getLorentzAngles().empty() && // single job might not have got events
          la->getLorentzAngles() != siStripLorentzAngleInput_->getLorentzAngles()) {
        // Throw exception instead of error?
        edm::LogError("NoInput") << "@SUB=SiStripLorentzAngleCalibration::getLorentzAnglesInput"
                                 << "Different input values from tree " << treeName
                                 << " in file " << *iFile << ".";
        
      }
      delete la;
    }
  }

  if (!siStripLorentzAngleInput_) { // no files nor ran on events
    siStripLorentzAngleInput_ = new SiStripLorentzAngle;
    edm::LogError("NoInput") << "@SUB=SiStripLorentzAngleCalibration::getLorentzAnglesInput"
			     << "No input, create an empty one ('" << readoutModeName_ << "' mode)!";
  } else if (siStripLorentzAngleInput_->getLorentzAngles().empty()) {
    edm::LogError("NoInput") << "@SUB=SiStripLorentzAngleCalibration::getLorentzAnglesInput"
			     << "Empty result ('" << readoutModeName_ << "' mode)!";
  }

  return siStripLorentzAngleInput_;
}

//======================================================================
double SiStripLorentzAngleCalibration::getParameterForDetId(unsigned int detId,
							    edm::RunNumber_t run) const
{
  const int index = this->getParameterIndexFromDetId(detId, run);

  return (index < 0 ? 0. : parameters_[index]);
}

//======================================================================
int SiStripLorentzAngleCalibration::getParameterIndexFromDetId(unsigned int detId,
							       edm::RunNumber_t run) const
{
  // Return the index of the parameter that is used for this DetId.
  // If this DetId is not treated, return values < 0.
  
  // FIXME: Extend to configurable granularity? 
  //        Including treatment of run dependence?
  const SiStripDetId id(detId);
  if (id.det() == DetId::Tracker) {
    if      (id.subDetector() == SiStripDetId::TIB) return 0;
    else if (id.subDetector() == SiStripDetId::TOB) return 1;
  }

  return -1;
}

//======================================================================
unsigned int SiStripLorentzAngleCalibration::numIovs() const
{
  // FIXME: Needed to include treatment of run dependence!
  return 1; 
}

//======================================================================
edm::RunNumber_t SiStripLorentzAngleCalibration::firstRunOfIOV(unsigned int iovNum) const
{
  // FIXME: Needed to include treatment of run dependence!
  if (iovNum < this->numIovs()) return 1;
  else return 0;
}


//======================================================================
void SiStripLorentzAngleCalibration::writeTree(const SiStripLorentzAngle *lorentzAngle,
					       const char *treeName) const
{
  if (!lorentzAngle) return;

  TFile* file = TFile::Open(outFileName_.c_str(), "UPDATE");
  if (!file) {
    edm::LogError("BadConfig") << "@SUB=SiStripLorentzAngleCalibration::writeTree"
			       << "Could not open file '" << outFileName_ << "'.";
    return;
  }

  TTree *tree = new TTree(treeName, treeName);
  unsigned int id = 0;
  float value = 0.;
  tree->Branch("detId", &id, "detId/i");
  tree->Branch("value", &value, "value/F");

  for (auto iterIdValue = lorentzAngle->getLorentzAngles().begin();
       iterIdValue != lorentzAngle->getLorentzAngles().end(); ++iterIdValue) {
    // type of (*iterIdValue) is pair<unsigned int, float>
    id = iterIdValue->first; // key of map is DetId
    value = iterIdValue->second;
    tree->Fill();
  }
  tree->Write();
  delete file; // tree vanishes with the file... (?)

}

//======================================================================
SiStripLorentzAngle* 
SiStripLorentzAngleCalibration::createFromTree(const char *fileName, const char *treeName) const
{
  // Check for file existence on your own to work around
  // https://hypernews.cern.ch/HyperNews/CMS/get/swDevelopment/2715.html:
  TFile* file = 0;
  FILE* testFile = fopen(fileName,"r");
  if (testFile) {
    fclose(testFile);
    file = TFile::Open(fileName, "READ");
  } // else not existing, see error below

  TTree *tree = 0;
  if (file) file->GetObject(treeName, tree);

  SiStripLorentzAngle *result = 0;
  if (tree) {
    unsigned int id = 0;
    float value = 0.;
    tree->SetBranchAddress("detId", &id);
    tree->SetBranchAddress("value", &value);

    result = new SiStripLorentzAngle;
    const Long64_t nEntries = tree->GetEntries();
    for (Long64_t iEntry = 0; iEntry < nEntries; ++iEntry) {
      tree->GetEntry(iEntry);
      result->putLorentzAngle(id, value);
    }
  } else { // Warning only since could be parallel job on no events.
    edm::LogWarning("Alignment") << "@SUB=SiStripLorentzAngleCalibration::createFromTree"
                                 << "Could not get TTree '" << treeName << "' from file '"
                                 << fileName << (file ? "'." : "' (file does not exist).");
  }

  delete file; // tree will vanish with file
  return result;
}


//======================================================================
//======================================================================
// Plugin definition

#include "Alignment/CommonAlignmentAlgorithm/interface/IntegratedCalibrationPluginFactory.h"

DEFINE_EDM_PLUGIN(IntegratedCalibrationPluginFactory,
		   SiStripLorentzAngleCalibration, "SiStripLorentzAngleCalibration");
