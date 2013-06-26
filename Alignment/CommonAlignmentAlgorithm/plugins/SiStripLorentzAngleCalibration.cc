/// \class SiStripLorentzAngleCalibration
///
/// Calibration of Lorentz angle for the strip tracker, integrated in the
/// alignment algorithms. Note that not all algorithms support this...
///
/// Use one instance for peak and/or one instance for deco mode data.
///
///  \author    : Gero Flucke
///  date       : August 2012
///  $Revision: 1.7 $
///  $Date: 2013/05/31 12:13:40 $
///  (last update by $Author: flucke $)

#include "Alignment/CommonAlignmentAlgorithm/interface/IntegratedCalibrationBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/TkModuleGroupSelector.h"
// include 'locally':
#include "SiStripReadoutModeEnums.h"
#include "TreeStruct.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
//#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "TTree.h"
#include "TFile.h"
#include "TString.h"

// #include <iostream>
#include <boost/assign/list_of.hpp>
#include <vector>
#include <map>
#include <sstream>
#include <cstdio>
#include <functional>

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
  /// returns false if out-of-bounds, true otherwise.
  virtual bool setParameter(unsigned int index, double value);

  /// Setting the determined parameter uncertainty identified by index,
  /// returns false if out-of-bounds, true otherwise.
  virtual bool setParameterError(unsigned int index, double error);

  /// Return current value of parameter identified by index.
  /// Returns 0. if index out-of-bounds.
  virtual double getParameter(unsigned int index) const;

  /// Return current value of parameter identified by index.
  /// Returns 0. if index out-of-bounds or if errors undetermined.
  virtual double getParameterError(unsigned int index) const;

  // /// Call at beginning of job:
  virtual void beginOfJob(AlignableTracker *tracker,
  			  AlignableMuon *muon,
  			  AlignableExtras *extras);
  

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

  /// Determined parameter value for this detId (detId not treated => 0.)
  /// and the given run.
  double getParameterForDetId(unsigned int detId, edm::RunNumber_t run) const;

  void writeTree(const SiStripLorentzAngle *lorentzAngle,
		 const std::map<unsigned int,TreeStruct> &treeInfo, const char *treeName) const;
  SiStripLorentzAngle* createFromTree(const char *fileName, const char *treeName) const;
  
  const std::string readoutModeName_;
  int16_t readoutMode_;
  const bool saveToDB_;
  const std::string recordNameDBwrite_;
  const std::string outFileName_;
  const std::vector<std::string> mergeFileNames_;

  edm::ESWatcher<SiStripLorentzAngleRcd> watchLorentzAngleRcd_;

  SiStripLorentzAngle *siStripLorentzAngleInput_;

  std::vector<double> parameters_;
  std::vector<double> paramUncertainties_;

  TkModuleGroupSelector *moduleGroupSelector_;
  const edm::ParameterSet moduleGroupSelCfg_;
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
    siStripLorentzAngleInput_(0),
    moduleGroupSelector_(0),
    moduleGroupSelCfg_(cfg.getParameter<edm::ParameterSet>("LorentzAngleModuleGroups"))
{

  // SiStripLatency::singleReadOutMode() returns
  // 1: all in peak, 0: all in deco, -1: mixed state
  // (in principle one could treat even mixed state APV by APV...)
  if (readoutModeName_ == "peak") {
    readoutMode_ = kPeakMode;
  } else if (readoutModeName_ == "deconvolution") {
    readoutMode_ = kDeconvolutionMode;
  } else {
    throw cms::Exception("BadConfig")
	  << "SiStripLorentzAngleCalibration:\n" << "Unknown mode '" 
	  << readoutModeName_ << "', should be 'peak' or 'deconvolution' .\n";
  }

}
  
//======================================================================
SiStripLorentzAngleCalibration::~SiStripLorentzAngleCalibration()
{
  delete moduleGroupSelector_;
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
  if (mode == readoutMode_) {
    if (hit.det()) { // otherwise 'constraint hit' or whatever
      
      const int index = moduleGroupSelector_->getParameterIndexFromDetId(hit.det()->geographicalId(),
                                                                    eventInfo.eventId_.run());
      if (index >= 0) { // otherwise not treated
        edm::ESHandle<MagneticField> magneticField;
        setup.get<IdealMagneticFieldRecord>().get(magneticField);
        const GlobalVector bField(magneticField->inTesla(hit.det()->surface().position()));
        const LocalVector bFieldLocal(hit.det()->surface().toLocal(bField));
        //std::cout << "SiStripLorentzAngleCalibration derivatives " << readoutModeName_ << std::endl;
        const double dZ = this->effectiveThickness(hit.det(), mode, setup);
        // shift due to LA: dx = tan(LA) * dz/2 = mobility * B_y * dz/2,
        // '-' since we have derivative of the residual r = hit - trk and mu is part of trk model
	//   (see GF's presentation in alignment meeting 25.10.2012,
	//    https://indico.cern.ch/conferenceDisplay.py?confId=174266#2012-10-25)
        // Hmm! StripCPE::fillParams() defines, together with 
        //      StripCPE::driftDirection(...):
        //      drift.x = -mobility * by * thickness (full drift from backside)
        //      So '-' already comes from that, not from mobility being part of
        //      track model...
        const double xDerivative = bFieldLocal.y() * dZ * -0.5; // parameter is mobility!
        if (xDerivative) { // If field is zero, this is zero: do not return it
          const Values derivs(xDerivative, 0.); // yDerivative = 0.
          outDerivInds.push_back(ValuesIndexPair(derivs, index));
        }
      }
    } else {
      edm::LogWarning("Alignment") << "@SUB=SiStripLorentzAngleCalibration::derivatives1"
                                   << "Hit without GeomDet, skip!";
    }
  } else if (mode != kDeconvolutionMode && mode != kPeakMode) {
    // warn only if unknown/mixed mode  
    edm::LogWarning("Alignment") << "@SUB=SiStripLorentzAngleCalibration::derivatives2"
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
  return (index >= parameters_.size() ? 0. : parameters_[index]);
}

//======================================================================
double SiStripLorentzAngleCalibration::getParameterError(unsigned int index) const
{
  return (index >= paramUncertainties_.size() ? 0. : paramUncertainties_[index]);
}

//======================================================================
void SiStripLorentzAngleCalibration::beginOfJob(AlignableTracker *aliTracker,
                                                AlignableMuon * /*aliMuon*/,
                                                AlignableExtras * /*aliExtras*/)
{
  //specify the sub-detectors for which the LA is determined
  const std::vector<int> sdets = boost::assign::list_of(SiStripDetId::TIB)(SiStripDetId::TOB); //no TEC,TID
  moduleGroupSelector_ = new TkModuleGroupSelector(aliTracker, moduleGroupSelCfg_, sdets);
 
  parameters_.resize(moduleGroupSelector_->getNumberOfParameters(), 0.);
  paramUncertainties_.resize(moduleGroupSelector_->getNumberOfParameters(), 0.);

  edm::LogInfo("Alignment") << "@SUB=SiStripLorentzAngleCalibration" << "Created with name "
                            << this->name() << " for readout mode '" << readoutModeName_
			    << "',\n" << this->numParameters() << " parameters to be determined."
                            << "\nsaveToDB = " << saveToDB_
                            << "\n outFileName = " << outFileName_
                            << "\n N(merge files) = " << mergeFileNames_.size()
                            << "\n number of IOVs = " << moduleGroupSelector_->numIovs();

  if (mergeFileNames_.size()) {
    edm::LogInfo("Alignment") << "@SUB=SiStripLorentzAngleCalibration"
                              << "First file to merge: " << mergeFileNames_[0];
  }
}


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

  std::map<unsigned int, TreeStruct> treeInfo; // map of TreeStruct for each detId

  // now write 'input' tree
  const SiStripLorentzAngle *input = this->getLorentzAnglesInput(); // never NULL
  const std::string treeName(this->name() + '_' + readoutModeName_ + '_');
  this->writeTree(input, treeInfo, (treeName + "input").c_str()); // empty treeInfo for input...

  if (input->getLorentzAngles().empty()) {
    edm::LogError("Alignment") << "@SUB=SiStripLorentzAngleCalibration::endOfJob"
			       << "Input Lorentz angle map is empty ('"
			       << readoutModeName_ << "' mode), skip writing output!";
    return;
  }

  const unsigned int nonZeroParamsOrErrors =   // Any determined value?
    count_if (parameters_.begin(), parameters_.end(), std::bind2nd(std::not_equal_to<double>(),0.))
    + count_if(paramUncertainties_.begin(), paramUncertainties_.end(),
               std::bind2nd(std::not_equal_to<double>(), 0.));

  for (unsigned int iIOV = 0; iIOV < moduleGroupSelector_->numIovs(); ++iIOV) {
    cond::Time_t firstRunOfIOV = moduleGroupSelector_->firstRunOfIOV(iIOV);
    SiStripLorentzAngle *output = new SiStripLorentzAngle;
    // Loop on map of values from input and add (possible) parameter results
    for (auto iterIdValue = input->getLorentzAngles().begin();
	 iterIdValue != input->getLorentzAngles().end(); ++iterIdValue) {
      // type of (*iterIdValue) is pair<unsigned int, float>
      const unsigned int detId = iterIdValue->first; // key of map is DetId
      // Some code one could use to miscalibrate wrt input:
      // double param = 0.;
      // const DetId id(detId);
      // if (id.subdetId() == 3) { // TIB
      //   param = (readoutMode_ == kPeakMode ? -0.003 : -0.002);
      // } else if (id.subdetId() == 5) { // TOB
      //   param = (readoutMode_ == kPeakMode ? 0.005 : 0.004);
      // }
      const double param = this->getParameterForDetId(detId, firstRunOfIOV);
      // put result in output, i.e. sum of input and determined parameter:
      output->putLorentzAngle(detId, iterIdValue->second + param);
      const int paramIndex = moduleGroupSelector_->getParameterIndexFromDetId(detId,firstRunOfIOV);
      treeInfo[detId] = TreeStruct(param, this->getParameterError(paramIndex), paramIndex);
    }

    if (saveToDB_ || nonZeroParamsOrErrors != 0) { // Skip writing mille jobs...
      this->writeTree(output, treeInfo, (treeName + Form("result_%lld", firstRunOfIOV)).c_str());
    }

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
    // FIXME: Should we call 'watchLorentzAngleRcd_.check(setup)' as well?
    //        Otherwise could be that next check has to check via following 'else', though
    //        no new IOV has started... (to be checked)
  } else {
    if (watchLorentzAngleRcd_.check(setup)) { // new IOV of input - but how to check peak vs deco?
      setup.get<SiStripLorentzAngleRcd>().get(readoutModeName_, lorentzAngleHandle);
      if (lorentzAngleHandle->getLorentzAngles() // but only bad if non-identical values
	  != siStripLorentzAngleInput_->getLorentzAngles()) { // (comparing maps)
	// Maps are containers sorted by key, but comparison problems may arise from
	// 'floating point comparison' problems (FIXME?)
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
  const SiStripDetId id(det->geographicalId());
  edm::ESHandle<SiStripBackPlaneCorrection> backPlaneHandle;
  // FIXME: which one? DepRcd->get(handle) or Rcd->get(readoutModeName_, handle)??
  // setup.get<SiStripBackPlaneCorrectionDepRcd>().get(backPlaneHandle); // get correct mode
  setup.get<SiStripBackPlaneCorrectionRcd>().get(readoutModeName_, backPlaneHandle);
  const double bpCor = backPlaneHandle->getBackPlaneCorrection(id); // it's a float...
  //  std::cout << "bpCor " << bpCor << " in subdet " << id.subdetId() << std::endl;
  dZ *= (1. - bpCor);
 
  return dZ;
} 

//======================================================================
const SiStripLorentzAngle* SiStripLorentzAngleCalibration::getLorentzAnglesInput()
{
  // For parallel processing in Millepede II, create SiStripLorentzAngle
  // from info stored in files of parallel jobs and check that they are identical.
  // If this job has run on events, still check that LA is identical to the ones
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
  const int index = moduleGroupSelector_->getParameterIndexFromDetId(detId, run);

  return (index < 0 ? 0. : parameters_[index]);
}

//======================================================================
void SiStripLorentzAngleCalibration::writeTree(const SiStripLorentzAngle *lorentzAngle,
					       const std::map<unsigned int, TreeStruct> &treeInfo,
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
  TreeStruct treeStruct;
  tree->Branch("detId", &id, "detId/i");
  tree->Branch("value", &value, "value/F");
  tree->Branch("treeStruct", &treeStruct, TreeStruct::LeafList());

  for (auto iterIdValue = lorentzAngle->getLorentzAngles().begin();
       iterIdValue != lorentzAngle->getLorentzAngles().end(); ++iterIdValue) {
    // type of (*iterIdValue) is pair<unsigned int, float>
    id = iterIdValue->first; // key of map is DetId
    value = iterIdValue->second;
    // type of (*treeStructIter) is pair<unsigned int, TreeStruct>
    auto treeStructIter = treeInfo.find(id); // find info for this id
    if (treeStructIter != treeInfo.end()) {
      treeStruct = treeStructIter->second; // info from input map
    } else { // if none found, fill at least parameter index (using 1st IOV...)
      const cond::Time_t run1of1stIov = moduleGroupSelector_->firstRunOfIOV(0);
      const int ind = moduleGroupSelector_->getParameterIndexFromDetId(id, run1of1stIov);
      treeStruct = TreeStruct(ind);
    }
    tree->Fill();
  }
  tree->Write();
  delete file; // tree vanishes with the file...
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
