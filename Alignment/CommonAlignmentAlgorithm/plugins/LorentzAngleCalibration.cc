/// \class LorentzAngleCalibration
///
/// Calibration of Lorentz angle (and strip deco mode backplane corrections?)
/// for the tracker, integrated in the alignment algorithms.
///
/// Note that not all algorithms support this...
///
///  \author    : Gero Flucke
///  date       : August 2012
///  $Revision: 1.1 $
///  $Date: 2012/08/10 09:10:53 $
///  (last update by $Author: flucke $)

#include "Alignment/CommonAlignmentAlgorithm/interface/IntegratedCalibrationBase.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"

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

// #include <iostream>
#include <vector>
#include <sstream>

class LorentzAngleCalibration : public IntegratedCalibrationBase
{
public:
  /// Constructor
  explicit LorentzAngleCalibration(const edm::ParameterSet &cfg);
  
  /// Destructor
  virtual ~LorentzAngleCalibration();

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
  /// Determined parameter value for this detId (detId not treated => 0.).
  double getParameterForDetId(unsigned int detId) const;
  /// Index of parameter for given detId (detId not treated => < 0)
  int getParameterIndexFromDetId(unsigned int detId) const;

  void writeTree(const SiStripLorentzAngle *lorentzAngle, const char *treeName) const;
  SiStripLorentzAngle* createFromTree(const char *fileName, const char *treeName) const;

  const std::string readoutModeName_;
  int16_t readoutMode_;
  const bool saveToDB_;
  const std::string outFileName_;
  const std::vector<std::string> mergeFileNames_;

  edm::ESWatcher<SiStripLorentzAngleRcd> watchLorentzAngleRcd_;

  // const AlignableTracker *alignableTracker_;
  SiStripLorentzAngle *siStripLorentzAngleInput_;
  std::vector<double> parameters_;
  std::vector<double> paramUncertainties_;
};

//======================================================================
//======================================================================
//======================================================================

LorentzAngleCalibration::LorentzAngleCalibration(const edm::ParameterSet &cfg)
  : IntegratedCalibrationBase(cfg),
    readoutModeName_(cfg.getParameter<std::string>("readoutMode")),
    saveToDB_(cfg.getParameter<bool>("saveToDB")),
    outFileName_(cfg.getParameter<std::string>("treeFile")),
    mergeFileNames_(cfg.getParameter<std::vector<std::string> >("mergeTreeFiles")),
    //    alignableTracker_(0),
    siStripLorentzAngleInput_(0)
{
  // SiStripLatency::singleReadOutMode() returns
  // 1: all in peak, 0: all in deco, -1: mixed state
  // (in principle one could of treat even mixed state APV by APV...)
  if (readoutModeName_ == "peak") {
    readoutMode_ = 1;
  } else if (readoutModeName_ == "deconvolution") {
    readoutMode_ = 0;
  } else {
    throw cms::Exception("BadConfig")
	  << "LorentzAngleCalibration:\n" << "Unknown mode '" 
	  << readoutModeName_ << "', should be 'peak' or 'deconvolution' .\n";
  }

  // FIXME: Which granularity, leading to how many parameters?
  parameters_.resize(2, 0.); // currently two parameters (TIB, TOB), start value 0.
  paramUncertainties_.resize(2, 0.); // dito for errors

  edm::LogInfo("Alignment") << "@SUB=LorentzAngleCalibration" << "Created with name "
                            << this->name() << " for readout mode '" << readoutModeName_
			    << "',\n" << this->numParameters() << " parameters to be determined."
                            << "\nsaveToDB = " << saveToDB_
                            << "\n outFileName = " << outFileName_
                            << "\n N(merge files) = " << mergeFileNames_.size();
  if (mergeFileNames_.size()) {
    edm::LogInfo("Alignment") << "@SUB=LorentzAngleCalibration"
                              << "First file to merge: " << mergeFileNames_[0];
  }
}
  
//======================================================================
LorentzAngleCalibration::~LorentzAngleCalibration()
{
  //  std::cout << "Destroy LorentzAngleCalibration named " << this->name() << std::endl;
  delete siStripLorentzAngleInput_;
}

//======================================================================
unsigned int LorentzAngleCalibration::numParameters() const
{
  return parameters_.size();
}

//======================================================================
unsigned int LorentzAngleCalibration::derivatives(std::vector<ValuesIndexPair> &outDerivInds,
						  const TransientTrackingRecHit &hit,
						  const TrajectoryStateOnSurface &tsos,
						  const edm::EventSetup &setup,
						  const EventInfo &eventInfo) const
{
  // ugly const-cast:
  // But it is either only first initialisation or throwing an exception...
  const_cast<LorentzAngleCalibration*>(this)->checkLorentzAngleInput(setup, eventInfo);

  outDerivInds.clear();

  edm::ESHandle<SiStripLatency> latency;  
  setup.get<SiStripLatencyRcd>().get(latency);
  const int16_t mode = latency->singleReadOutMode();
  if(mode == readoutMode_) {
    if (hit.det()) { // otherwise 'constraint hit' or whatever
      const int index = this->getParameterIndexFromDetId(hit.det()->geographicalId());
      if (index >= 0) { // otherwise not treated
        edm::ESHandle<MagneticField> magneticField;
        setup.get<IdealMagneticFieldRecord>().get(magneticField);
        const GlobalVector bField(magneticField->inTesla(hit.det()->surface().position()));
        const LocalVector bFieldLocal(hit.det()->surface().toLocal(bField));
        const double dZ = hit.det()->surface().bounds().thickness(); // it is a float only...
        // shift due to LA: dx = tan(LA) * dz = mobility * B_y * dz
        const double xDerivative = bFieldLocal.y() * dZ; // parameter is mobility!
        if (xDerivative) { // If field is zero, this is zero: do not return it
          const Values derivs(xDerivative, 0.); // yDerivative = 0.
          outDerivInds.push_back(ValuesIndexPair(derivs, index));
        }
      }
    } else {
      edm::LogWarning("Alignment") << "@SUB=LorentzAngleCalibration::derivatives2"
                                   << "Hit without GeomDet, skip!";
    }
  } else if (mode != 0 && mode != 1) { // warn only if unknown/mixed mode  
    edm::LogWarning("Alignment") << "@SUB=LorentzAngleCalibration::derivatives3"
                                 << "Readout mode is " << mode << ", but looking for "
                                 << readoutMode_ << " (" << readoutModeName_ << ").";
  }
  
  return outDerivInds.size();
}

//======================================================================
bool LorentzAngleCalibration::setParameter(unsigned int index, double value)
{
  if (index >= parameters_.size()) {
    return false;
  } else {
    parameters_[index] = value;
    return true;
  }
}

//======================================================================
bool LorentzAngleCalibration::setParameterError(unsigned int index, double error)
{
  if (index >= paramUncertainties_.size()) {
    return false;
  } else {
    paramUncertainties_[index] = error;
    return true;
  }
}

//======================================================================
double LorentzAngleCalibration::getParameter(unsigned int index) const
{
  //   if (index >= parameters_.size()) {
  //     return 0.;
  //   } else {
  //     return parameters_[index];
  //   }
  return (index >= parameters_.size() ? 0. : parameters_[index]);
}

//======================================================================
double LorentzAngleCalibration::getParameterError(unsigned int index) const
{
  //   if (index >= paramUncertainties_.size()) {
  //     return 0.;
  //   } else {
  //     return paramUncertainties_[index];
  //   }
  return (index >= paramUncertainties_.size() ? 0. : paramUncertainties_[index]);
}

// //======================================================================
// void LorentzAngleCalibration::beginOfJob(const AlignableTracker *tracker,
//                                          const AlignableMuon */*muon*/,
//                                          const AlignableExtras */*extras*/)
// {
//   alignableTracker_ = tracker;
// }


//======================================================================
void LorentzAngleCalibration::endOfJob()
{
  // loginfo output
  std::ostringstream out;
  out << "Parameter results for readout mode '" << readoutModeName_ << "'\n";
  for (unsigned int iPar = 0; iPar < parameters_.size(); ++iPar) {
    out << iPar << ": " << parameters_[iPar] << " +- " << paramUncertainties_[iPar] << "\n";
  }
  edm::LogInfo("Alignment") << "@SUB=LorentzAngleCalibration::endOfJob" << out.str();

  // now write 'input' tree
  const SiStripLorentzAngle *input = this->getLorentzAnglesInput(); // never NULL
  const std::string treeName(this->name() + '_' + readoutModeName_ + '_');
  this->writeTree(input, (treeName + "input").c_str());

  // If output has IOVs (see below), would have to loop here on runs...
  SiStripLorentzAngle *output = new SiStripLorentzAngle;
  // Loop on map of values from input and add (possible) parameter results
  for (auto iterIdValue = input->getLorentzAngles().begin();
       iterIdValue != input->getLorentzAngles().end(); ++iterIdValue) {
    // type of (*iterIdValue) is pair<unsigned int, float>
    const unsigned int detId = iterIdValue->first; // key of map is DetId
    const float value = iterIdValue->second + this->getParameterForDetId(detId); // pass run?
    output->putLorentzAngle(detId, value); // put result in output
  }

  // Write this even for mille jobs?
  this->writeTree(output, (treeName + "result").c_str()); // add run?

  if (saveToDB_) { // If requested, write out to DB 
    edm::Service<cond::service::PoolDBOutputService> dbService;
    if (dbService.isAvailable()) {
      // 1 is the run number - could we add a label, i.e. readoutModeName_ (FIXME)?
      dbService->writeOne(output, 1, "SiStripLorentzAngleRcd");
      // If time dependence is requested, i.e. if you have various periods, do a loop.
      // Start with 1 and for the second IOV state its first run number etc.

      // no 'delete output;': writeOne(..) took over ownership
    } else {
      delete output;
      edm::LogError("BadConfig") << "@SUB=LorentzAngleCalibration::endOfJob"
				 << "No PoolDBOutputService available, but saveToDB true!";
    }
  } else {
    delete output;
  }

  // Here would be the end of the loop on run numbers...
}

//======================================================================
bool LorentzAngleCalibration::checkLorentzAngleInput(const edm::EventSetup &setup,
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
	  << "LorentzAngleCalibration::checkLorentzAngleInput:\n"
	  << "Content of SiStripLorentzAngle changed at run " << eventInfo.eventId_.run()
	  << ", but algorithm expects constant input!\n";
	return false; // not reached...
      }
    }
  }
  
  return true;
}

//======================================================================
const SiStripLorentzAngle* LorentzAngleCalibration::getLorentzAnglesInput()
{
  // For parallel processing in Millepede II, create SiStripLorentzAngle
  // from info stored in files of parallel jobs and check that !

  const std::string treeName(((this->name() + '_') += readoutModeName_) += "_input");
  for (auto iFile = mergeFileNames_.begin(); iFile != mergeFileNames_.end(); ++iFile) {
    SiStripLorentzAngle* la = this->createFromTree(iFile->c_str(), treeName.c_str());
    // siStripLorentzAngleInput_ could be non-null from previous file of this loop
    // or from checkLorentzAngleInput(..) when runnin on data in this job as well
    if (!siStripLorentzAngleInput_ || siStripLorentzAngleInput_->getLorentzAngles().empty()) {
      delete siStripLorentzAngleInput_; // NULL or empty
      siStripLorentzAngleInput_ = la;
    } else {
      // FIXME: about comparison of maps see comments in checkLorentzAngleInput
      if (!la->getLorentzAngles().empty() && // single job might not have got events
          la->getLorentzAngles() != siStripLorentzAngleInput_->getLorentzAngles()) {
        // Throw exception instead of error?
        edm::LogError("NoInput") << "@SUB=LorentzAngleCalibration::getLorentzAnglesInput"
                                 << "Different input values from tree " << treeName
                                 << " in file " << *iFile << ".";
        
      }
      delete la;
    }
  }

  if (!siStripLorentzAngleInput_) { // no files nor ran on events
    siStripLorentzAngleInput_ = new SiStripLorentzAngle;
    edm::LogError("NoInput") << "@SUB=LorentzAngleCalibration::getLorentzAnglesInput"
			     << "No input, create an empty one ('" << readoutModeName_ << "' mode)!";
  } else if (siStripLorentzAngleInput_->getLorentzAngles().empty()) {
    edm::LogError("NoInput") << "@SUB=LorentzAngleCalibration::getLorentzAnglesInput"
			     << "Empty result ('" << readoutModeName_ << "' mode)!";
  }

  return siStripLorentzAngleInput_;
}

//======================================================================
double LorentzAngleCalibration::getParameterForDetId(unsigned int detId) const
{
  const int index = this->getParameterIndexFromDetId(detId);

  return (index < 0 ? 0. : parameters_[index]);
}

//======================================================================
int LorentzAngleCalibration::getParameterIndexFromDetId(unsigned int detId) const
{
  // Return the index of the parameter that is used for this DetId.
  // If this DetId is not treated, return values < 0.
  
  // FIXME: Extend to configurable granularity?
  const SiStripDetId id(detId);
  if (id.det() == DetId::Tracker) {
    if      (id.subDetector() == SiStripDetId::TIB) return 0;
    else if (id.subDetector() == SiStripDetId::TOB) return 1;
  }

  return -1;
}

//======================================================================
void LorentzAngleCalibration::writeTree(const SiStripLorentzAngle *lorentzAngle,
					const char *treeName) const
{
  if (!lorentzAngle) return;

  TFile* file = TFile::Open(outFileName_.c_str(), "UPDATE");
  if (!file) {
    edm::LogError("BadConfig") << "@SUB=LorentzAngleCalibration::writeTree"
			       << "Could not open file '" << outFileName_ << "'.";
    return;
  }

  TTree *tree = new TTree(treeName, treeName);
  unsigned int id = 0;
  double value = 0.;
  tree->Branch("detId", &id, "detId/i");
  tree->Branch("value", &value, "value/D");

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
LorentzAngleCalibration::createFromTree(const char *fileName, const char *treeName) const
{
  TTree *tree = 0;
  TFile* file = TFile::Open(fileName, "READ");
  if (file) file->GetObject(treeName, tree);

  SiStripLorentzAngle *result = 0;
  if (tree) {
    unsigned int id = 0;
    double value = 0.;
    tree->SetBranchAddress("detId", &id);
    tree->SetBranchAddress("value", &value);

    result = new SiStripLorentzAngle;
    const Long64_t nEntries = tree->GetEntries();
    for (Long64_t iEntry = 0; iEntry < nEntries; ++iEntry) {
      tree->GetEntry(iEntry);
      result->putLorentzAngle(id, value);
    }
  } else {
    edm::LogError("Alignment") << "@SUB=LorentzAngleCalibration::createFromTree"
			       << "Could not get TTree '" << treeName << "' from file '"
			       << fileName << "'.";
  }

  delete file; // tree will vanish with file
  return result;
}


//======================================================================
//======================================================================
// Plugin definition

#include "Alignment/CommonAlignmentAlgorithm/interface/IntegratedCalibrationPluginFactory.h"

DEFINE_EDM_PLUGIN(IntegratedCalibrationPluginFactory,
		   LorentzAngleCalibration, "LorentzAngleCalibration");
