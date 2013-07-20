/// \class SiPixelLorentzAngleCalibration
///
/// Calibration of Lorentz angle for the pixel,
/// integrated in the alignment algorithms.
///
/// Note that not all algorithms support this...
///
///  \author    : Gero Flucke
///  date       : September 2012
///  $Revision: 1.6 $
///  $Date: 2013/05/31 12:13:40 $
///  (last update by $Author: flucke $)

#include "Alignment/CommonAlignmentAlgorithm/interface/IntegratedCalibrationBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/TkModuleGroupSelector.h"
// include 'locally':
#include "TreeStruct.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "TTree.h"
#include "TFile.h"
#include "TString.h"

#include <boost/assign/list_of.hpp>
#include <vector>
#include <map>
#include <sstream>
#include <list>
#include <utility>
#include <set>

#include <functional>

class SiPixelLorentzAngleCalibration : public IntegratedCalibrationBase
{
public:
  /// Constructor
  explicit SiPixelLorentzAngleCalibration(const edm::ParameterSet &cfg);
  
  /// Destructor
  virtual ~SiPixelLorentzAngleCalibration();

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
  /// Return 0. if index out-of-bounds.
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
  /// If called the first time, fill 'siPixelLorentzAngleInput_',
  /// later check that LorentzAngle has not changed.
  bool checkLorentzAngleInput(const edm::EventSetup &setup, const EventInfo &eventInfo);
  /// Input LorentzAngle values:
  /// - either from EventSetup of first call to derivatives(..)
  /// - or created from files of passed by configuration (i.e. from parallel processing)
  const SiPixelLorentzAngle* getLorentzAnglesInput();

  /// Determined parameter value for this detId (detId not treated => 0.)
  /// and the given run.
  double getParameterForDetId(unsigned int detId, edm::RunNumber_t run) const;

  void writeTree(const SiPixelLorentzAngle *lorentzAngle,
		 const std::map<unsigned int,TreeStruct>& treeInfo, const char *treeName) const;
  SiPixelLorentzAngle* createFromTree(const char *fileName, const char *treeName) const;
  
  const bool saveToDB_;
  const std::string recordNameDBwrite_;
  const std::string outFileName_;
  const std::vector<std::string> mergeFileNames_;

  edm::ESWatcher<SiPixelLorentzAngleRcd> watchLorentzAngleRcd_;

  // const AlignableTracker *alignableTracker_;
  SiPixelLorentzAngle *siPixelLorentzAngleInput_;
  std::vector<double> parameters_;
  std::vector<double> paramUncertainties_;

  TkModuleGroupSelector *moduleGroupSelector_;
  const edm::ParameterSet moduleGroupSelCfg_;
};

//======================================================================
//======================================================================
//======================================================================

SiPixelLorentzAngleCalibration::SiPixelLorentzAngleCalibration(const edm::ParameterSet &cfg)
  : IntegratedCalibrationBase(cfg),
    saveToDB_(cfg.getParameter<bool>("saveToDB")),
    recordNameDBwrite_(cfg.getParameter<std::string>("recordNameDBwrite")),
    outFileName_(cfg.getParameter<std::string>("treeFile")),
    mergeFileNames_(cfg.getParameter<std::vector<std::string> >("mergeTreeFiles")),
    siPixelLorentzAngleInput_(0),
    moduleGroupSelector_(0),
    moduleGroupSelCfg_(cfg.getParameter<edm::ParameterSet>("LorentzAngleModuleGroups"))
{

}

//======================================================================
SiPixelLorentzAngleCalibration::~SiPixelLorentzAngleCalibration()
{
  delete moduleGroupSelector_;
  //  std::cout << "Destroy SiPixelLorentzAngleCalibration named " << this->name() << std::endl;
  delete siPixelLorentzAngleInput_;
}

//======================================================================
unsigned int SiPixelLorentzAngleCalibration::numParameters() const
{
  return parameters_.size();
}

//======================================================================
unsigned int
SiPixelLorentzAngleCalibration::derivatives(std::vector<ValuesIndexPair> &outDerivInds,
					    const TransientTrackingRecHit &hit,
					    const TrajectoryStateOnSurface &tsos,
					    const edm::EventSetup &setup,
					    const EventInfo &eventInfo) const
{
  // ugly const-cast:
  // But it is either only first initialisation or throwing an exception...
  const_cast<SiPixelLorentzAngleCalibration*>(this)->checkLorentzAngleInput(setup, eventInfo);

  outDerivInds.clear();

  if (hit.det()) { // otherwise 'constraint hit' or whatever
    
    const int index = moduleGroupSelector_->getParameterIndexFromDetId(hit.det()->geographicalId(),
                                                                       eventInfo.eventId_.run());
    if (index >= 0) { // otherwise not treated
      edm::ESHandle<MagneticField> magneticField;
      setup.get<IdealMagneticFieldRecord>().get(magneticField);
      const GlobalVector bField(magneticField->inTesla(hit.det()->surface().position()));
      const LocalVector bFieldLocal(hit.det()->surface().toLocal(bField));
      const double dZ = hit.det()->surface().bounds().thickness(); // it is a float only...
      // shift due to LA: dx = tan(LA) * dz/2 = mobility * B_y * dz/2,
      // '-' since we have derivative of the residual r = trk -hit
      const double xDerivative = bFieldLocal.y() * dZ * -0.5; // parameter is mobility!
      // FIXME: Have to treat that for FPIX yDerivative != 0., due to higher order effects! 
      if (xDerivative) { // If field is zero, this is zero: do not return it
	const Values derivs(xDerivative, 0.); // yDerivative = 0.
	outDerivInds.push_back(ValuesIndexPair(derivs, index));
      }
    }
  } else {
    edm::LogWarning("Alignment") << "@SUB=SiPixelLorentzAngleCalibration::derivatives2"
				 << "Hit without GeomDet, skip!";
  }
  
  return outDerivInds.size();
}

//======================================================================
bool SiPixelLorentzAngleCalibration::setParameter(unsigned int index, double value)
{
  if (index >= parameters_.size()) {
    return false;
  } else {
    parameters_[index] = value;
    return true;
  }
}

//======================================================================
bool SiPixelLorentzAngleCalibration::setParameterError(unsigned int index, double error)
{
  if (index >= paramUncertainties_.size()) {
    return false;
  } else {
    paramUncertainties_[index] = error;
    return true;
  }
}

//======================================================================
double SiPixelLorentzAngleCalibration::getParameter(unsigned int index) const
{
  return (index >= parameters_.size() ? 0. : parameters_[index]);
}

//======================================================================
double SiPixelLorentzAngleCalibration::getParameterError(unsigned int index) const
{
  return (index >= paramUncertainties_.size() ? 0. : paramUncertainties_[index]);
}




//======================================================================
void SiPixelLorentzAngleCalibration::beginOfJob(AlignableTracker *aliTracker,
                                                AlignableMuon * /*aliMuon*/,
                                                AlignableExtras * /*aliExtras*/)
{
  //specify the sub-detectors for which the LA is determined
  const std::vector<int> sdets = boost::assign::list_of(PixelSubdetector::PixelBarrel)(PixelSubdetector::PixelEndcap);
  
  moduleGroupSelector_ = new TkModuleGroupSelector(aliTracker, moduleGroupSelCfg_, sdets);

  parameters_.resize(moduleGroupSelector_->getNumberOfParameters(), 0.);
  paramUncertainties_.resize(moduleGroupSelector_->getNumberOfParameters(), 0.);
  
  edm::LogInfo("Alignment") << "@SUB=SiPixelLorentzAngleCalibration" << "Created with name "
                            << this->name() << "',\n" << this->numParameters() << " parameters to be determined,"
                            << "\n saveToDB = " << saveToDB_
                            << "\n outFileName = " << outFileName_
                            << "\n N(merge files) = " << mergeFileNames_.size()
                            << "\n number of IOVs = " << moduleGroupSelector_->numIovs();
     
  if (mergeFileNames_.size()) {
    edm::LogInfo("Alignment") << "@SUB=SiPixelLorentzAngleCalibration"
                              << "First file to merge: " << mergeFileNames_[0];
  }
}


//======================================================================
void SiPixelLorentzAngleCalibration::endOfJob()
{
  // loginfo output
  std::ostringstream out;
  out << "Parameter results\n";
  for (unsigned int iPar = 0; iPar < parameters_.size(); ++iPar) {
    out << iPar << ": " << parameters_[iPar] << " +- " << paramUncertainties_[iPar] << "\n";
  }
  edm::LogInfo("Alignment") << "@SUB=SiPixelLorentzAngleCalibration::endOfJob" << out.str();

  std::map<unsigned int, TreeStruct> treeInfo; // map of TreeStruct for each detId

  // now write 'input' tree
  const SiPixelLorentzAngle *input = this->getLorentzAnglesInput(); // never NULL
  const std::string treeName(this->name() + '_');
  this->writeTree(input, treeInfo, (treeName + "input").c_str()); // empty treeInfo for input...

  if (input->getLorentzAngles().empty()) {
    edm::LogError("Alignment") << "@SUB=SiPixelLorentzAngleCalibration::endOfJob"
			       << "Input Lorentz angle map is empty, skip writing output!";
    return;
  }

  const unsigned int nonZeroParamsOrErrors =   // Any determined value?
    count_if (parameters_.begin(), parameters_.end(), std::bind2nd(std::not_equal_to<double>(),0.))
    + count_if(paramUncertainties_.begin(), paramUncertainties_.end(),
               std::bind2nd(std::not_equal_to<double>(), 0.));

  for (unsigned int iIOV = 0; iIOV < moduleGroupSelector_->numIovs(); ++iIOV) {
    //  for (unsigned int iIOV = 0; iIOV < 1; ++iIOV) {   // For writing out the modified values
    cond::Time_t firstRunOfIOV = moduleGroupSelector_->firstRunOfIOV(iIOV);
    SiPixelLorentzAngle *output = new SiPixelLorentzAngle;
    // Loop on map of values from input and add (possible) parameter results
    for (auto iterIdValue = input->getLorentzAngles().begin();
	 iterIdValue != input->getLorentzAngles().end(); ++iterIdValue) {
      // type of (*iterIdValue) is pair<unsigned int, float>
      const unsigned int detId = iterIdValue->first; // key of map is DetId
      const double param = this->getParameterForDetId(detId, firstRunOfIOV);
      // put result in output, i.e. sum of input and determined parameter, but the nasty
      // putLorentzAngle(..) takes float by reference - not even const reference:
      float value = iterIdValue->second + param;
      output->putLorentzAngle(detId, value);
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
	edm::LogError("BadConfig") << "@SUB=SiPixelLorentzAngleCalibration::endOfJob"
				   << "No PoolDBOutputService available, but saveToDB true!";
      }
    } else {
      delete output;
    }
  } // end loop on IOVs

}

//======================================================================
bool SiPixelLorentzAngleCalibration::checkLorentzAngleInput(const edm::EventSetup &setup,
							    const EventInfo &eventInfo)
{
  edm::ESHandle<SiPixelLorentzAngle> lorentzAngleHandle;
  if (!siPixelLorentzAngleInput_) {
    setup.get<SiPixelLorentzAngleRcd>().get(lorentzAngleHandle);
    siPixelLorentzAngleInput_ = new SiPixelLorentzAngle(*lorentzAngleHandle);
  } else {
    if (watchLorentzAngleRcd_.check(setup)) { // new IOV of input
      setup.get<SiPixelLorentzAngleRcd>().get(lorentzAngleHandle);
      if (lorentzAngleHandle->getLorentzAngles() // but only bad if non-identical values
	  != siPixelLorentzAngleInput_->getLorentzAngles()) { // (comparing maps)
	// Maps are containers sorted by key, but comparison problems may arise from
	// 'floating point comparison' problems (FIXME?)
	throw cms::Exception("BadInput")
	  << "SiPixelLorentzAngleCalibration::checkLorentzAngleInput:\n"
	  << "Content of SiPixelLorentzAngle changed at run " << eventInfo.eventId_.run()
	  << ", but algorithm expects constant input!\n";
	return false; // not reached...
      }
    }
  }
  
  return true;
}

//======================================================================
const SiPixelLorentzAngle* SiPixelLorentzAngleCalibration::getLorentzAnglesInput()
{
  // For parallel processing in Millepede II, create SiPixelLorentzAngle
  // from info stored in files of parallel jobs and check that they are identical.
  // If this job has run on data, still check that LA is identical to the ones
  // from mergeFileNames_.
  const std::string treeName(this->name() + "_input");
  for (auto iFile = mergeFileNames_.begin(); iFile != mergeFileNames_.end(); ++iFile) {
    SiPixelLorentzAngle* la = this->createFromTree(iFile->c_str(), treeName.c_str());
    // siPixelLorentzAngleInput_ could be non-null from previous file of this loop
    // or from checkLorentzAngleInput(..) when running on data in this job as well
    if (!siPixelLorentzAngleInput_ || siPixelLorentzAngleInput_->getLorentzAngles().empty()) {
      delete siPixelLorentzAngleInput_; // NULL or empty
      siPixelLorentzAngleInput_ = la;
    } else {
      // FIXME: about comparison of maps see comments in checkLorentzAngleInput
      if (la && !la->getLorentzAngles().empty() && // single job might not have got events
          la->getLorentzAngles() != siPixelLorentzAngleInput_->getLorentzAngles()) {
        // Throw exception instead of error?
        edm::LogError("NoInput") << "@SUB=SiPixelLorentzAngleCalibration::getLorentzAnglesInput"
                                 << "Different input values from tree " << treeName
                                 << " in file " << *iFile << ".";
        
      }
      delete la;
    }
  }

  if (!siPixelLorentzAngleInput_) { // no files nor ran on events
    siPixelLorentzAngleInput_ = new SiPixelLorentzAngle;
    edm::LogError("NoInput") << "@SUB=SiPixelLorentzAngleCalibration::getLorentzAnglesInput"
                             << "No input, create an empty one!";
  } else if (siPixelLorentzAngleInput_->getLorentzAngles().empty()) {
    edm::LogError("NoInput") << "@SUB=SiPixelLorentzAngleCalibration::getLorentzAnglesInput"
                             << "Empty result!";
  }

  return siPixelLorentzAngleInput_;
}

//======================================================================
double SiPixelLorentzAngleCalibration::getParameterForDetId(unsigned int detId,
							    edm::RunNumber_t run) const
{
  const int index = moduleGroupSelector_->getParameterIndexFromDetId(detId, run);
  return (index < 0 ? 0. : parameters_[index]);
}

//======================================================================
void SiPixelLorentzAngleCalibration::writeTree(const SiPixelLorentzAngle *lorentzAngle,
					       const std::map<unsigned int,TreeStruct> &treeInfo, 
					       const char *treeName) const
{
  if (!lorentzAngle) return;

  TFile* file = TFile::Open(outFileName_.c_str(), "UPDATE");
  if (!file) {
    edm::LogError("BadConfig") << "@SUB=SiPixelLorentzAngleCalibration::writeTree"
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
  delete file; // tree vanishes with the file... (?)

}

//======================================================================
SiPixelLorentzAngle* 
SiPixelLorentzAngleCalibration::createFromTree(const char *fileName, const char *treeName) const
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

  SiPixelLorentzAngle *result = 0;
  if (tree) {
    unsigned int id = 0;
    float value = 0.;
    tree->SetBranchAddress("detId", &id);
    tree->SetBranchAddress("value", &value);

    result = new SiPixelLorentzAngle;
    const Long64_t nEntries = tree->GetEntries();
    for (Long64_t iEntry = 0; iEntry < nEntries; ++iEntry) {
      tree->GetEntry(iEntry);
      result->putLorentzAngle(id, value);
    }
  } else { // Warning only since could be parallel job on no events.
    edm::LogWarning("Alignment") << "@SUB=SiPixelLorentzAngleCalibration::createFromTree"
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
                  SiPixelLorentzAngleCalibration, "SiPixelLorentzAngleCalibration");
