/// \class SiStripLorentzAngleCalibration
///
/// Calibration of Lorentz angle for the strip tracker, integrated in the
/// alignment algorithms. Note that not all algorithms support this...
///
/// Use one instance for peak and/or one instance for deco mode data.
///
///  \author    : Gero Flucke
///  date       : August 2012
///  $Revision: 1.6.2.14 $
///  $Date: 2013/05/31 08:37:12 $
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
#include "FWCore/Framework/interface/Run.h"
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

#include <vector>
#include <map>
#include <sstream>
#include <cstdio>
#include <memory>
#include <functional>

class SiStripLorentzAngleCalibration : public IntegratedCalibrationBase
{
public:
  /// Constructor
  explicit SiStripLorentzAngleCalibration(const edm::ParameterSet &cfg);

  /// Destructor
  ~SiStripLorentzAngleCalibration() override = default;

  /// How many parameters does this calibration define?
  unsigned int numParameters() const override;

  /// Return non-zero derivatives for x- and y-measurements with their indices by reference.
  /// Return value is their number.
  unsigned int derivatives(std::vector<ValuesIndexPair> &outDerivInds,
                           const TransientTrackingRecHit &hit,
                           const TrajectoryStateOnSurface &tsos,
                           const edm::EventSetup &setup,
                           const EventInfo &eventInfo) const override;

  /// Setting the determined parameter identified by index,
  /// returns false if out-of-bounds, true otherwise.
  bool setParameter(unsigned int index, double value) override;

  /// Setting the determined parameter uncertainty identified by index,
  /// returns false if out-of-bounds, true otherwise.
  bool setParameterError(unsigned int index, double error) override;

  /// Return current value of parameter identified by index.
  /// Returns 0. if index out-of-bounds.
  double getParameter(unsigned int index) const override;

  /// Return current value of parameter identified by index.
  /// Returns 0. if index out-of-bounds or if errors undetermined.
  double getParameterError(unsigned int index) const override;

  // /// Call at beginning of job:
  void beginOfJob(AlignableTracker *tracker,
                  AlignableMuon *muon,
                  AlignableExtras *extras) override;

  /// Call at beginning of run:
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  /// Called at end of a the job of the AlignmentProducer.
  /// Write out determined parameters.
  void endOfJob() override;

private:
  /// Input LorentzAngle values:
  /// - either from EventSetup of first call to derivatives(..)
  /// - or created from files of passed by configuration (i.e. from parallel processing)
  const SiStripLorentzAngle* getLorentzAnglesInput(const align::RunNumber& = 0);
  /// in non-peak mode the effective thickness is reduced...
  double effectiveThickness(const GeomDet *det, int16_t mode, const edm::EventSetup &setup) const;

  /// Determined parameter value for this detId (detId not treated => 0.)
  /// and the given run.
  double getParameterForDetId(unsigned int detId, edm::RunNumber_t run) const;

  void writeTree(const SiStripLorentzAngle *lorentzAngle,
                 const std::map<unsigned int,TreeStruct> &treeInfo, const char *treeName) const;
  SiStripLorentzAngle createFromTree(const char *fileName, const char *treeName) const;

  const std::string readoutModeName_;
  int16_t readoutMode_;
  const bool saveToDB_;
  const std::string recordNameDBwrite_;
  const std::string outFileName_;
  const std::vector<std::string> mergeFileNames_;

  edm::ESWatcher<SiStripLorentzAngleRcd> watchLorentzAngleRcd_;

  std::map<align::RunNumber, SiStripLorentzAngle> cachedLorentzAngleInputs_;
  SiStripLorentzAngle* siStripLorentzAngleInput_{nullptr};
  align::RunNumber currentIOV_{0};

  std::vector<double> parameters_;
  std::vector<double> paramUncertainties_;

  std::unique_ptr<TkModuleGroupSelector> moduleGroupSelector_;
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
unsigned int SiStripLorentzAngleCalibration::numParameters() const
{
  return parameters_.size();
}

//======================================================================
void
SiStripLorentzAngleCalibration::beginRun(const edm::Run& run,
                                         const edm::EventSetup& setup) {

  // no action needed if the LA record didn't change
  if (!(watchLorentzAngleRcd_.check(setup))) return;

  const auto runNumber = run.run();
  auto firstRun = cond::timeTypeSpecs[cond::runnumber].beginValue;

  // avoid infinite loop due to wrap-around of unsigned variable 'i' including
  // arrow from i to zero and a nice smiley ;)
  for (unsigned int i = moduleGroupSelector_->numIovs(); i-->0 ;) {
    const auto firstRunOfIOV = moduleGroupSelector_->firstRunOfIOV(i);
    if (runNumber >= firstRunOfIOV) {
      firstRun = firstRunOfIOV;
      break;
    }
  }

  edm::ESHandle<SiStripLorentzAngle> lorentzAngleHandle;
  const auto& lorentzAngleRcd = setup.get<SiStripLorentzAngleRcd>();
  lorentzAngleRcd.get(readoutModeName_, lorentzAngleHandle);
  if (cachedLorentzAngleInputs_.find(firstRun) == cachedLorentzAngleInputs_.end()) {
    cachedLorentzAngleInputs_.emplace(firstRun, SiStripLorentzAngle(*lorentzAngleHandle));
  } else {
    if (lorentzAngleRcd.validityInterval().first().eventID().run() > firstRun &&
        lorentzAngleHandle->getLorentzAngles()  // only bad if non-identical values
        != cachedLorentzAngleInputs_[firstRun].getLorentzAngles()) { // (comparing maps)
      // Maps are containers sorted by key, but comparison problems may arise from
      // 'floating point comparison' problems (FIXME?)
      throw cms::Exception("BadInput")
        << "Trying to cache SiStripLorentzAngle payload for a run (" << runNumber
        << ") in an IOV (" << firstRun << ") that was already cached.\n"
        << "The following record in your input database tag has an IOV "
        << "boundary that does not match your IOV definition:\n"
        << " - SiStripLorentzAngleRcd '" << lorentzAngleRcd.key().name()
        << "' (since "
        << lorentzAngleRcd.validityInterval().first().eventID().run() << ")\n";
    }
  }

  siStripLorentzAngleInput_ = &(cachedLorentzAngleInputs_[firstRun]);
  currentIOV_ = firstRun;
}

//======================================================================
unsigned int
SiStripLorentzAngleCalibration::derivatives(std::vector<ValuesIndexPair> &outDerivInds,
                                            const TransientTrackingRecHit &hit,
                                            const TrajectoryStateOnSurface &tsos,
                                            const edm::EventSetup &setup,
                                            const EventInfo &eventInfo) const
{
  outDerivInds.clear();

  edm::ESHandle<SiStripLatency> latency;
  setup.get<SiStripLatencyRcd>().get(latency);
  const int16_t mode = latency->singleReadOutMode();
  if (mode == readoutMode_) {
    if (hit.det()) { // otherwise 'constraint hit' or whatever

      const int index = moduleGroupSelector_->getParameterIndexFromDetId(hit.det()->geographicalId(),
                                                                         eventInfo.eventId().run());
      if (index >= 0) { // otherwise not treated
        edm::ESHandle<MagneticField> magneticField;
        setup.get<IdealMagneticFieldRecord>().get(magneticField);
        const GlobalVector bField(magneticField->inTesla(hit.det()->surface().position()));
        const LocalVector bFieldLocal(hit.det()->surface().toLocal(bField));
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
	// GM: sign convention is the same as for pixel LA, i.e. adopt it here, too
        const double xDerivative = bFieldLocal.y() * dZ * -0.5; // parameter is mobility!
        const double yDerivative = bFieldLocal.x() * dZ * 0.5; // parameter is mobility!
        if (xDerivative || yDerivative) { // If field is zero, this is zero: do not return it
          const Values derivs{xDerivative, yDerivative};
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
  const std::vector<int> sdets = {SiStripDetId::TIB, SiStripDetId::TOB,
                                  SiStripDetId::TID, SiStripDetId::TEC};
  moduleGroupSelector_ =
    std::make_unique<TkModuleGroupSelector>(aliTracker, moduleGroupSelCfg_, sdets);

  parameters_.resize(moduleGroupSelector_->getNumberOfParameters(), 0.);
  paramUncertainties_.resize(moduleGroupSelector_->getNumberOfParameters(), 0.);

  edm::LogInfo("Alignment") << "@SUB=SiStripLorentzAngleCalibration" << "Created with name "
                            << this->name() << " for readout mode '" << readoutModeName_
                            << "',\n" << this->numParameters() << " parameters to be determined."
                            << "\nsaveToDB = " << saveToDB_
                            << "\n outFileName = " << outFileName_
                            << "\n N(merge files) = " << mergeFileNames_.size()
                            << "\n number of IOVs = " << moduleGroupSelector_->numIovs();

  if (!mergeFileNames_.empty()) {
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
  const std::string treeName{this->name() + '_' + readoutModeName_ + '_'};
  std::vector<const SiStripLorentzAngle*> inputs{};
  inputs.reserve(moduleGroupSelector_->numIovs());
  for (unsigned int iIOV = 0; iIOV < moduleGroupSelector_->numIovs(); ++iIOV) {
    const auto firstRunOfIOV = moduleGroupSelector_->firstRunOfIOV(iIOV);
    inputs.push_back(this->getLorentzAnglesInput(firstRunOfIOV)); // never NULL
    this->writeTree(inputs.back(), treeInfo,
                    (treeName + "input_" + std::to_string(firstRunOfIOV)).c_str()); // empty treeInfo for input...

    if (inputs.back()->getLorentzAngles().empty()) {
      edm::LogError("Alignment") << "@SUB=SiStripLorentzAngleCalibration::endOfJob"
                                 << "Input Lorentz angle map is empty ('"
                                 << readoutModeName_ << "' mode), skip writing output!";
      return;
    }
  }

  const unsigned int nonZeroParamsOrErrors =   // Any determined value?
    count_if (parameters_.begin(), parameters_.end(), [](auto c){return c!=0.;})
    + count_if(paramUncertainties_.begin(), paramUncertainties_.end(),
               [](auto c){return c!=0.;});

  for (unsigned int iIOV = 0; iIOV < moduleGroupSelector_->numIovs(); ++iIOV) {
    auto firstRunOfIOV = static_cast<cond::Time_t>(moduleGroupSelector_->firstRunOfIOV(iIOV));
    SiStripLorentzAngle output{};
    // Loop on map of values from input and add (possible) parameter results
    for (const auto& iterIdValue: inputs[iIOV]->getLorentzAngles()) {
      // type of 'iterIdValue' is pair<unsigned int, float>
      const auto detId = iterIdValue.first; // key of map is DetId
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
      auto value = iterIdValue.second + static_cast<float>(param);
      output.putLorentzAngle(detId, value);
      const int paramIndex = moduleGroupSelector_->getParameterIndexFromDetId(detId,firstRunOfIOV);
      treeInfo[detId] = TreeStruct(param, this->getParameterError(paramIndex), paramIndex);
    }

    if (saveToDB_ || nonZeroParamsOrErrors != 0) { // Skip writing mille jobs...
      this->writeTree(&output, treeInfo, (treeName + Form("result_%lld", firstRunOfIOV)).c_str());
    }

    if (saveToDB_) { // If requested, write out to DB
      edm::Service<cond::service::PoolDBOutputService> dbService;
      if (dbService.isAvailable()) {
        dbService->writeOne(&output, firstRunOfIOV, recordNameDBwrite_);
      } else {
        edm::LogError("BadConfig") << "@SUB=SiStripLorentzAngleCalibration::endOfJob"
                                   << "No PoolDBOutputService available, but saveToDB true!";
      }
    }
  } // end loop on IOVs
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
const SiStripLorentzAngle*
SiStripLorentzAngleCalibration::getLorentzAnglesInput(const align::RunNumber& run)
{
  const auto& resolvedRun = run > 0 ? run : currentIOV_;
  // For parallel processing in Millepede II, create SiStripLorentzAngle
  // from info stored in files of parallel jobs and check that they are identical.
  // If this job has run on events, still check that LA is identical to the ones
  // from mergeFileNames_.
  const std::string treeName{this->name()+"_"+readoutModeName_+"_input_"+
                             std::to_string(resolvedRun)};
  for (const auto& iFile: mergeFileNames_) {
    auto la = this->createFromTree(iFile.c_str(), treeName.c_str());
    // siStripLorentzAngleInput_ could be non-null from previous file of this loop
    // or from checkLorentzAngleInput(..) when running on data in this job as well
    if (!siStripLorentzAngleInput_ || siStripLorentzAngleInput_->getLorentzAngles().empty()) {
      cachedLorentzAngleInputs_[resolvedRun] = la;
      siStripLorentzAngleInput_ = &(cachedLorentzAngleInputs_[resolvedRun]);
      currentIOV_ = resolvedRun;
    } else {
      // FIXME: about comparison of maps see comments in checkLorentzAngleInput
      if (!la.getLorentzAngles().empty() && // single job might not have got events
          la.getLorentzAngles() != siStripLorentzAngleInput_->getLorentzAngles()) {
        // Throw exception instead of error?
        edm::LogError("NoInput") << "@SUB=SiStripLorentzAngleCalibration::getLorentzAnglesInput"
                                 << "Different input values from tree " << treeName
                                 << " in file " << iFile << ".";

      }
    }
  }

  if (!siStripLorentzAngleInput_) { // no files nor ran on events
    // [] operator default-constructs an empty SiStripLorentzAngle object in place:
    siStripLorentzAngleInput_ = &(cachedLorentzAngleInputs_[resolvedRun]);
    currentIOV_ = resolvedRun;
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
SiStripLorentzAngle
SiStripLorentzAngleCalibration::createFromTree(const char *fileName, const char *treeName) const
{
  // Check for file existence on your own to work around
  // https://hypernews.cern.ch/HyperNews/CMS/get/swDevelopment/2715.html:
  TFile* file = nullptr;
  FILE* testFile = fopen(fileName,"r");
  if (testFile) {
    fclose(testFile);
    file = TFile::Open(fileName, "READ");
  } // else not existing, see error below

  TTree *tree = nullptr;
  if (file) file->GetObject(treeName, tree);

  SiStripLorentzAngle result{};
  if (tree) {
    unsigned int id = 0;
    float value = 0.;
    tree->SetBranchAddress("detId", &id);
    tree->SetBranchAddress("value", &value);

    const Long64_t nEntries = tree->GetEntries();
    for (Long64_t iEntry = 0; iEntry < nEntries; ++iEntry) {
      tree->GetEntry(iEntry);
      result.putLorentzAngle(id, value);
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
