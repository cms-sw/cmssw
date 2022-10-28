/**
 * \file MillePedeAlignmentAlgorithm.cc
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.80 $
 *  $Date: 2013/01/07 20:21:32 $
 *  (last update by $Author: wmtan $)
 */

#include "MillePedeAlignmentAlgorithm.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeFileReader.h"
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeMonitor.h"
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariables.h"
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariablesIORoot.h"
#include "Alignment/MillePedeAlignmentAlgorithm/src/Mille.h"        // 'unpublished' interface located in src
#include "Alignment/MillePedeAlignmentAlgorithm/src/PedeSteerer.h"  // ditto
#include "Alignment/MillePedeAlignmentAlgorithm/src/PedeReader.h"   // ditto
#include "Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerBase.h"
#include "Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerPluginFactory.h"

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryPlugin.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORoot.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/IntegratedCalibrationBase.h"

#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"

// includes to make known that they inherit from Alignable:
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/CommonAlignment/interface/AlignableExtras.h"

#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/Alignment/interface/TkFittedLasBeam.h"
#include "Alignment/LaserAlignment/interface/TsosVectorCollection.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <sys/stat.h>

#include <TMath.h>
typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;
typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;
typedef TrajectoryFactoryBase::ReferenceTrajectoryCollection RefTrajColl;

// Includes for PXB survey
#include "Alignment/SurveyAnalysis/interface/SurveyPxbImage.h"
#include "Alignment/SurveyAnalysis/interface/SurveyPxbImageLocalFit.h"
#include "Alignment/SurveyAnalysis/interface/SurveyPxbImageReader.h"
#include "Alignment/SurveyAnalysis/interface/SurveyPxbDicer.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "Alignment/CommonAlignmentParametrization/interface/AlignmentParametersFactory.h"

using namespace gbl;

// Constructor ----------------------------------------------------------------
//____________________________________________________
MillePedeAlignmentAlgorithm::MillePedeAlignmentAlgorithm(const edm::ParameterSet &cfg, edm::ConsumesCollector &iC)
    : AlignmentAlgorithmBase(cfg, iC),
      topoToken_(iC.esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()),
      aliThrToken_(iC.esConsumes<AlignPCLThresholdsHG, AlignPCLThresholdsHGRcd, edm::Transition::BeginRun>()),
      geomToken_(iC.esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      theConfig(cfg),
      theMode(this->decodeMode(theConfig.getUntrackedParameter<std::string>("mode"))),
      theDir(theConfig.getUntrackedParameter<std::string>("fileDir")),
      theAlignmentParameterStore(nullptr),
      theAlignables(),
      theTrajectoryFactory(
          TrajectoryFactoryPlugin::get()->create(theConfig.getParameter<edm::ParameterSet>("TrajectoryFactory")
                                                     .getParameter<std::string>("TrajectoryFactoryName"),
                                                 theConfig.getParameter<edm::ParameterSet>("TrajectoryFactory"),
                                                 iC)),
      theMinNumHits(cfg.getParameter<unsigned int>("minNumHits")),
      theMaximalCor2D(cfg.getParameter<double>("max2Dcorrelation")),
      firstIOV_(cfg.getUntrackedParameter<AlignmentAlgorithmBase::RunNumber>("firstIOV")),
      ignoreFirstIOVCheck_(cfg.getUntrackedParameter<bool>("ignoreFirstIOVCheck")),
      enableAlignableUpdates_(cfg.getUntrackedParameter<bool>("enableAlignableUpdates")),
      theLastWrittenIov(0),
      theGblDoubleBinary(cfg.getParameter<bool>("doubleBinary")),
      runAtPCL_(cfg.getParameter<bool>("runAtPCL")),
      ignoreHitsWithoutGlobalDerivatives_(cfg.getParameter<bool>("ignoreHitsWithoutGlobalDerivatives")),
      skipGlobalPositionRcdCheck_(cfg.getParameter<bool>("skipGlobalPositionRcdCheck")),
      uniqueRunRanges_(align::makeUniqueRunRanges(cfg.getUntrackedParameter<edm::VParameterSet>("RunRangeSelection"),
                                                  cond::timeTypeSpecs[cond::runnumber].beginValue)),
      enforceSingleIOVInput_(!(enableAlignableUpdates_ && areIOVsSpecified())),
      lastProcessedRun_(cond::timeTypeSpecs[cond::runnumber].beginValue) {
  if (!theDir.empty() && theDir.find_last_of('/') != theDir.size() - 1)
    theDir += '/';  // may need '/'
  edm::LogInfo("Alignment") << "@SUB=MillePedeAlignmentAlgorithm"
                            << "Start in mode '" << theConfig.getUntrackedParameter<std::string>("mode")
                            << "' with output directory '" << theDir << "'.";
  if (this->isMode(myMilleBit)) {
    theMille = std::make_unique<Mille>(
        (theDir + theConfig.getParameter<std::string>("binaryFile")).c_str());  // add ', false);' for text output);
    // use same file for GBL
    theBinary = std::make_unique<MilleBinary>((theDir + theConfig.getParameter<std::string>("binaryFile")).c_str(),
                                              theGblDoubleBinary);
  }
}

// Destructor ----------------------------------------------------------------
//____________________________________________________
MillePedeAlignmentAlgorithm::~MillePedeAlignmentAlgorithm() {}

// Call at beginning of job ---------------------------------------------------
//____________________________________________________
void MillePedeAlignmentAlgorithm::initialize(const edm::EventSetup &setup,
                                             AlignableTracker *tracker,
                                             AlignableMuon *muon,
                                             AlignableExtras *extras,
                                             AlignmentParameterStore *store) {
  if (muon) {
    edm::LogWarning("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::initialize"
                                 << "Running with AlignabeMuon not yet tested.";
  }

  if (!runAtPCL_ && enforceSingleIOVInput_) {
    const auto MIN_VAL = cond::timeTypeSpecs[cond::runnumber].beginValue;
    const auto MAX_VAL = cond::timeTypeSpecs[cond::runnumber].endValue;
    const auto &iov_alignments = setup.get<TrackerAlignmentRcd>().validityInterval();
    const auto &iov_surfaces = setup.get<TrackerSurfaceDeformationRcd>().validityInterval();
    const auto &iov_errors = setup.get<TrackerAlignmentErrorExtendedRcd>().validityInterval();

    std::ostringstream message;
    bool throwException{false};
    if (iov_alignments.first().eventID().run() != MIN_VAL || iov_alignments.last().eventID().run() != MAX_VAL) {
      message << "\nTrying to apply " << setup.get<TrackerAlignmentRcd>().key().name()
              << " with multiple IOVs in tag without specifying 'RunRangeSelection'.\n"
              << "Validity range is " << iov_alignments.first().eventID().run() << " - "
              << iov_alignments.last().eventID().run() << "\n";
      throwException = true;
    }
    if (iov_surfaces.first().eventID().run() != MIN_VAL || iov_surfaces.last().eventID().run() != MAX_VAL) {
      message << "\nTrying to apply " << setup.get<TrackerSurfaceDeformationRcd>().key().name()
              << " with multiple IOVs in tag without specifying 'RunRangeSelection'.\n"
              << "Validity range is " << iov_surfaces.first().eventID().run() << " - "
              << iov_surfaces.last().eventID().run() << "\n";
      throwException = true;
    }
    if (iov_errors.first().eventID().run() != MIN_VAL || iov_errors.last().eventID().run() != MAX_VAL) {
      message << "\nTrying to apply " << setup.get<TrackerAlignmentErrorExtendedRcd>().key().name()
              << " with multiple IOVs in tag without specifying 'RunRangeSelection'.\n"
              << "Validity range is " << iov_errors.first().eventID().run() << " - "
              << iov_errors.last().eventID().run() << "\n";
      throwException = true;
    }
    if (throwException) {
      throw cms::Exception("DatabaseError") << "@SUB=MillePedeAlignmentAlgorithm::initialize" << message.str();
    }
  }

  //Retrieve tracker topology from geometry
  const TrackerTopology *const tTopo = &setup.getData(topoToken_);

  //Retrieve the thresolds cuts from DB for the PCL
  if (runAtPCL_) {
    const auto &th = &setup.getData(aliThrToken_);
    theThresholds = std::make_shared<AlignPCLThresholdsHG>();
    storeThresholds(th->getNrecords(), th->getThreshold_Map(), th->getFloatMap());

    //Retrieve tracker geometry
    const TrackerGeometry *tGeom = &setup.getData(geomToken_);
    //Retrieve PixelTopologyMap
    pixelTopologyMap = std::make_shared<PixelTopologyMap>(tGeom, tTopo);
  }

  theAlignableNavigator = std::make_unique<AlignableNavigator>(extras, tracker, muon);
  theAlignmentParameterStore = store;
  theAlignables = theAlignmentParameterStore->alignables();

  edm::ParameterSet pedeLabelerCfg(theConfig.getParameter<edm::ParameterSet>("pedeLabeler"));
  edm::VParameterSet RunRangeSelectionVPSet(theConfig.getUntrackedParameter<edm::VParameterSet>("RunRangeSelection"));
  pedeLabelerCfg.addUntrackedParameter<edm::VParameterSet>("RunRangeSelection", RunRangeSelectionVPSet);

  std::string labelerPlugin = "PedeLabeler";
  if (!RunRangeSelectionVPSet.empty()) {
    labelerPlugin = "RunRangeDependentPedeLabeler";
    if (pedeLabelerCfg.exists("plugin")) {
      std::string labelerPluginCfg = pedeLabelerCfg.getParameter<std::string>("plugin");
      if ((labelerPluginCfg != "PedeLabeler" && labelerPluginCfg != "RunRangeDependentPedeLabeler") ||
          !pedeLabelerCfg.getUntrackedParameter<edm::VParameterSet>("parameterInstances").empty()) {
        throw cms::Exception("BadConfig") << "MillePedeAlignmentAlgorithm::initialize"
                                          << "both RunRangeSelection and generic labeler specified in config file. "
                                          << "Please get rid of either one of them.\n";
      }
    }
  } else {
    if (pedeLabelerCfg.exists("plugin")) {
      labelerPlugin = pedeLabelerCfg.getParameter<std::string>("plugin");
    }
  }

  if (!pedeLabelerCfg.exists("plugin")) {
    pedeLabelerCfg.addUntrackedParameter<std::string>("plugin", labelerPlugin);
  }

  edm::LogInfo("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::initialize"
                            << "Using plugin '" << labelerPlugin << "' to generate labels.";

  thePedeLabels = std::shared_ptr<PedeLabelerBase>(PedeLabelerPluginFactory::get()->create(
      labelerPlugin, PedeLabelerBase::TopLevelAlignables(tracker, muon, extras), pedeLabelerCfg));

  // 1) Create PedeSteerer: correct alignable positions for coordinate system selection
  edm::ParameterSet pedeSteerCfg(theConfig.getParameter<edm::ParameterSet>("pedeSteerer"));
  thePedeSteer = std::make_unique<PedeSteerer>(tracker,
                                               muon,
                                               extras,
                                               theAlignmentParameterStore,
                                               thePedeLabels.get(),
                                               pedeSteerCfg,
                                               theDir,
                                               !this->isMode(myPedeSteerBit));

  // 2) If requested, directly read in and apply result of previous pede run,
  //    assuming that correction from 1) was also applied to create the result:
  const std::vector<edm::ParameterSet> mprespset(
      theConfig.getParameter<std::vector<edm::ParameterSet> >("pedeReaderInputs"));
  if (!mprespset.empty()) {
    edm::LogInfo("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::initialize"
                              << "Apply " << mprespset.end() - mprespset.begin()
                              << " previous MillePede constants from 'pedeReaderInputs'.";
  }

  // FIXME: add selection of run range via 'pedeReaderInputs'
  // Note: Results for parameters of IntegratedCalibration's cannot be treated...
  RunRange runrange(cond::timeTypeSpecs[cond::runnumber].beginValue, cond::timeTypeSpecs[cond::runnumber].endValue);
  for (std::vector<edm::ParameterSet>::const_iterator iSet = mprespset.begin(), iE = mprespset.end(); iSet != iE;
       ++iSet) {
    // This read will ignore calibrations as long as they are not yet passed to Millepede
    // during/before initialize(..) - currently addCalibrations(..) is called later in AlignmentProducer
    if (!this->readFromPede((*iSet), false, runrange)) {  // false: do not erase SelectionUserVariables
      throw cms::Exception("BadConfig")
          << "MillePedeAlignmentAlgorithm::initialize: Problems reading input constants of "
          << "pedeReaderInputs entry " << iSet - mprespset.begin() << '.';
    }
    theAlignmentParameterStore->applyParameters();
    // Needed to shut up later warning from checkAliParams:
    theAlignmentParameterStore->resetParameters();
  }

  // 3) Now create steerings with 'final' start position:
  thePedeSteer->buildSubSteer(tracker, muon, extras);

  // After (!) 1-3 of PedeSteerer which uses the SelectionUserVariables attached to the parameters:
  this->buildUserVariables(theAlignables);  // for hit statistics and/or pede result

  if (this->isMode(myMilleBit)) {
    if (!theConfig.getParameter<std::vector<std::string> >("mergeBinaryFiles").empty() ||
        !theConfig.getParameter<std::vector<std::string> >("mergeTreeFiles").empty()) {
      throw cms::Exception("BadConfig") << "'vstring mergeTreeFiles' and 'vstring mergeBinaryFiles' must be empty for "
                                        << "modes running mille.";
    }
    const std::string moniFile(theConfig.getUntrackedParameter<std::string>("monitorFile"));
    if (!moniFile.empty())
      theMonitor = std::make_unique<MillePedeMonitor>(tTopo, (theDir + moniFile).c_str());

    // Get trajectory factory. In case nothing found, FrameWork will throw...
  }

  if (this->isMode(myPedeSteerBit)) {
    // Get config for survey and set flag accordingly
    const edm::ParameterSet pxbSurveyCfg(theConfig.getParameter<edm::ParameterSet>("surveyPixelBarrel"));
    theDoSurveyPixelBarrel = pxbSurveyCfg.getParameter<bool>("doSurvey");
    if (theDoSurveyPixelBarrel)
      this->addPxbSurvey(pxbSurveyCfg);
  }
}

//____________________________________________________
bool MillePedeAlignmentAlgorithm::supportsCalibrations() { return true; }

//____________________________________________________
bool MillePedeAlignmentAlgorithm::addCalibrations(const std::vector<IntegratedCalibrationBase *> &iCals) {
  theCalibrations.insert(theCalibrations.end(), iCals.begin(), iCals.end());
  thePedeLabels->addCalibrations(iCals);
  return true;
}

//____________________________________________________
bool MillePedeAlignmentAlgorithm::storeThresholds(const int &nRecords,
                                                  const AlignPCLThresholdsHG::threshold_map &thresholdMap,
                                                  const AlignPCLThresholdsHG::param_map &floatMap) {
  theThresholds->setAlignPCLThresholds(nRecords, thresholdMap);
  theThresholds->setFloatMap(floatMap);
  return true;
}

//_____________________________________________________________________________
bool MillePedeAlignmentAlgorithm::processesEvents() {
  if (isMode(myMilleBit)) {
    return true;
  } else {
    return false;
  }
}

//_____________________________________________________________________________
bool MillePedeAlignmentAlgorithm::storeAlignments() {
  if (isMode(myPedeReadBit)) {
    if (runAtPCL_) {
      MillePedeFileReader mpReader(theConfig.getParameter<edm::ParameterSet>("MillePedeFileReader"),
                                   thePedeLabels,
                                   theThresholds,
                                   pixelTopologyMap);
      mpReader.read();
      return mpReader.storeAlignments();
    } else {
      return true;
    }
  } else {
    return false;
  }
}

//____________________________________________________
bool MillePedeAlignmentAlgorithm::setParametersForRunRange(const RunRange &runrange) {
  if (this->isMode(myPedeReadBit)) {
    if (not theAlignmentParameterStore) {
      return false;
    }
    // restore initial positions, rotations and deformations
    if (enableAlignableUpdates_) {
      theAlignmentParameterStore->restoreCachedTransformations(runrange.first);
    } else {
      theAlignmentParameterStore->restoreCachedTransformations();
    }

    // Needed to shut up later warning from checkAliParams:
    theAlignmentParameterStore->resetParameters();
    // To avoid that they keep values from previous IOV if no new one in pede result
    this->buildUserVariables(theAlignables);

    if (!this->readFromPede(theConfig.getParameter<edm::ParameterSet>("pedeReader"), true, runrange)) {
      edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::setParametersForRunRange"
                                 << "Problems reading pede result, but applying!";
    }
    theAlignmentParameterStore->applyParameters();

    this->doIO(++theLastWrittenIov);  // pre-increment!
  }

  return true;
}

// Call at end of job ---------------------------------------------------------
//____________________________________________________
void MillePedeAlignmentAlgorithm::terminate(const edm::EventSetup &iSetup) { terminate(); }
void MillePedeAlignmentAlgorithm::terminate() {
  theMille.reset();  // delete to close binary before running pede below (flush would be enough...)
  theBinary.reset();

  std::vector<std::string> files;
  if (this->isMode(myMilleBit) || !theConfig.getParameter<std::string>("binaryFile").empty()) {
    files.push_back(theDir + theConfig.getParameter<std::string>("binaryFile"));
  } else {
    const std::vector<std::string> plainFiles(theConfig.getParameter<std::vector<std::string> >("mergeBinaryFiles"));
    files = getExistingFormattedFiles(plainFiles, theDir);
    // Do some logging:
    std::string filesForLogOutput;
    for (const auto &file : files)
      filesForLogOutput += " " + file + ",";
    if (filesForLogOutput.length() != 0)
      filesForLogOutput.pop_back();
    edm::LogInfo("Alignment") << "Based on the config parameter mergeBinaryFiles, using the following "
                              << "files as input (assigned weights are indicated by ' -- <weight>'):"
                              << filesForLogOutput;
  }

  if (not theAlignmentParameterStore)
    return;

  // cache all positions, rotations and deformations
  theAlignmentParameterStore->cacheTransformations();
  if (this->isMode(myPedeReadBit) && enableAlignableUpdates_) {
    if (lastProcessedRun_ < uniqueRunRanges_.back().first) {
      throw cms::Exception("BadConfig") << "@SUB=MillePedeAlignmentAlgorithm::terminate\n"
                                        << "Last IOV of 'RunRangeSelection' has not been processed. "
                                        << "Please reconfigure your source to process the runs at least up to "
                                        << uniqueRunRanges_.back().first << ".";
    }
    auto lastCachedRun = uniqueRunRanges_.front().first;
    for (const auto &runRange : uniqueRunRanges_) {
      const auto run = runRange.first;
      if (std::find(cachedRuns_.begin(), cachedRuns_.end(), run) == cachedRuns_.end()) {
        theAlignmentParameterStore->restoreCachedTransformations(lastCachedRun);
        theAlignmentParameterStore->cacheTransformations(run);
      } else {
        lastCachedRun = run;
      }
    }
    theAlignmentParameterStore->restoreCachedTransformations();
  }

  const std::string masterSteer(thePedeSteer->buildMasterSteer(files));  // do only if myPedeSteerBit?
  if (this->isMode(myPedeRunBit)) {
    thePedeSteer->runPede(masterSteer);
  }

  // parameters from pede are not yet applied,
  // so we can still write start positions (but with hit statistics in case of mille):
  this->doIO(0);
  theLastWrittenIov = 0;
}

std::vector<std::string> MillePedeAlignmentAlgorithm::getExistingFormattedFiles(
    const std::vector<std::string> &plainFiles, const std::string &theDir) {
  std::vector<std::string> files;
  for (const auto &plainFile : plainFiles) {
    const std::string &theInputFileName = plainFile;
    int theNumber = 0;
    while (true) {
      // Create a formatted version of the filename, with growing numbers
      // If the parameter doesn't contain a formatting directive, it just stays unchanged
      char theNumberedInputFileName[200];
      sprintf(theNumberedInputFileName, theInputFileName.c_str(), theNumber);
      std::string theCompleteInputFileName = theDir + theNumberedInputFileName;
      const auto endOfStrippedFileName = theCompleteInputFileName.rfind(" --");
      const auto strippedInputFileName = theCompleteInputFileName.substr(0, endOfStrippedFileName);
      // Check if the file exists
      struct stat buffer;
      if (stat(strippedInputFileName.c_str(), &buffer) == 0) {
        // If the file exists, add it to the list
        files.push_back(theCompleteInputFileName);
        if (theNumberedInputFileName == theInputFileName) {
          // If the filename didn't contain a formatting directive, no reason to look any further, break out of the loop
          break;
        } else {
          // Otherwise look for the next number
          theNumber++;
        }
      } else {
        // The file doesn't exist, break out of the loop
        break;
      }
    }
    // warning if unformatted (-> theNumber stays at 0) does not exist
    if (theNumber == 0 && (files.empty() || files.back() != plainFile)) {
      edm::LogWarning("Alignment") << "The input file '" << plainFile << "' does not exist.";
    }
  }
  return files;
}

// Run the algorithm on trajectories and tracks -------------------------------
//____________________________________________________
void MillePedeAlignmentAlgorithm::run(const edm::EventSetup &setup, const EventInfo &eventInfo) {
  if (!this->isMode(myMilleBit))
    return;  // no theMille created...
  const auto &tracks = eventInfo.trajTrackPairs();

  if (theMonitor) {  // monitor input tracks
    for (const auto &iTrajTrack : tracks) {
      theMonitor->fillTrack(iTrajTrack.second);
    }
  }

  const RefTrajColl trajectories(theTrajectoryFactory->trajectories(setup, tracks, eventInfo.beamSpot()));

  // Now loop over ReferenceTrajectoryCollection
  unsigned int refTrajCount = 0;  // counter for track monitoring
  const auto tracksPerTraj = theTrajectoryFactory->tracksPerTrajectory();
  for (auto iRefTraj = trajectories.cbegin(), iRefTrajE = trajectories.cend(); iRefTraj != iRefTrajE;
       ++iRefTraj, ++refTrajCount) {
    RefTrajColl::value_type refTrajPtr = *iRefTraj;
    if (theMonitor)
      theMonitor->fillRefTrajectory(refTrajPtr);

    const auto nHitXy = this->addReferenceTrajectory(setup, eventInfo, refTrajPtr);

    if (theMonitor && (nHitXy.first || nHitXy.second)) {
      // if trajectory used (i.e. some hits), fill monitoring
      const auto offset = tracksPerTraj * refTrajCount;
      for (unsigned int iTrack = 0; iTrack < tracksPerTraj; ++iTrack) {
        theMonitor->fillUsedTrack(tracks[offset + iTrack].second, nHitXy.first, nHitXy.second);
      }
    }

  }  // end of reference trajectory and track loop
}

//____________________________________________________
std::pair<unsigned int, unsigned int> MillePedeAlignmentAlgorithm::addReferenceTrajectory(
    const edm::EventSetup &setup, const EventInfo &eventInfo, const RefTrajColl::value_type &refTrajPtr) {
  std::pair<unsigned int, unsigned int> hitResultXy(0, 0);
  if (refTrajPtr->isValid()) {
    // GblTrajectory?
    if (!refTrajPtr->gblInput().empty()) {
      // by construction: number of GblPoints == number of recHits or == zero !!!
      unsigned int iHit = 0;
      unsigned int numPointsWithMeas = 0;
      std::vector<GblPoint>::iterator itPoint;
      auto theGblInput = refTrajPtr->gblInput();
      for (unsigned int iTraj = 0; iTraj < refTrajPtr->gblInput().size(); ++iTraj) {
        for (itPoint = refTrajPtr->gblInput()[iTraj].first.begin(); itPoint < refTrajPtr->gblInput()[iTraj].first.end();
             ++itPoint) {
          if (this->addGlobalData(setup, eventInfo, refTrajPtr, iHit++, *itPoint) < 0)
            return hitResultXy;
          if (itPoint->numMeasurements() >= 1)
            ++numPointsWithMeas;
        }
      }
      hitResultXy.first = numPointsWithMeas;
      // check #hits criterion
      if (hitResultXy.first == 0 || hitResultXy.first < theMinNumHits)
        return hitResultXy;
      // construct GBL trajectory
      if (refTrajPtr->gblInput().size() == 1) {
        // from single track
        GblTrajectory aGblTrajectory(refTrajPtr->gblInput()[0].first, refTrajPtr->nominalField() != 0);
        // GBL fit trajectory
        /*double Chi2;
        int Ndf;
        double lostWeight;
        aGblTrajectory.fit(Chi2, Ndf, lostWeight);
        std::cout << " GblFit: " << Chi2 << ", " << Ndf << ", " << lostWeight << std::endl; */
        // write to MP binary file
        if (aGblTrajectory.isValid() && aGblTrajectory.getNumPoints() >= theMinNumHits)
          aGblTrajectory.milleOut(*theBinary);
      }
      if (refTrajPtr->gblInput().size() == 2) {
        // from TwoBodyDecay
        GblTrajectory aGblTrajectory(refTrajPtr->gblInput(),
                                     refTrajPtr->gblExtDerivatives(),
                                     refTrajPtr->gblExtMeasurements(),
                                     refTrajPtr->gblExtPrecisions());
        // write to MP binary file
        if (aGblTrajectory.isValid() && aGblTrajectory.getNumPoints() >= theMinNumHits)
          aGblTrajectory.milleOut(*theBinary);
      }
    } else {
      // to add hits if all fine:
      std::vector<AlignmentParameters *> parVec(refTrajPtr->recHits().size());
      // collect hit statistics, assuming that there are no y-only hits
      std::vector<bool> validHitVecY(refTrajPtr->recHits().size(), false);
      // Use recHits from ReferenceTrajectory (since they have the right order!):
      for (unsigned int iHit = 0; iHit < refTrajPtr->recHits().size(); ++iHit) {
        const int flagXY = this->addMeasurementData(setup, eventInfo, refTrajPtr, iHit, parVec[iHit]);

        if (flagXY < 0) {  // problem
          hitResultXy.first = 0;
          break;
        } else {  // hit is fine, increase x/y statistics
          if (flagXY >= 1)
            ++hitResultXy.first;
          validHitVecY[iHit] = (flagXY >= 2);
        }
      }  // end loop on hits

      // add virtual measurements
      for (unsigned int iVirtualMeas = 0; iVirtualMeas < refTrajPtr->numberOfVirtualMeas(); ++iVirtualMeas) {
        this->addVirtualMeas(refTrajPtr, iVirtualMeas);
      }

      // kill or end 'track' for mille, depends on #hits criterion
      if (hitResultXy.first == 0 || hitResultXy.first < theMinNumHits) {
        theMille->kill();
        hitResultXy.first = hitResultXy.second = 0;  //reset
      } else {
        theMille->end();
        // add x/y hit count to MillePedeVariables of parVec,
        // returning number of y-hits of the reference trajectory
        hitResultXy.second = this->addHitCount(parVec, validHitVecY);
        //
      }
    }

  }  // end if valid trajectory

  return hitResultXy;
}

//____________________________________________________
unsigned int MillePedeAlignmentAlgorithm::addHitCount(const std::vector<AlignmentParameters *> &parVec,
                                                      const std::vector<bool> &validHitVecY) const {
  // Loop on all hit information in the input arrays and count valid y-hits:
  unsigned int nHitY = 0;
  for (unsigned int iHit = 0; iHit < validHitVecY.size(); ++iHit) {
    Alignable *ali = (parVec[iHit] ? parVec[iHit]->alignable() : nullptr);
    // Loop upwards on hierarchy of alignables to add hits to all levels
    // that are currently aligned. If only a non-selected alignable was hit,
    // (i.e. flagXY == 0 in addReferenceTrajectory(..)), there is no loop at all...
    while (ali) {
      AlignmentParameters *pars = ali->alignmentParameters();
      if (pars) {  // otherwise hierarchy level not selected
        // cast ensured by previous checks:
        MillePedeVariables *mpVar = static_cast<MillePedeVariables *>(pars->userVariables());
        // every hit has an x-measurement, cf. addReferenceTrajectory(..):
        mpVar->increaseHitsX();
        if (validHitVecY[iHit]) {
          mpVar->increaseHitsY();
          if (pars == parVec[iHit])
            ++nHitY;  // do not count hits twice
        }
      }
      ali = ali->mother();
    }
  }

  return nHitY;
}

void MillePedeAlignmentAlgorithm::beginRun(const edm::Run &run, const edm::EventSetup &setup, bool changed) {
  if (run.run() < firstIOV_ && !ignoreFirstIOVCheck_) {
    throw cms::Exception("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::beginRun\n"
                                      << "Using data (run = " << run.run() << ") prior to the first defined IOV ("
                                      << firstIOV_ << ").";
  }

  lastProcessedRun_ = run.run();

  if (changed && enableAlignableUpdates_) {
    const auto runNumber = run.run();
    auto firstRun = cond::timeTypeSpecs[cond::runnumber].beginValue;
    for (auto runRange = uniqueRunRanges_.crbegin(); runRange != uniqueRunRanges_.crend(); ++runRange) {
      if (runNumber >= runRange->first) {
        firstRun = runRange->first;
        break;
      }
    }
    if (std::find(cachedRuns_.begin(), cachedRuns_.end(), firstRun) != cachedRuns_.end()) {
      const auto &geometryRcd = setup.get<IdealGeometryRecord>();
      const auto &globalPosRcd = setup.get<GlobalPositionRcd>();
      const auto &alignmentRcd = setup.get<TrackerAlignmentRcd>();
      const auto &surfaceRcd = setup.get<TrackerSurfaceDeformationRcd>();
      const auto &errorRcd = setup.get<TrackerAlignmentErrorExtendedRcd>();

      std::ostringstream message;
      bool throwException{false};
      message << "Trying to cache tracker alignment payloads for a run (" << runNumber << ") in an IOV (" << firstRun
              << ") that was already cached.\n"
              << "The following records in your input database tag have an IOV "
              << "boundary that does not match your IOV definition:\n";
      if (geometryRcd.validityInterval().first().eventID().run() > firstRun) {
        message << " - IdealGeometryRecord '" << geometryRcd.key().name() << "' (since "
                << geometryRcd.validityInterval().first().eventID().run() << ")\n";
        throwException = true;
      }
      if (globalPosRcd.validityInterval().first().eventID().run() > firstRun) {
        message << " - GlobalPositionRecord '" << globalPosRcd.key().name() << "' (since "
                << globalPosRcd.validityInterval().first().eventID().run() << ")";
        if (skipGlobalPositionRcdCheck_) {
          message << " --> ignored\n";
        } else {
          message << "\n";
          throwException = true;
        }
      }
      if (alignmentRcd.validityInterval().first().eventID().run() > firstRun) {
        message << " - TrackerAlignmentRcd '" << alignmentRcd.key().name() << "' (since "
                << alignmentRcd.validityInterval().first().eventID().run() << ")\n";
        throwException = true;
      }
      if (surfaceRcd.validityInterval().first().eventID().run() > firstRun) {
        message << " - TrackerSurfaceDeformationRcd '" << surfaceRcd.key().name() << "' (since "
                << surfaceRcd.validityInterval().first().eventID().run() << ")\n";
        throwException = true;
      }
      if (errorRcd.validityInterval().first().eventID().run() > firstRun) {
        message << " - TrackerAlignmentErrorExtendedRcd '" << errorRcd.key().name() << "' (since "
                << errorRcd.validityInterval().first().eventID().run() << ")\n";
        throwException = true;
      }

      if (throwException) {
        throw cms::Exception("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::beginRun\n" << message.str();
      }
    } else {
      cachedRuns_.push_back(firstRun);
      theAlignmentParameterStore->cacheTransformations(firstRun);
    }
  }
}

//____________________________________________________
void MillePedeAlignmentAlgorithm::endRun(const EventInfo &eventInfo,
                                         const EndRunInfo &runInfo,
                                         const edm::EventSetup &setup) {
  if (runInfo.tkLasBeams() && runInfo.tkLasBeamTsoses()) {
    // LAS beam treatment
    this->addLaserData(eventInfo, *(runInfo.tkLasBeams()), *(runInfo.tkLasBeamTsoses()));
  }
  if (this->isMode(myMilleBit))
    theMille->flushOutputFile();
}

// Implementation of endRun that DOES get called. (Because we need it.)
void MillePedeAlignmentAlgorithm::endRun(const EndRunInfo &runInfo, const edm::EventSetup &setup) {
  if (this->isMode(myMilleBit))
    theMille->flushOutputFile();
}

//____________________________________________________
void MillePedeAlignmentAlgorithm::beginLuminosityBlock(const edm::EventSetup &) {
  if (!runAtPCL_)
    return;
  if (this->isMode(myMilleBit))
    theMille->resetOutputFile();
}

//____________________________________________________
void MillePedeAlignmentAlgorithm::endLuminosityBlock(const edm::EventSetup &) {
  if (!runAtPCL_)
    return;
  if (this->isMode(myMilleBit))
    theMille->flushOutputFile();
}

//____________________________________________________
int MillePedeAlignmentAlgorithm::addMeasurementData(const edm::EventSetup &setup,
                                                    const EventInfo &eventInfo,
                                                    const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
                                                    unsigned int iHit,
                                                    AlignmentParameters *&params) {
  params = nullptr;
  theFloatBufferX.clear();
  theFloatBufferY.clear();
  theIntBuffer.clear();

  const TrajectoryStateOnSurface &tsos = refTrajPtr->trajectoryStates()[iHit];
  const ConstRecHitPointer &recHitPtr = refTrajPtr->recHits()[iHit];
  // ignore invalid hits
  if (!recHitPtr->isValid())
    return 0;

  // First add the derivatives from IntegratedCalibration's,
  // should even be OK if problems for "usual" derivatives from Alignables
  this->globalDerivativesCalibration(recHitPtr,
                                     tsos,
                                     setup,
                                     eventInfo,  // input
                                     theFloatBufferX,
                                     theFloatBufferY,
                                     theIntBuffer);  // output

  // get AlignableDet/Unit for this hit
  AlignableDetOrUnitPtr alidet(theAlignableNavigator->alignableFromDetId(recHitPtr->geographicalId()));

  if (!this->globalDerivativesHierarchy(eventInfo,
                                        tsos,
                                        alidet,
                                        alidet,
                                        theFloatBufferX,  // 2x alidet, sic!
                                        theFloatBufferY,
                                        theIntBuffer,
                                        params)) {
    return -1;  // problem
  } else if (theFloatBufferX.empty() && ignoreHitsWithoutGlobalDerivatives_) {
    return 0;  // empty for X: no alignable for hit, nor calibrations
  } else {
    // store measurement even if no alignable or calibrations
    // -> measurement used for pede-internal track-fit
    return this->callMille(refTrajPtr, iHit, theIntBuffer, theFloatBufferX, theFloatBufferY);
  }
}

//____________________________________________________

int MillePedeAlignmentAlgorithm::addGlobalData(const edm::EventSetup &setup,
                                               const EventInfo &eventInfo,
                                               const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
                                               unsigned int iHit,
                                               GblPoint &gblPoint) {
  AlignmentParameters *params = nullptr;
  std::vector<double> theDoubleBufferX, theDoubleBufferY;
  theDoubleBufferX.clear();
  theDoubleBufferY.clear();
  theIntBuffer.clear();
  int iret = 0;

  const TrajectoryStateOnSurface &tsos = refTrajPtr->trajectoryStates()[iHit];
  const ConstRecHitPointer &recHitPtr = refTrajPtr->recHits()[iHit];
  // ignore invalid hits
  if (!recHitPtr->isValid())
    return 0;

  // get AlignableDet/Unit for this hit
  AlignableDetOrUnitPtr alidet(theAlignableNavigator->alignableFromDetId(recHitPtr->geographicalId()));

  if (!this->globalDerivativesHierarchy(eventInfo,
                                        tsos,
                                        alidet,
                                        alidet,
                                        theDoubleBufferX,  // 2x alidet, sic!
                                        theDoubleBufferY,
                                        theIntBuffer,
                                        params)) {
    return -1;  // problem
  }
  //calibration parameters
  int globalLabel;
  std::vector<IntegratedCalibrationBase::ValuesIndexPair> derivs;
  for (auto iCalib = theCalibrations.begin(); iCalib != theCalibrations.end(); ++iCalib) {
    // get all derivatives of this calibration // const unsigned int num =
    (*iCalib)->derivatives(derivs, *recHitPtr, tsos, setup, eventInfo);
    for (auto iValuesInd = derivs.begin(); iValuesInd != derivs.end(); ++iValuesInd) {
      // transfer label and x/y derivatives
      globalLabel = thePedeLabels->calibrationLabel(*iCalib, iValuesInd->second);
      if (globalLabel > 0 && globalLabel <= 2147483647) {
        theIntBuffer.push_back(globalLabel);
        theDoubleBufferX.push_back(iValuesInd->first.first);
        theDoubleBufferY.push_back(iValuesInd->first.second);
      } else {
        edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::addGlobalData"
                                   << "Invalid label " << globalLabel << " <= 0 or > 2147483647";
      }
    }
  }
  unsigned int numGlobals = theIntBuffer.size();
  if (numGlobals > 0) {
    Eigen::Matrix<double, 2, Eigen::Dynamic> globalDer{2, numGlobals};
    for (unsigned int i = 0; i < numGlobals; ++i) {
      globalDer(0, i) = theDoubleBufferX[i];
      globalDer(1, i) = theDoubleBufferY[i];
    }
    gblPoint.addGlobals(theIntBuffer, globalDer);
    iret = 1;
  }
  return iret;
}

//____________________________________________________
bool MillePedeAlignmentAlgorithm ::globalDerivativesHierarchy(const EventInfo &eventInfo,
                                                              const TrajectoryStateOnSurface &tsos,
                                                              Alignable *ali,
                                                              const AlignableDetOrUnitPtr &alidet,
                                                              std::vector<float> &globalDerivativesX,
                                                              std::vector<float> &globalDerivativesY,
                                                              std::vector<int> &globalLabels,
                                                              AlignmentParameters *&lowestParams) const {
  // derivatives and labels are recursively attached
  if (!ali)
    return true;  // no mother might be OK

  if (false && theMonitor && alidet != ali)
    theMonitor->fillFrameToFrame(alidet, ali);

  AlignmentParameters *params = ali->alignmentParameters();

  if (params) {
    if (!lowestParams)
      lowestParams = params;  // set parameters of lowest level

    bool hasSplitParameters = thePedeLabels->hasSplitParameters(ali);
    const unsigned int alignableLabel = thePedeLabels->alignableLabel(ali);

    if (0 == alignableLabel) {  // FIXME: what about regardAllHits in Markus' code?
      edm::LogWarning("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::globalDerivativesHierarchy"
                                   << "Label not found, skip Alignable.";
      return false;
    }

    const std::vector<bool> &selPars = params->selector();
    const AlgebraicMatrix derivs(params->derivatives(tsos, alidet));

    // cols: 2, i.e. x&y, rows: parameters, usually RigidBodyAlignmentParameters::N_PARAM
    for (unsigned int iSel = 0; iSel < selPars.size(); ++iSel) {
      if (selPars[iSel]) {
        globalDerivativesX.push_back(derivs[iSel][kLocalX] / thePedeSteer->cmsToPedeFactor(iSel));
        if (hasSplitParameters == true) {
          globalLabels.push_back(thePedeLabels->parameterLabel(ali, iSel, eventInfo, tsos));
        } else {
          globalLabels.push_back(thePedeLabels->parameterLabel(alignableLabel, iSel));
        }
        globalDerivativesY.push_back(derivs[iSel][kLocalY] / thePedeSteer->cmsToPedeFactor(iSel));
      }
    }
    // Exclude mothers if Alignable selected to be no part of a hierarchy:
    if (thePedeSteer->isNoHiera(ali))
      return true;
  }
  // Call recursively for mother, will stop if mother == 0:
  return this->globalDerivativesHierarchy(
      eventInfo, tsos, ali->mother(), alidet, globalDerivativesX, globalDerivativesY, globalLabels, lowestParams);
}

//____________________________________________________
bool MillePedeAlignmentAlgorithm ::globalDerivativesHierarchy(const EventInfo &eventInfo,
                                                              const TrajectoryStateOnSurface &tsos,
                                                              Alignable *ali,
                                                              const AlignableDetOrUnitPtr &alidet,
                                                              std::vector<double> &globalDerivativesX,
                                                              std::vector<double> &globalDerivativesY,
                                                              std::vector<int> &globalLabels,
                                                              AlignmentParameters *&lowestParams) const {
  // derivatives and labels are recursively attached
  if (!ali)
    return true;  // no mother might be OK

  if (false && theMonitor && alidet != ali)
    theMonitor->fillFrameToFrame(alidet, ali);

  AlignmentParameters *params = ali->alignmentParameters();

  if (params) {
    if (!lowestParams)
      lowestParams = params;  // set parameters of lowest level

    bool hasSplitParameters = thePedeLabels->hasSplitParameters(ali);
    const unsigned int alignableLabel = thePedeLabels->alignableLabel(ali);

    if (0 == alignableLabel) {  // FIXME: what about regardAllHits in Markus' code?
      edm::LogWarning("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::globalDerivativesHierarchy"
                                   << "Label not found, skip Alignable.";
      return false;
    }

    const std::vector<bool> &selPars = params->selector();
    const AlgebraicMatrix derivs(params->derivatives(tsos, alidet));
    int globalLabel;

    // cols: 2, i.e. x&y, rows: parameters, usually RigidBodyAlignmentParameters::N_PARAM
    for (unsigned int iSel = 0; iSel < selPars.size(); ++iSel) {
      if (selPars[iSel]) {
        if (hasSplitParameters == true) {
          globalLabel = thePedeLabels->parameterLabel(ali, iSel, eventInfo, tsos);
        } else {
          globalLabel = thePedeLabels->parameterLabel(alignableLabel, iSel);
        }
        if (globalLabel > 0 && globalLabel <= 2147483647) {
          globalLabels.push_back(globalLabel);
          globalDerivativesX.push_back(derivs[iSel][kLocalX] / thePedeSteer->cmsToPedeFactor(iSel));
          globalDerivativesY.push_back(derivs[iSel][kLocalY] / thePedeSteer->cmsToPedeFactor(iSel));
        } else {
          edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::globalDerivativesHierarchy"
                                     << "Invalid label " << globalLabel << " <= 0 or > 2147483647";
        }
      }
    }
    // Exclude mothers if Alignable selected to be no part of a hierarchy:
    if (thePedeSteer->isNoHiera(ali))
      return true;
  }
  // Call recursively for mother, will stop if mother == 0:
  return this->globalDerivativesHierarchy(
      eventInfo, tsos, ali->mother(), alidet, globalDerivativesX, globalDerivativesY, globalLabels, lowestParams);
}

//____________________________________________________
void MillePedeAlignmentAlgorithm::globalDerivativesCalibration(const TransientTrackingRecHit::ConstRecHitPointer &recHit,
                                                               const TrajectoryStateOnSurface &tsos,
                                                               const edm::EventSetup &setup,
                                                               const EventInfo &eventInfo,
                                                               std::vector<float> &globalDerivativesX,
                                                               std::vector<float> &globalDerivativesY,
                                                               std::vector<int> &globalLabels) const {
  std::vector<IntegratedCalibrationBase::ValuesIndexPair> derivs;
  for (auto iCalib = theCalibrations.begin(); iCalib != theCalibrations.end(); ++iCalib) {
    // get all derivatives of this calibration // const unsigned int num =
    (*iCalib)->derivatives(derivs, *recHit, tsos, setup, eventInfo);
    for (auto iValuesInd = derivs.begin(); iValuesInd != derivs.end(); ++iValuesInd) {
      // transfer label and x/y derivatives
      globalLabels.push_back(thePedeLabels->calibrationLabel(*iCalib, iValuesInd->second));
      globalDerivativesX.push_back(iValuesInd->first.first);
      globalDerivativesY.push_back(iValuesInd->first.second);
    }
  }
}

// //____________________________________________________
// void MillePedeAlignmentAlgorithm
// ::callMille(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
//             unsigned int iTrajHit, MeasurementDirection xOrY,
//             const std::vector<float> &globalDerivatives, const std::vector<int> &globalLabels)
// {
//   const unsigned int xyIndex = iTrajHit*2 + xOrY;
//   // FIXME: here for residuum and sigma we could use KALMAN-Filter results
//   const float residuum =
//     refTrajPtr->measurements()[xyIndex] - refTrajPtr->trajectoryPositions()[xyIndex];
//   const float covariance = refTrajPtr->measurementErrors()[xyIndex][xyIndex];
//   const float sigma = (covariance > 0. ? TMath::Sqrt(covariance) : 0.);

//   const AlgebraicMatrix &locDerivMatrix = refTrajPtr->derivatives();

//   std::vector<float> localDerivs(locDerivMatrix.num_col());
//   for (unsigned int i = 0; i < localDerivs.size(); ++i) {
//     localDerivs[i] = locDerivMatrix[xyIndex][i];
//   }

//   // &(vector[0]) is valid - as long as vector is not empty
//   // cf. http://www.parashift.com/c++-faq-lite/containers.html#faq-34.3
//   theMille->mille(localDerivs.size(), &(localDerivs[0]),
//                globalDerivatives.size(), &(globalDerivatives[0]), &(globalLabels[0]),
//                residuum, sigma);
//   if (theMonitor) {
//     theMonitor->fillDerivatives(refTrajPtr->recHits()[iTrajHit],localDerivs, globalDerivatives,
//                              (xOrY == kLocalY));
//     theMonitor->fillResiduals(refTrajPtr->recHits()[iTrajHit],
//                            refTrajPtr->trajectoryStates()[iTrajHit],
//                            iTrajHit, residuum, sigma, (xOrY == kLocalY));
//   }
// }

//____________________________________________________
bool MillePedeAlignmentAlgorithm::is2D(const ConstRecHitPointer &recHit) const {
  // FIXME: Check whether this is a reliable and recommended way to find out...

  if (recHit->dimension() < 2) {
    return false;                  // some muon and TIB/TOB stuff really has RecHit1D
  } else if (recHit->detUnit()) {  // detunit in strip is 1D, in pixel 2D
    return recHit->detUnit()->type().isTrackerPixel();
  } else {  // stereo strips  (FIXME: endcap trouble due to non-parallel strips (wedge sensors)?)
    if (dynamic_cast<const ProjectedSiStripRecHit2D *>(recHit->hit())) {  // check persistent hit
      // projected: 1D measurement on 'glued' module
      return false;
    } else {
      return true;
    }
  }
}

//__________________________________________________________________________________________________
bool MillePedeAlignmentAlgorithm::readFromPede(const edm::ParameterSet &mprespset,
                                               bool setUserVars,
                                               const RunRange &runrange) {
  bool allEmpty = this->areEmptyParams(theAlignables);

  PedeReader reader(mprespset, *thePedeSteer, *thePedeLabels, runrange);
  align::Alignables alis;
  bool okRead = reader.read(alis, setUserVars);  // also may set params of IntegratedCalibration's
  bool numMatch = true;

  std::stringstream out;
  out << "Read " << alis.size() << " alignables";
  if (alis.size() != theAlignables.size()) {
    out << " while " << theAlignables.size() << " in store";
    numMatch = false;  // FIXME: Should we check one by one? Or transfer 'alis' to the store?
  }
  if (!okRead)
    out << ", but problems in reading";
  if (!allEmpty)
    out << ", possibly overwriting previous settings";
  out << ".";

  if (okRead && allEmpty) {
    if (numMatch) {  // as many alignables with result as trying to align
      edm::LogInfo("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::readFromPede" << out.str();
    } else if (!alis.empty()) {  // dead module do not get hits and no pede result
      edm::LogWarning("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::readFromPede" << out.str();
    } else {  // serious problem: no result read - and not all modules can be dead...
      edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::readFromPede" << out.str();
      return false;
    }
    return true;
  }
  // the rest is not OK:
  edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::readFromPede" << out.str();
  return false;
}

//__________________________________________________________________________________________________
bool MillePedeAlignmentAlgorithm::areEmptyParams(const align::Alignables &alignables) const {
  for (const auto &iAli : alignables) {
    const AlignmentParameters *params = iAli->alignmentParameters();
    if (params) {
      const auto &parVec(params->parameters());
      const auto &parCov(params->covariance());
      for (int i = 0; i < parVec.num_row(); ++i) {
        if (parVec[i] != 0.)
          return false;
        for (int j = i; j < parCov.num_col(); ++j) {
          if (parCov[i][j] != 0.)
            return false;
        }
      }
    }
  }

  return true;
}

//__________________________________________________________________________________________________
unsigned int MillePedeAlignmentAlgorithm::doIO(int loop) const {
  unsigned int result = 0;

  const std::string outFilePlain(theConfig.getParameter<std::string>("treeFile"));
  if (outFilePlain.empty()) {
    edm::LogInfo("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO"
                              << "treeFile parameter empty => skip writing for 'loop' " << loop;
    return result;
  }

  const std::string outFile(theDir + outFilePlain);

  AlignmentIORoot aliIO;
  int ioerr = 0;
  if (loop == 0) {
    aliIO.writeAlignableOriginalPositions(theAlignables, outFile.c_str(), loop, false, ioerr);
    if (ioerr) {
      edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO"
                                 << "Problem " << ioerr << " in writeAlignableOriginalPositions";
      ++result;
    }
  } else if (loop == 1) {
    // only for first iov add hit counts, else 2x, 3x,... number of hits in IOV 2, 3,...
    const std::vector<std::string> inFiles(theConfig.getParameter<std::vector<std::string> >("mergeTreeFiles"));
    const std::vector<std::string> binFiles(theConfig.getParameter<std::vector<std::string> >("mergeBinaryFiles"));
    if (inFiles.size() != binFiles.size()) {
      edm::LogWarning("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO"
                                   << "'vstring mergeTreeFiles' and 'vstring mergeBinaryFiles' "
                                   << "differ in size";
    }
    this->addHitStatistics(0, outFile, inFiles);  // add hit info from tree 0 in 'infiles'
  }
  MillePedeVariablesIORoot millePedeIO;
  millePedeIO.writeMillePedeVariables(theAlignables, outFile.c_str(), loop, false, ioerr);
  if (ioerr) {
    edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO"
                               << "Problem " << ioerr << " writing MillePedeVariables";
    ++result;
  }

  aliIO.writeOrigRigidBodyAlignmentParameters(theAlignables, outFile.c_str(), loop, false, ioerr);
  if (ioerr) {
    edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO"
                               << "Problem " << ioerr << " in writeOrigRigidBodyAlignmentParameters, " << loop;
    ++result;
  }
  aliIO.writeAlignableAbsolutePositions(theAlignables, outFile.c_str(), loop, false, ioerr);
  if (ioerr) {
    edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO"
                               << "Problem " << ioerr << " in writeAlignableAbsolutePositions, " << loop;
    ++result;
  }

  return result;
}

//__________________________________________________________________________________________________
void MillePedeAlignmentAlgorithm::buildUserVariables(const align::Alignables &alis) const {
  for (const auto &iAli : alis) {
    AlignmentParameters *params = iAli->alignmentParameters();
    if (!params) {
      throw cms::Exception("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::buildUserVariables"
                                        << "No parameters for alignable";
    }
    MillePedeVariables *userVars = dynamic_cast<MillePedeVariables *>(params->userVariables());
    if (userVars) {  // Just re-use existing, keeping label and numHits:
      for (unsigned int iPar = 0; iPar < userVars->size(); ++iPar) {
        //      if (params->hierarchyLevel() > 0) {
        //std::cout << params->hierarchyLevel() << "\nBefore: " << userVars->parameter()[iPar];
        //      }
        userVars->setAllDefault(iPar);
        //std::cout << "\nAfter: " << userVars->parameter()[iPar] << std::endl;
      }
    } else {  // Nothing yet or erase wrong type:
      userVars = new MillePedeVariables(
          params->size(),
          thePedeLabels->alignableLabel(iAli),
          thePedeLabels->alignableTracker()->objectIdProvider().typeToName(iAli->alignableObjectId()));
      params->setUserVariables(userVars);
    }
  }
}

//__________________________________________________________________________________________________
unsigned int MillePedeAlignmentAlgorithm::decodeMode(const std::string &mode) const {
  if (mode == "full") {
    return myMilleBit + myPedeSteerBit + myPedeRunBit + myPedeReadBit;
  } else if (mode == "mille") {
    return myMilleBit;  // + myPedeSteerBit; // sic! Including production of steerig file. NO!
  } else if (mode == "pede") {
    return myPedeSteerBit + myPedeRunBit + myPedeReadBit;
  } else if (mode == "pedeSteer") {
    return myPedeSteerBit;
  } else if (mode == "pedeRun") {
    return myPedeSteerBit + myPedeRunBit + myPedeReadBit;  // sic! Including steering and reading of result.
  } else if (mode == "pedeRead") {
    return myPedeReadBit;
  }

  throw cms::Exception("BadConfig") << "Unknown mode '" << mode
                                    << "', use 'full', 'mille', 'pede', 'pedeRun', 'pedeSteer' or 'pedeRead'.";

  return 0;
}

//__________________________________________________________________________________________________
bool MillePedeAlignmentAlgorithm::addHitStatistics(int fromIov,
                                                   const std::string &outFile,
                                                   const std::vector<std::string> &inFiles) const {
  bool allOk = true;
  int ierr = 0;
  MillePedeVariablesIORoot millePedeIO;
  for (std::vector<std::string>::const_iterator iFile = inFiles.begin(); iFile != inFiles.end(); ++iFile) {
    const std::string inFile(theDir + *iFile);
    const std::vector<AlignmentUserVariables *> mpVars =
        millePedeIO.readMillePedeVariables(theAlignables, inFile.c_str(), fromIov, ierr);
    if (ierr || !this->addHits(theAlignables, mpVars)) {
      edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::addHitStatistics"
                                 << "Error " << ierr << " reading from " << inFile << ", tree " << fromIov
                                 << ", or problems in addHits";
      allOk = false;
    }
    for (std::vector<AlignmentUserVariables *>::const_iterator i = mpVars.begin(); i != mpVars.end(); ++i) {
      delete *i;  // clean created objects
    }
  }

  return allOk;
}

//__________________________________________________________________________________________________
bool MillePedeAlignmentAlgorithm::addHits(const align::Alignables &alis,
                                          const std::vector<AlignmentUserVariables *> &mpVars) const {
  bool allOk = (mpVars.size() == alis.size());
  std::vector<AlignmentUserVariables *>::const_iterator iUser = mpVars.begin();
  for (auto iAli = alis.cbegin(); iAli != alis.cend() && iUser != mpVars.end(); ++iAli, ++iUser) {
    MillePedeVariables *mpVarNew = dynamic_cast<MillePedeVariables *>(*iUser);
    AlignmentParameters *ps = (*iAli)->alignmentParameters();
    MillePedeVariables *mpVarOld = (ps ? dynamic_cast<MillePedeVariables *>(ps->userVariables()) : nullptr);
    if (!mpVarNew || !mpVarOld || mpVarOld->size() != mpVarNew->size()) {
      allOk = false;
      continue;  // FIXME error etc.?
    }

    mpVarOld->increaseHitsX(mpVarNew->hitsX());
    mpVarOld->increaseHitsY(mpVarNew->hitsY());
  }

  return allOk;
}

//__________________________________________________________________________________________________
template <typename GlobalDerivativeMatrix>
void MillePedeAlignmentAlgorithm::makeGlobDerivMatrix(const std::vector<float> &globalDerivativesx,
                                                      const std::vector<float> &globalDerivativesy,
                                                      Eigen::MatrixBase<GlobalDerivativeMatrix> &aGlobalDerivativesM) {
  static_assert(GlobalDerivativeMatrix::RowsAtCompileTime == 2, "global derivative matrix must have two rows");

  for (size_t i = 0; i < globalDerivativesx.size(); ++i) {
    aGlobalDerivativesM(0, i) = globalDerivativesx[i];
    aGlobalDerivativesM(1, i) = globalDerivativesy[i];
  }
}

//__________________________________________________________________________________________________
template <typename CovarianceMatrix,
          typename LocalDerivativeMatrix,
          typename ResidualMatrix,
          typename GlobalDerivativeMatrix>
void MillePedeAlignmentAlgorithm::diagonalize(Eigen::MatrixBase<CovarianceMatrix> &aHitCovarianceM,
                                              Eigen::MatrixBase<LocalDerivativeMatrix> &aLocalDerivativesM,
                                              Eigen::MatrixBase<ResidualMatrix> &aHitResidualsM,
                                              Eigen::MatrixBase<GlobalDerivativeMatrix> &aGlobalDerivativesM) const {
  static_assert(std::is_same<typename LocalDerivativeMatrix::Scalar, typename ResidualMatrix::Scalar>::value,
                "'aLocalDerivativesM' and 'aHitResidualsM' must have the "
                "same underlying scalar type");
  static_assert(std::is_same<typename LocalDerivativeMatrix::Scalar, typename GlobalDerivativeMatrix::Scalar>::value,
                "'aLocalDerivativesM' and 'aGlobalDerivativesM' must have the "
                "same underlying scalar type");

  Eigen::SelfAdjointEigenSolver<typename CovarianceMatrix::PlainObject> myDiag{aHitCovarianceM};
  // eigenvectors of real symmetric matrices are orthogonal, i.e. invert == transpose
  auto aTranfoToDiagonalSystemInv =
      myDiag.eigenvectors().transpose().template cast<typename LocalDerivativeMatrix::Scalar>();

  aHitCovarianceM = myDiag.eigenvalues().asDiagonal();
  aLocalDerivativesM = aTranfoToDiagonalSystemInv * aLocalDerivativesM;
  aHitResidualsM = aTranfoToDiagonalSystemInv * aHitResidualsM;
  if (aGlobalDerivativesM.size() > 0) {
    // diagonalize only if measurement depends on alignables or calibrations
    aGlobalDerivativesM = aTranfoToDiagonalSystemInv * aGlobalDerivativesM;
  }
}

//__________________________________________________________________________________________________
template <typename CovarianceMatrix, typename ResidualMatrix, typename LocalDerivativeMatrix>
void MillePedeAlignmentAlgorithm ::addRefTrackVirtualMeas1D(
    const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
    unsigned int iVirtualMeas,
    Eigen::MatrixBase<CovarianceMatrix> &aHitCovarianceM,
    Eigen::MatrixBase<ResidualMatrix> &aHitResidualsM,
    Eigen::MatrixBase<LocalDerivativeMatrix> &aLocalDerivativesM) {
  // This Method is valid for 1D measurements only

  const unsigned int xIndex = iVirtualMeas + refTrajPtr->numberOfHitMeas();

  aHitCovarianceM(0, 0) = refTrajPtr->measurementErrors()[xIndex][xIndex];
  aHitResidualsM(0, 0) = refTrajPtr->measurements()[xIndex];

  const auto &locDerivMatrix = refTrajPtr->derivatives();
  for (int i = 0; i < locDerivMatrix.num_col(); ++i) {
    aLocalDerivativesM(0, i) = locDerivMatrix[xIndex][i];
  }
}

//__________________________________________________________________________________________________
template <typename CovarianceMatrix, typename ResidualMatrix, typename LocalDerivativeMatrix>
void MillePedeAlignmentAlgorithm ::addRefTrackData2D(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
                                                     unsigned int iTrajHit,
                                                     Eigen::MatrixBase<CovarianceMatrix> &aHitCovarianceM,
                                                     Eigen::MatrixBase<ResidualMatrix> &aHitResidualsM,
                                                     Eigen::MatrixBase<LocalDerivativeMatrix> &aLocalDerivativesM) {
  // This Method is valid for 2D measurements only

  const unsigned int xIndex = iTrajHit * 2;
  const unsigned int yIndex = iTrajHit * 2 + 1;

  aHitCovarianceM(0, 0) = refTrajPtr->measurementErrors()[xIndex][xIndex];
  aHitCovarianceM(0, 1) = refTrajPtr->measurementErrors()[xIndex][yIndex];
  aHitCovarianceM(1, 0) = refTrajPtr->measurementErrors()[yIndex][xIndex];
  aHitCovarianceM(1, 1) = refTrajPtr->measurementErrors()[yIndex][yIndex];

  aHitResidualsM(0, 0) = refTrajPtr->measurements()[xIndex] - refTrajPtr->trajectoryPositions()[xIndex];
  aHitResidualsM(1, 0) = refTrajPtr->measurements()[yIndex] - refTrajPtr->trajectoryPositions()[yIndex];

  const auto &locDerivMatrix = refTrajPtr->derivatives();
  for (int i = 0; i < locDerivMatrix.num_col(); ++i) {
    aLocalDerivativesM(0, i) = locDerivMatrix[xIndex][i];
    aLocalDerivativesM(1, i) = locDerivMatrix[yIndex][i];
  }
}

//__________________________________________________________________________________________________
int MillePedeAlignmentAlgorithm ::callMille(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
                                            unsigned int iTrajHit,
                                            const std::vector<int> &globalLabels,
                                            const std::vector<float> &globalDerivativesX,
                                            const std::vector<float> &globalDerivativesY) {
  const ConstRecHitPointer aRecHit(refTrajPtr->recHits()[iTrajHit]);

  if ((aRecHit)->dimension() == 1) {
    return this->callMille1D(refTrajPtr, iTrajHit, globalLabels, globalDerivativesX);
  } else {
    return this->callMille2D(refTrajPtr, iTrajHit, globalLabels, globalDerivativesX, globalDerivativesY);
  }
}

//__________________________________________________________________________________________________
int MillePedeAlignmentAlgorithm ::callMille1D(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
                                              unsigned int iTrajHit,
                                              const std::vector<int> &globalLabels,
                                              const std::vector<float> &globalDerivativesX) {
  const ConstRecHitPointer aRecHit(refTrajPtr->recHits()[iTrajHit]);
  const unsigned int xIndex = iTrajHit * 2;  // the even ones are local x

  // local derivatives
  const AlgebraicMatrix &locDerivMatrix = refTrajPtr->derivatives();
  const int nLocal = locDerivMatrix.num_col();
  std::vector<float> localDerivatives(nLocal);
  for (unsigned int i = 0; i < localDerivatives.size(); ++i) {
    localDerivatives[i] = locDerivMatrix[xIndex][i];
  }

  // residuum and error
  float residX = refTrajPtr->measurements()[xIndex] - refTrajPtr->trajectoryPositions()[xIndex];
  float hitErrX = TMath::Sqrt(refTrajPtr->measurementErrors()[xIndex][xIndex]);

  // number of global derivatives
  const int nGlobal = globalDerivativesX.size();

  // &(localDerivatives[0]) etc. are valid - as long as vector is not empty
  // cf. http://www.parashift.com/c++-faq-lite/containers.html#faq-34.3
  theMille->mille(
      nLocal, &(localDerivatives[0]), nGlobal, &(globalDerivativesX[0]), &(globalLabels[0]), residX, hitErrX);

  if (theMonitor) {
    theMonitor->fillDerivatives(
        aRecHit, &(localDerivatives[0]), nLocal, &(globalDerivativesX[0]), nGlobal, &(globalLabels[0]));
    theMonitor->fillResiduals(aRecHit, refTrajPtr->trajectoryStates()[iTrajHit], iTrajHit, residX, hitErrX, false);
  }

  return 1;
}

//__________________________________________________________________________________________________
int MillePedeAlignmentAlgorithm ::callMille2D(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
                                              unsigned int iTrajHit,
                                              const std::vector<int> &globalLabels,
                                              const std::vector<float> &globalDerivativesx,
                                              const std::vector<float> &globalDerivativesy) {
  const ConstRecHitPointer aRecHit(refTrajPtr->recHits()[iTrajHit]);

  if ((aRecHit)->dimension() != 2) {
    edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::callMille2D"
                               << "You try to call method for 2D hits for a " << (aRecHit)->dimension()
                               << "D Hit. Hit gets ignored!";
    return -1;
  }

  Eigen::Matrix<double, 2, 2> aHitCovarianceM;
  Eigen::Matrix<float, 2, 1> aHitResidualsM;
  Eigen::Matrix<float, 2, Eigen::Dynamic> aLocalDerivativesM{2, refTrajPtr->derivatives().num_col()};
  // below method fills above 3 matrices
  this->addRefTrackData2D(refTrajPtr, iTrajHit, aHitCovarianceM, aHitResidualsM, aLocalDerivativesM);
  Eigen::Matrix<float, 2, Eigen::Dynamic> aGlobalDerivativesM{2, globalDerivativesx.size()};
  this->makeGlobDerivMatrix(globalDerivativesx, globalDerivativesy, aGlobalDerivativesM);

  // calculates correlation between Hit measurements
  // FIXME: Should take correlation (and resulting transformation) from original hit,
  //        not 2x2 matrix from ReferenceTrajectory: That can come from error propagation etc.!
  const double corr = aHitCovarianceM(0, 1) / sqrt(aHitCovarianceM(0, 0) * aHitCovarianceM(1, 1));
  if (theMonitor)
    theMonitor->fillCorrelations2D(corr, aRecHit);
  bool diag = false;  // diagonalise only tracker TID, TEC
  switch (aRecHit->geographicalId().subdetId()) {
    case SiStripDetId::TID:
    case SiStripDetId::TEC:
      if (aRecHit->geographicalId().det() == DetId::Tracker && TMath::Abs(corr) > theMaximalCor2D) {
        this->diagonalize(aHitCovarianceM, aLocalDerivativesM, aHitResidualsM, aGlobalDerivativesM);
        diag = true;
      }
      break;
    default:;
  }

  float newResidX = aHitResidualsM(0, 0);
  float newResidY = aHitResidualsM(1, 0);
  float newHitErrX = TMath::Sqrt(aHitCovarianceM(0, 0));
  float newHitErrY = TMath::Sqrt(aHitCovarianceM(1, 1));

  // change from column major (Eigen default) to row major to have row entries
  // in continuous memory
  std::vector<float> newLocalDerivs(aLocalDerivativesM.size());
  Eigen::Map<Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajor> >(
      newLocalDerivs.data(), aLocalDerivativesM.rows(), aLocalDerivativesM.cols()) = aLocalDerivativesM;
  float *newLocalDerivsX = &(newLocalDerivs[0]);
  float *newLocalDerivsY = &(newLocalDerivs[aLocalDerivativesM.cols()]);

  // change from column major (Eigen default) to row major to have row entries
  // in continuous memory
  std::vector<float> newGlobDerivs(aGlobalDerivativesM.size());
  Eigen::Map<Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajor> >(
      newGlobDerivs.data(), aGlobalDerivativesM.rows(), aGlobalDerivativesM.cols()) = aGlobalDerivativesM;
  float *newGlobDerivsX = &(newGlobDerivs[0]);
  float *newGlobDerivsY = &(newGlobDerivs[aGlobalDerivativesM.cols()]);

  const int nLocal = aLocalDerivativesM.cols();
  const int nGlobal = aGlobalDerivativesM.cols();

  if (diag && (newHitErrX > newHitErrY)) {  // also for 2D hits?
    // measurement with smaller error is x-measurement (for !is2D do not fill y-measurement):
    std::swap(newResidX, newResidY);
    std::swap(newHitErrX, newHitErrY);
    std::swap(newLocalDerivsX, newLocalDerivsY);
    std::swap(newGlobDerivsX, newGlobDerivsY);
  }

  // &(globalLabels[0]) is valid - as long as vector is not empty
  // cf. http://www.parashift.com/c++-faq-lite/containers.html#faq-34.3
  theMille->mille(nLocal, newLocalDerivsX, nGlobal, newGlobDerivsX, &(globalLabels[0]), newResidX, newHitErrX);

  if (theMonitor) {
    theMonitor->fillDerivatives(aRecHit, newLocalDerivsX, nLocal, newGlobDerivsX, nGlobal, &(globalLabels[0]));
    theMonitor->fillResiduals(
        aRecHit, refTrajPtr->trajectoryStates()[iTrajHit], iTrajHit, newResidX, newHitErrX, false);
  }
  const bool isReal2DHit = this->is2D(aRecHit);  // strip is 1D (except matched hits)
  if (isReal2DHit) {
    theMille->mille(nLocal, newLocalDerivsY, nGlobal, newGlobDerivsY, &(globalLabels[0]), newResidY, newHitErrY);
    if (theMonitor) {
      theMonitor->fillDerivatives(aRecHit, newLocalDerivsY, nLocal, newGlobDerivsY, nGlobal, &(globalLabels[0]));
      theMonitor->fillResiduals(
          aRecHit, refTrajPtr->trajectoryStates()[iTrajHit], iTrajHit, newResidY, newHitErrY, true);  // true: y
    }
  }

  return (isReal2DHit ? 2 : 1);
}

//__________________________________________________________________________________________________
void MillePedeAlignmentAlgorithm ::addVirtualMeas(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
                                                  unsigned int iVirtualMeas) {
  Eigen::Matrix<double, 1, 1> aHitCovarianceM;
  Eigen::Matrix<float, 1, 1> aHitResidualsM;
  Eigen::Matrix<float, 1, Eigen::Dynamic> aLocalDerivativesM{1, refTrajPtr->derivatives().num_col()};
  // below method fills above 3 'matrices'
  this->addRefTrackVirtualMeas1D(refTrajPtr, iVirtualMeas, aHitCovarianceM, aHitResidualsM, aLocalDerivativesM);

  // no global parameters (use dummy 0)
  auto aGlobalDerivativesM = Eigen::Matrix<float, 1, 1>::Zero();

  float newResidX = aHitResidualsM(0, 0);
  float newHitErrX = TMath::Sqrt(aHitCovarianceM(0, 0));
  std::vector<float> newLocalDerivsX(aLocalDerivativesM.size());
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(
      newLocalDerivsX.data(), aLocalDerivativesM.rows(), aLocalDerivativesM.cols()) = aLocalDerivativesM;

  std::vector<float> newGlobDerivsX(aGlobalDerivativesM.size());
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(
      newGlobDerivsX.data(), aGlobalDerivativesM.rows(), aGlobalDerivativesM.cols()) = aGlobalDerivativesM;

  const int nLocal = aLocalDerivativesM.cols();
  const int nGlobal = 0;

  theMille->mille(nLocal, newLocalDerivsX.data(), nGlobal, newGlobDerivsX.data(), &nGlobal, newResidX, newHitErrX);
}

//____________________________________________________
void MillePedeAlignmentAlgorithm::addLaserData(const EventInfo &eventInfo,
                                               const TkFittedLasBeamCollection &lasBeams,
                                               const TsosVectorCollection &lasBeamTsoses) {
  TsosVectorCollection::const_iterator iTsoses = lasBeamTsoses.begin();
  for (TkFittedLasBeamCollection::const_iterator iBeam = lasBeams.begin(), iEnd = lasBeams.end(); iBeam != iEnd;
       ++iBeam, ++iTsoses) {  // beam/tsoses parallel!

    edm::LogInfo("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::addLaserData"
                              << "Beam " << iBeam->getBeamId() << " with " << iBeam->parameters().size()
                              << " parameters and " << iBeam->getData().size() << " hits.\n There are "
                              << iTsoses->size() << " TSOSes.";

    this->addLasBeam(eventInfo, *iBeam, *iTsoses);
  }
}

//____________________________________________________
void MillePedeAlignmentAlgorithm::addLasBeam(const EventInfo &eventInfo,
                                             const TkFittedLasBeam &lasBeam,
                                             const std::vector<TrajectoryStateOnSurface> &tsoses) {
  AlignmentParameters *dummyPtr = nullptr;                                          // for globalDerivativesHierarchy()
  std::vector<float> lasLocalDerivsX;                                               // buffer for local derivatives
  const unsigned int beamLabel = thePedeLabels->lasBeamLabel(lasBeam.getBeamId());  // for global par

  for (unsigned int iHit = 0; iHit < tsoses.size(); ++iHit) {
    if (!tsoses[iHit].isValid())
      continue;
    // clear buffer
    theFloatBufferX.clear();
    theFloatBufferY.clear();
    theIntBuffer.clear();
    lasLocalDerivsX.clear();
    // get alignables and global parameters
    const SiStripLaserRecHit2D &hit = lasBeam.getData()[iHit];
    AlignableDetOrUnitPtr lasAli(theAlignableNavigator->alignableFromDetId(hit.getDetId()));
    this->globalDerivativesHierarchy(
        eventInfo, tsoses[iHit], lasAli, lasAli, theFloatBufferX, theFloatBufferY, theIntBuffer, dummyPtr);
    // fill derivatives vector from derivatives matrix
    for (unsigned int nFitParams = 0; nFitParams < static_cast<unsigned int>(lasBeam.parameters().size());
         ++nFitParams) {
      const float derivative = lasBeam.derivatives()[iHit][nFitParams];
      if (nFitParams < lasBeam.firstFixedParameter()) {  // first local beam parameters
        lasLocalDerivsX.push_back(derivative);
      } else {  // now global ones
        const unsigned int numPar = nFitParams - lasBeam.firstFixedParameter();
        theIntBuffer.push_back(thePedeLabels->parameterLabel(beamLabel, numPar));
        theFloatBufferX.push_back(derivative);
      }
    }  // end loop over parameters

    const float residual = hit.localPosition().x() - tsoses[iHit].localPosition().x();
    // error from file or assume 0.003
    const float error = 0.003;  // hit.localPositionError().xx(); sqrt???

    theMille->mille(lasLocalDerivsX.size(),
                    &(lasLocalDerivsX[0]),
                    theFloatBufferX.size(),
                    &(theFloatBufferX[0]),
                    &(theIntBuffer[0]),
                    residual,
                    error);
  }  // end of loop over hits

  theMille->end();
}

void MillePedeAlignmentAlgorithm::addPxbSurvey(const edm::ParameterSet &pxbSurveyCfg) {
  // do some printing, if requested
  const bool doOutputOnStdout(pxbSurveyCfg.getParameter<bool>("doOutputOnStdout"));
  if (doOutputOnStdout) {
    edm::LogInfo("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::addPxbSurvey"
                              << "# Output from addPxbSurvey follows below because "
                              << "doOutputOnStdout is set to True";
  }

  // instantiate a dicer object
  SurveyPxbDicer dicer(pxbSurveyCfg.getParameter<std::vector<edm::ParameterSet> >("toySurveyParameters"),
                       pxbSurveyCfg.getParameter<unsigned int>("toySurveySeed"));
  std::ofstream outfile(pxbSurveyCfg.getUntrackedParameter<std::string>("toySurveyFile").c_str());

  // read data from file
  std::vector<SurveyPxbImageLocalFit> measurements;
  std::string filename(pxbSurveyCfg.getParameter<edm::FileInPath>("infile").fullPath());
  SurveyPxbImageReader<SurveyPxbImageLocalFit> reader(filename, measurements, 800);

  // loop over photographs (=measurements) and perform the fit
  for (std::vector<SurveyPxbImageLocalFit>::size_type i = 0; i != measurements.size(); i++) {
    if (doOutputOnStdout) {
      edm::LogInfo("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::addPxbSurvey"
                                << "Module " << i << ": ";
    }

    // get the Alignables and their surfaces
    AlignableDetOrUnitPtr mod1(theAlignableNavigator->alignableFromDetId(measurements[i].getIdFirst()));
    AlignableDetOrUnitPtr mod2(theAlignableNavigator->alignableFromDetId(measurements[i].getIdSecond()));
    const AlignableSurface &surf1 = mod1->surface();
    const AlignableSurface &surf2 = mod2->surface();

    // the position of the fiducial points in local frame of a PXB module
    const LocalPoint fidpoint0(-0.91, +3.30);
    const LocalPoint fidpoint1(+0.91, +3.30);
    const LocalPoint fidpoint2(+0.91, -3.30);
    const LocalPoint fidpoint3(-0.91, -3.30);

    // We choose the local frame of the first module as reference,
    // so take the fidpoints of the second module and calculate their
    // positions in the reference frame
    const GlobalPoint surf2point0(surf2.toGlobal(fidpoint0));
    const GlobalPoint surf2point1(surf2.toGlobal(fidpoint1));
    const LocalPoint fidpoint0inSurf1frame(surf1.toLocal(surf2point0));
    const LocalPoint fidpoint1inSurf1frame(surf1.toLocal(surf2point1));

    // Create the vector for the fit
    SurveyPxbImageLocalFit::fidpoint_t fidpointvec;
    fidpointvec.push_back(fidpoint0inSurf1frame);
    fidpointvec.push_back(fidpoint1inSurf1frame);
    fidpointvec.push_back(fidpoint2);
    fidpointvec.push_back(fidpoint3);

    // if toy survey is requested, dice the values now
    if (pxbSurveyCfg.getParameter<bool>("doToySurvey")) {
      dicer.doDice(fidpointvec, measurements[i].getIdPair(), outfile);
    }

    // do the fit
    measurements[i].doFit(fidpointvec, thePedeLabels->alignableLabel(mod1), thePedeLabels->alignableLabel(mod2));
    SurveyPxbImageLocalFit::localpars_t a;  // local pars from fit
    a = measurements[i].getLocalParameters();
    const SurveyPxbImageLocalFit::value_t chi2 = measurements[i].getChi2();

    // do some reporting, if requested
    if (doOutputOnStdout) {
      edm::LogInfo("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::addPxbSurvey"
                                << "a: " << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3]
                                << " S= " << sqrt(a[2] * a[2] + a[3] * a[3]) << " phi= " << atan(a[3] / a[2])
                                << " chi2= " << chi2 << std::endl;
    }
    if (theMonitor) {
      theMonitor->fillPxbSurveyHistsChi2(chi2);
      theMonitor->fillPxbSurveyHistsLocalPars(a[0], a[1], sqrt(a[2] * a[2] + a[3] * a[3]), atan(a[3] / a[2]));
    }

    // pass the results from the local fit to mille
    for (SurveyPxbImageLocalFit::count_t j = 0; j != SurveyPxbImageLocalFit::nMsrmts; j++) {
      theMille->mille((int)measurements[i].getLocalDerivsSize(),
                      measurements[i].getLocalDerivsPtr(j),
                      (int)measurements[i].getGlobalDerivsSize(),
                      measurements[i].getGlobalDerivsPtr(j),
                      measurements[i].getGlobalDerivsLabelPtr(j),
                      measurements[i].getResiduum(j),
                      measurements[i].getSigma(j));
    }
    theMille->end();
  }
  outfile.close();
}

bool MillePedeAlignmentAlgorithm::areIOVsSpecified() const {
  const auto runRangeSelection = theConfig.getUntrackedParameter<edm::VParameterSet>("RunRangeSelection");

  if (runRangeSelection.empty())
    return false;

  const auto runRanges =
      align::makeNonOverlappingRunRanges(runRangeSelection, cond::timeTypeSpecs[cond::runnumber].beginValue);

  return !(runRanges.empty());
}
