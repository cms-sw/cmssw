/**
 * \file MillePedeAlignmentAlgorithm.cc
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.16.2.7 $
 *  $Date: 2007/08/15 08:38:19 $
 *  (last update by $Author: flucke $)
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
// in header, too
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
// end in header, too

//#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeMonitor.h"
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeAlignmentAlgorithm.h"
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariables.h"
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariablesIORoot.h"
#include "Mille.h"       // 'unpublished' interface located in src
#include "PedeSteerer.h" // dito
#include "PedeReader.h" // dito

#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectoryBase.h"
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryPlugin.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORoot.h"

#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h>
#include <Geometry/CommonDetUnit/interface/GeomDetType.h>

#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"

#include <fstream>
#include <vector>
#include <sstream>

#include <TMath.h>

typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;
typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;

// Constructor ----------------------------------------------------------------
//____________________________________________________
MillePedeAlignmentAlgorithm::MillePedeAlignmentAlgorithm(const edm::ParameterSet &cfg) :
  AlignmentAlgorithmBase(cfg), 
  theConfig(cfg), theMode(this->decodeMode(theConfig.getUntrackedParameter<std::string>("mode"))),
  theDir(theConfig.getUntrackedParameter<std::string>("fileDir")),
  theAlignmentParameterStore(0), theAlignables(), theAlignableNavigator(0),
  theMonitor(0), theMille(0), thePedeSteer(0), theTrajectoryFactory(0),
  theMinNumHits(cfg.getParameter<int>("minNumHits")),
  theUseTrackTsos(cfg.getParameter<bool>("useTrackTsos"))
{
  if (!theDir.empty() && theDir.find_last_of('/') != theDir.size()-1) theDir += '/';// may need '/'
  edm::LogInfo("Alignment") << "@SUB=MillePedeAlignmentAlgorithm" << "Start in mode '"
                            << theConfig.getUntrackedParameter<std::string>("mode")
                            << "' with output directory '" << theDir << "'.";
}

// Destructor ----------------------------------------------------------------
//____________________________________________________
MillePedeAlignmentAlgorithm::~MillePedeAlignmentAlgorithm()
{
  delete theAlignableNavigator;
  delete theMille;
  delete theMonitor;
  delete thePedeSteer;
  delete theTrajectoryFactory;
}

// Call at beginning of job ---------------------------------------------------
//____________________________________________________
void MillePedeAlignmentAlgorithm::initialize(const edm::EventSetup &setup, 
                                             AlignableTracker *tracker, AlignableMuon *muon,
                                             AlignmentParameterStore *store)
{
  if (muon) {
    edm::LogWarning("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::initialize"
                               << "Running with AlignabeMuon not yet tested.";
  }

  theAlignableNavigator = new AlignableNavigator(tracker, muon);
  theAlignmentParameterStore = store;
  theAlignables = theAlignmentParameterStore->alignables();

  edm::ParameterSet pedeSteerCfg(theConfig.getParameter<edm::ParameterSet>("pedeSteerer"));
  thePedeSteer = new PedeSteerer(tracker, muon, theAlignmentParameterStore, pedeSteerCfg, theDir);
  // After (!) PedeSteerer which uses the SelectionUserVariables attached to the parameters:
  this->buildUserVariables(theAlignables); // for hit statistics and/or pede result

  if (this->isMode(myMilleBit)) {
    if (!theConfig.getParameter<std::vector<std::string> >("mergeBinaryFiles").empty() ||
        !theConfig.getParameter<std::vector<std::string> >("mergeTreeFiles").empty()) {
      throw cms::Exception("BadConfig")
        << "'vstring mergeTreeFiles' and 'vstring mergeBinaryFiles' must be empty for "
        << "modes running mille.";
    }
    theMille = new Mille((theDir + theConfig.getParameter<std::string>("binaryFile")).c_str());
    const std::string moniFile(theConfig.getUntrackedParameter<std::string>("monitorFile"));
    if (moniFile.size()) theMonitor = new MillePedeMonitor((theDir + moniFile).c_str());

    // Get trajectory factory. In case nothing found, FrameWork will throw...
    const edm::ParameterSet fctCfg(theConfig.getParameter<edm::ParameterSet>("TrajectoryFactory"));
    const std::string fctName(fctCfg.getParameter<std::string>("TrajectoryFactoryName"));
    theTrajectoryFactory = TrajectoryFactoryPlugin::get()->create(fctName, fctCfg);
  }

  // FIXME: for PlotMillePede hit statistics stuff we also might want doIO(0)... ?
  if (this->isMode(myPedeSteerBit) // for pedeRun and pedeRead we might want to merge
      || !theConfig.getParameter<std::vector<std::string> >("mergeTreeFiles").empty()) {
    this->doIO(0);
  }
}

// Call at end of job ---------------------------------------------------------
//____________________________________________________
void MillePedeAlignmentAlgorithm::terminate()
{
  delete theMille;// delete to close binary before running pede below (flush would be enough...)
  theMille = 0;

  std::vector<std::string> files;
  if (this->isMode(myMilleBit) || !theConfig.getParameter<std::string>("binaryFile").empty()) {
    files.push_back(theDir + theConfig.getParameter<std::string>("binaryFile"));
  } else {
    const std::vector<std::string> plainFiles
      (theConfig.getParameter<std::vector<std::string> >("mergeBinaryFiles"));
    for (std::vector<std::string>::const_iterator i = plainFiles.begin(), iEnd = plainFiles.end();
         i != iEnd; ++i) {
      files.push_back(theDir + *i);
    }
  }
  const std::string masterSteer(thePedeSteer->buildMasterSteer(files));// do only if myPedeSteerBit?
  bool pedeOk = true;
  if (this->isMode(myPedeRunBit)) {
    pedeOk = thePedeSteer->runPede(masterSteer);
  }
  
  if (this->isMode(myPedeReadBit)) {
    if (!pedeOk || !this->readFromPede()) {
      edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::terminate"
                                 << "Problems running pede or reading result, but applying!";
    }
    // FIXME: problem if what is read in does not correspond to store
    theAlignmentParameterStore->applyParameters();
    // thePedeSteer->correctToReferenceSystem(); // Already done before, here for possible rounding reasons...??
  }

  if (this->isMode(myMilleBit)) { // if mille was run, we store trees with suffix _1...
    this->doIO(1);
  } else if (this->isMode(myPedeReadBit)) {// if pede runs otherwise, we use _2 (=> probably merge)
    this->doIO(2);
  }

  // FIXME: should we delete here or in destructor?
  delete theAlignableNavigator;
  theAlignableNavigator = 0;
  delete theMonitor;
  theMonitor = 0;
  delete thePedeSteer;
  thePedeSteer = 0;
  delete theTrajectoryFactory;
  theTrajectoryFactory = 0;
}

// Run the algorithm on trajectories and tracks -------------------------------
//____________________________________________________
void MillePedeAlignmentAlgorithm::run(const edm::EventSetup &setup,
				      const ConstTrajTrackPairCollection &tracks) 
{
  if (!this->isMode(myMilleBit)) return; // no theMille created...

  typedef TrajectoryFactoryBase::ReferenceTrajectoryCollection RefTrajColl;
  const RefTrajColl trajectories(theTrajectoryFactory->trajectories(setup, tracks));
  // Assume that same container size means same order... :-(
  const bool canUseTrack = (trajectories.size() == tracks.size());
  const bool useTrackTsosBack = theUseTrackTsos;
  if (!canUseTrack) theUseTrackTsos = false;

  std::vector<TrajectoryStateOnSurface> trackTsos; // some buffer...
  // loop over ReferenceTrajectoryCollection and possibly over tracks  
  ConstTrajTrackPairCollection::const_iterator iTrajTrack = tracks.begin();
  for (RefTrajColl::const_iterator iRefTraj = trajectories.begin(), iRefTrajE = trajectories.end();
       iRefTraj != iRefTrajE; ++iRefTraj) {
    if (canUseTrack) {
      if (!this->orderedTsos((*iTrajTrack).first, trackTsos)) continue; // first is Trajectory*
      if (theMonitor) theMonitor->fillTrack((*iTrajTrack).second); // second is reco::Track*
    } else {
      trackTsos.clear();
      trackTsos.resize((*iTrajTrack).second->recHitsSize());
    }

    RefTrajColl::value_type refTrajPtr = *iRefTraj; 
    if (!refTrajPtr->isValid()) continue; // currently e.g. if any invalid hit (FIXME for cosmic?)
    
    std::vector<AlignmentParameters*> parVec(refTrajPtr->recHits().size());//to add hits if all fine
    std::vector<bool> validHitVecY(refTrajPtr->recHits().size()); // collect hit statistics...
    int nValidHitsX = 0;                                // ...assuming that there are no y-only hits
    // Use recHits from ReferenceTrajectory (since they have the right order!):
    for (unsigned int iHit = 0; iHit < refTrajPtr->recHits().size(); ++iHit) {
      const int flagXY = this->addGlobalDerivatives(refTrajPtr,iHit,trackTsos[iHit],parVec[iHit]);
      if (flagXY < 0) { // problem
        nValidHitsX = -1;
        break;
      } else { // hit is fine, increase x/y statistics
        if (flagXY >= 1) ++nValidHitsX;
        validHitVecY[iHit] = (flagXY >= 2);
      } 
    } // end loop on hits
    
    if (nValidHitsX >= theMinNumHits) { // enough 'good' alignables hit: increase the hit statistics
      unsigned int nValidHitsY = 0;
      for (unsigned int iHit = 0; iHit < validHitVecY.size(); ++iHit) {
        if (!parVec[iHit]) continue; // in case a non-selected alignable was hit (flagXY == 0)
        MillePedeVariables *mpVar = static_cast<MillePedeVariables*>(parVec[iHit]->userVariables());
        mpVar->increaseHitsX(); // every hit has an x-measurement, cf. above...
        if (validHitVecY[iHit]) {
	  mpVar->increaseHitsY();
	  ++nValidHitsY;
	}
      }
      theMille->end();
      if (canUseTrack && theMonitor) {
        theMonitor->fillUsedTrack((*iTrajTrack).second, nValidHitsX, nValidHitsY);
      }
    } else {
      theMille->kill();
    }
    if (canUseTrack) ++iTrajTrack;
  } // end of reference trajectory and track loop

  theUseTrackTsos = useTrackTsosBack;
}

//____________________________________________________
int MillePedeAlignmentAlgorithm::addGlobalDerivatives
(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr, unsigned int iHit,
 const TrajectoryStateOnSurface &trackTsos, AlignmentParameters *&params)
{
  params = 0;
  theFloatBufferX.clear();
  theFloatBufferY.clear();
  theIntBuffer.clear();

  const TrajectoryStateOnSurface &tsos = 
    (theUseTrackTsos ? trackTsos : refTrajPtr->trajectoryStates()[iHit]);
  const ConstRecHitPointer &recHitPtr = refTrajPtr->recHits()[iHit];
  // get AlignableDet/Unit for this hit
  AlignableDetOrUnitPtr alidet(theAlignableNavigator->alignableFromGeomDet(recHitPtr->det()));
  const bool is2DHit = this->is2D(recHitPtr);

  if (!this->globalDerivativesHierarchy(tsos, alidet, alidet, is2DHit,// 2x alidet, sic!
					theFloatBufferX, theFloatBufferY, theIntBuffer,
					params)) {
    return -1; // problem
  } else if (theFloatBufferX.empty()) {
    return 0; // empty for X: no alignable for hit
  } else {
    this->callMille(refTrajPtr, iHit, kLocalX, theFloatBufferX, theIntBuffer);
    if (is2DHit) {
      this->callMille(refTrajPtr, iHit, kLocalY, theFloatBufferY, theIntBuffer);
      return 2; // 2D information used
    } else { 
      return 1; // 1D information used
    }
  }
}

//____________________________________________________
bool MillePedeAlignmentAlgorithm
::globalDerivativesHierarchy(const TrajectoryStateOnSurface &tsos,
                             Alignable *ali, const AlignableDetOrUnitPtr &alidet, bool is2DHit,
                             std::vector<float> &globalDerivativesX,
                             std::vector<float> &globalDerivativesY,
                             std::vector<int> &globalLabels,
                             AlignmentParameters *&lowestParams) const
{
  // derivatives and labels are recursively attached
  if (!ali) return true; // no mother might be OK

  if (theMonitor && alidet != ali) theMonitor->fillFrameToFrame(alidet, ali);

  AlignmentParameters *params = ali->alignmentParameters();
  if (params) {
    if (!lowestParams) lowestParams = params; // set parameters of lowest level

    const unsigned int alignableLabel = thePedeSteer->alignableLabel(ali);
    if (0 == alignableLabel) { // FIXME: what about regardAllHits in Markus' code?
      edm::LogWarning("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::globalDerivativesHierarchy"
                                   << "Label not found, skip Alignable.";
      return false;
    }
    
    const std::vector<bool> &selPars = params->selector();
    const AlgebraicMatrix derivs(params->derivatives(tsos, alidet));
    // cols: 2, i.e. x&y, rows: parameters, usually RigidBodyAlignmentParameters::N_PARAM
    for (unsigned int iSel = 0; iSel < selPars.size(); ++iSel) {
      if (selPars[iSel]) {
        globalDerivativesX.push_back(derivs[iSel][kLocalX]
				     /thePedeSteer->cmsToPedeFactor(iSel));
        globalLabels.push_back(thePedeSteer->parameterLabel(alignableLabel, iSel));
        if (is2DHit) {
	  globalDerivativesY.push_back(derivs[iSel][kLocalY]
				       /thePedeSteer->cmsToPedeFactor(iSel));
	}
      }
    }
    // Exclude mothers if Alignable selected to be no part of a hierarchy:
    if (thePedeSteer->isNoHiera(ali)) return true;
  }
  // Call recursively for mother, will stop if mother == 0:
  return this->globalDerivativesHierarchy(tsos, ali->mother(), alidet, is2DHit,
                                          globalDerivativesX, globalDerivativesY,
					  globalLabels, lowestParams);
}

//____________________________________________________
void MillePedeAlignmentAlgorithm::callMille
(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
 unsigned int iTrajHit, MeasurementDirection xOrY,
 const std::vector<float> &globalDerivatives, const std::vector<int> &globalLabels)
{
  const unsigned int xyIndex = iTrajHit*2 + xOrY;
  // FIXME: here for residuum and sigma we could use KALMAN-Filter results
  const float residuum =
    refTrajPtr->measurements()[xyIndex] - refTrajPtr->trajectoryPositions()[xyIndex];
  const float covariance = refTrajPtr->measurementErrors()[xyIndex][xyIndex];
  const float sigma = (covariance > 0. ? TMath::Sqrt(covariance) : 0.);

  const AlgebraicMatrix &locDerivMatrix = refTrajPtr->derivatives();

  std::vector<float> localDerivs(locDerivMatrix.num_col());
  for (unsigned int i = 0; i < localDerivs.size(); ++i) {
    localDerivs[i] = locDerivMatrix[xyIndex][i];
  }

  // &(vector[0]) is valid - as long as vector is not empty 
  // cf. http://www.parashift.com/c++-faq-lite/containers.html#faq-34.3
  theMille->mille(localDerivs.size(), &(localDerivs[0]),
		  globalDerivatives.size(), &(globalDerivatives[0]), &(globalLabels[0]),
		  residuum, sigma);
  if (theMonitor) {
    theMonitor->fillDerivatives(refTrajPtr->recHits()[iTrajHit],localDerivs, globalDerivatives,
				(xOrY == kLocalY));
    theMonitor->fillResiduals(refTrajPtr->recHits()[iTrajHit],
			      refTrajPtr->trajectoryStates()[iTrajHit],
			      iTrajHit, residuum, sigma, (xOrY == kLocalY));
  }
}

//____________________________________________________
bool MillePedeAlignmentAlgorithm::is2D(const ConstRecHitPointer &recHit) const
{
  // FIXME: Check whether this is a reliable and recommended way to find out...

  if (recHit->dimension() < 2) {
    return false; // some muon stuff really has RecHit1D
  } else if (recHit->detUnit()) { // detunit in strip is 1D, in pixel 2D 
    return recHit->detUnit()->type().isTrackerPixel();
  } else { // stereo strips  (FIXME: endcap trouble due to non-parallel strips (wedge sensors)?)
    if (dynamic_cast<const ProjectedSiStripRecHit2D*>(recHit->hit())) { // check persistent hit
      // projected: 1D measurement on 'glued' module
      return false; // (FIXME: if it's the stereo module, x and measurement not parallel...)
    } else {
      return true;
    }
  }
}

//__________________________________________________________________________________________________
bool MillePedeAlignmentAlgorithm::readFromPede()
{
  bool allEmpty = this->areEmptyParams(theAlignables);
  
  PedeReader reader(theConfig.getParameter<edm::ParameterSet>("pedeReader"), *thePedeSteer);
  std::vector<Alignable*> alis;
  bool okRead = reader.read(alis);
  bool numMatch = true;

  std::stringstream out("Read ");
  out << alis.size() << " alignables";
  if (alis.size() != theAlignables.size()) {
    out << " while " << theAlignables.size() << " in store";
    numMatch = false; // FIXME: Should we check one by one? Or transfer 'alis' to the store?
  }
  if (!okRead) out << ", but problems in reading"; 
  if (!allEmpty) out << ", possibly overwriting previous settings";
  out << ".";

  if (okRead && allEmpty && numMatch) {
    edm::LogInfo("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::readFromPede" << out.str();
    return true;
  } else {
    edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::readFromPede" << out.str();
    return false;
  }
}

//__________________________________________________________________________________________________
bool MillePedeAlignmentAlgorithm::areEmptyParams(const std::vector<Alignable*> &alignables) const
{
  
  for (std::vector<Alignable*>::const_iterator iAli = alignables.begin();
       iAli != alignables.end(); ++iAli) {
    const AlignmentParameters *params = (*iAli)->alignmentParameters();
    if (params) {
      const AlgebraicVector &parVec(params->parameters());
      const AlgebraicMatrix &parCov(params->covariance());
      for (int i = 0; i < parVec.num_row(); ++i) {
        if (parVec[i] != 0.) return false;
        for (int j = i; j < parCov.num_col(); ++j) {
          if (parCov[i][j] != 0.) return false;
        }
      }
    }
  }
  
  return true;
}

//__________________________________________________________________________________________________
unsigned int MillePedeAlignmentAlgorithm::doIO(int loop) const
{

  const std::string outFile(theDir + theConfig.getParameter<std::string>("treeFile"));

  AlignmentIORoot aliIO;
  int ioerr = 0;
  unsigned int result = 0;
  if (loop == 0) {
    aliIO.writeAlignableOriginalPositions(theAlignables, outFile.c_str(), loop, false, ioerr);
    if (ioerr) {
      edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO"
                                 << "Problem " << ioerr << " in writeAlignableOriginalPositions";
      ++result;
    }
  } else {
    if (loop > 1) {
      const std::vector<std::string> inFiles
        (theConfig.getParameter<std::vector<std::string> >("mergeTreeFiles"));
      const std::vector<std::string> binFiles
        (theConfig.getParameter<std::vector<std::string> >("mergeBinaryFiles"));
      if (inFiles.size() != binFiles.size()) {
        edm::LogWarning("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO"
                                     << "'vstring mergeTreeFiles' and 'vstring mergeBinaryFiles' "
                                     << "differ in size";
      }
      this->addHitStatistics(loop - 1, outFile, inFiles);
    }
    MillePedeVariablesIORoot millePedeIO;
    millePedeIO.writeMillePedeVariables(theAlignables, outFile.c_str(), loop, false, ioerr);
    if (ioerr) {
      edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO"
                                 << "Problem " << ioerr << " writing MillePedeVariables";
      ++result;
    }
// // problem with following writeOrigRigidBodyAlignmentParameters
//     aliIO.writeAlignmentParameters(theAlignables, outFile.c_str(), loop, false, ioerr);
//     if (ioerr) {
//       edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO"
//                                  << "Problem " << ioerr << " in writeAlignmentParameters, " << loop;
//       ++result;
//     }
  }
  
  aliIO.writeOrigRigidBodyAlignmentParameters(theAlignables, outFile.c_str(), loop, false, ioerr);
  if (ioerr) {
    edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO" << "Problem " << ioerr
			       << " in writeOrigRigidBodyAlignmentParameters, " << loop;
    ++result;
  }
  aliIO.writeAlignableAbsolutePositions(theAlignables, outFile.c_str(), loop, false, ioerr);
  if (ioerr) {
    edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO" << "Problem " << ioerr
                               << " in writeAlignableAbsolutePositions, " << loop;
    ++result;
  }
  aliIO.writeAlignableRelativePositions(theAlignables, outFile.c_str(), loop, false, ioerr);
  if (ioerr) {
    edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::doIO" << "Problem " << ioerr
                               << " in writeAlignableRelativePositions, " << loop;
    ++result;
  }

  return result;
}

//__________________________________________________________________________________________________
void MillePedeAlignmentAlgorithm::buildUserVariables(const std::vector<Alignable*> &alis) const
{
  for (std::vector<Alignable*>::const_iterator iAli = alis.begin(); iAli != alis.end(); ++iAli) {
    AlignmentParameters *params = (*iAli)->alignmentParameters();
    if (!params) {
      throw cms::Exception("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::buildUserVariables"
                                        << "No parameters for alignable";
    }
    params->setUserVariables(new MillePedeVariables(params->size()));
  }
}

//__________________________________________________________________________________________________
unsigned int MillePedeAlignmentAlgorithm::decodeMode(const std::string &mode) const
{
  if (mode == "full") {
    return myMilleBit + myPedeSteerBit + myPedeRunBit + myPedeReadBit;
  } else if (mode == "mille") {
    return myMilleBit; // + myPedeSteerBit; // sic! Including production of steerig file. NO!
  } else if (mode == "pede") {
    return myPedeSteerBit + myPedeRunBit + myPedeReadBit;
  } else if (mode == "pedeSteer") {
    return myPedeSteerBit;
  } else if (mode == "pedeRun") {
    return myPedeRunBit + myPedeReadBit; // sic! Including reading of result.
  } else if (mode == "pedeRead") {
    return myPedeReadBit;
  }

  throw cms::Exception("BadConfig") 
    << "Unknown mode '" << mode  
    << "', use 'full', 'mille', 'pede', 'pedeRun', 'pedeSteer' or 'pedeRead'.";

  return 0;
}

//__________________________________________________________________________________________________
bool MillePedeAlignmentAlgorithm::addHitStatistics(int fromLoop, const std::string &outFile,
                                                   const std::vector<std::string> &inFiles) const
{
  bool allOk = true;
  int ierr = 0;
  MillePedeVariablesIORoot millePedeIO;
  if (inFiles.empty()) {
    const std::vector<AlignmentUserVariables*> mpVars =
      millePedeIO.readMillePedeVariables(theAlignables, outFile.c_str(), fromLoop, ierr);
    if (ierr || !this->addHits(theAlignables, mpVars)) {
      edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::addHitStatistics"
                                 << "Error " << ierr << " reading from " << outFile 
                                 << ", tree " << fromLoop << ", or problems in addHits";
      allOk = false;
    }
    for (std::vector<AlignmentUserVariables*>::const_iterator i = mpVars.begin();
         i != mpVars.end(); ++i){
      delete *i; // clean created objects
    }
  } else {
    for (std::vector<std::string>::const_iterator iFile = inFiles.begin();
         iFile != inFiles.end(); ++iFile) {
      const std::string inFile(theDir + *iFile); 
      const std::vector<AlignmentUserVariables*> mpVars =
        millePedeIO.readMillePedeVariables(theAlignables, inFile.c_str(), fromLoop, ierr);
      if (ierr || !this->addHits(theAlignables, mpVars)) {
        edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::addHitStatistics"
                                   << "Error " << ierr << " reading from " << inFile 
                                   << ", tree " << fromLoop << ", or problems in addHits";
        allOk = false;
      }
      for (std::vector<AlignmentUserVariables*>::const_iterator i = mpVars.begin();
           i != mpVars.end(); ++i) {
        delete *i; // clean created objects
      }
    }
  }

  return allOk;
}

//__________________________________________________________________________________________________
bool MillePedeAlignmentAlgorithm::addHits(const std::vector<Alignable*> &alis,
                                          const std::vector<AlignmentUserVariables*> &mpVars) const
{
  bool allOk = (mpVars.size() == alis.size());
  std::vector<AlignmentUserVariables*>::const_iterator iUser = mpVars.begin();
  for (std::vector<Alignable*>::const_iterator iAli = alis.begin(); 
       iAli != alis.end() && iUser != mpVars.end(); ++iAli, ++iUser) {
    MillePedeVariables *mpVarNew = dynamic_cast<MillePedeVariables*>(*iUser);
    AlignmentParameters *ps = (*iAli)->alignmentParameters();
    MillePedeVariables *mpVarOld = (ps ? dynamic_cast<MillePedeVariables*>(ps->userVariables()) : 0);
    if (!mpVarNew || !mpVarOld || mpVarOld->size() != mpVarNew->size()) {
      allOk = false;
      continue; // FIXME error etc.?
    }

    mpVarOld->increaseHitsX(mpVarNew->hitsX());
    mpVarOld->increaseHitsY(mpVarNew->hitsY());
  }
  
  return allOk;
}


//__________________________________________________________________________________________________
bool MillePedeAlignmentAlgorithm::orderedTsos(const Trajectory *traj, 
                                              std::vector<TrajectoryStateOnSurface> &trackTsos)const
{
  trackTsos.clear();
  // FIXME: if (theUseTrackTsos == false) fill only first/last!
  Trajectory::DataContainer trajMeas(traj->measurements());
  PropagationDirection dir = traj->direction();
  if (dir == oppositeToMomentum) {
    // why does const_reverse_operator not compile?
    for (Trajectory::DataContainer::reverse_iterator rMeas = trajMeas.rbegin();
         rMeas != trajMeas.rend(); ++rMeas) {
      trackTsos.push_back((*rMeas).updatedState());
    }
  } else if (dir == alongMomentum) {
    for (Trajectory::DataContainer::const_iterator iMeas = trajMeas.begin();
         iMeas != trajMeas.end(); ++iMeas) {
      trackTsos.push_back((*iMeas).updatedState());
    }
  } else {
    edm::LogError("Alignment") << "$SUB=MillePedeAlignmentAlgorithm::orderedTsos"
                               << "Trajectory neither along nor opposite to momentum.";
    return false;
  }

  for (std::vector<TrajectoryStateOnSurface>::const_iterator iTraj = trackTsos.begin(),
	 iEnd = trackTsos.end(); iTraj != iEnd; ++iTraj) {
    if (!(*iTraj).isValid()) {
      edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::orderedTsos"
				 << "an invalid  TSOS...?";
      return false;
    }
  }
  

  return true;
}
