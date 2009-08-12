/**
 * \file MillePedeAlignmentAlgorithm.cc
 *
 *  \author    : Gero Flucke
 *  date       : October 2006
 *  $Revision: 1.48 $
 *  $Date: 2009/08/10 16:29:40 $
 *  (last update by $Author: flucke $)
 */

#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeAlignmentAlgorithm.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
// in header, too
// end in header, too

#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeMonitor.h"
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariables.h"
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariablesIORoot.h"
#include "Mille.h"       // 'unpublished' interface located in src
#include "PedeSteerer.h" // dito
#include "PedeReader.h" // dito
#include "PedeLabeler.h" // dito

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryPlugin.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORoot.h"

#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"

// includes to make known that they inherit from Alignable:
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"


#include <Geometry/CommonDetUnit/interface/GeomDetUnit.h>
#include <Geometry/CommonDetUnit/interface/GeomDetType.h>

#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"

#include <fstream>
#include <sstream>
#include <algorithm>

#include <TMath.h>
#include <TMatrixDSymEigen.h>
typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;
typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;

// Constructor ----------------------------------------------------------------
//____________________________________________________
MillePedeAlignmentAlgorithm::MillePedeAlignmentAlgorithm(const edm::ParameterSet &cfg) :
  AlignmentAlgorithmBase(cfg), 
  theConfig(cfg), theMode(this->decodeMode(theConfig.getUntrackedParameter<std::string>("mode"))),
  theDir(theConfig.getUntrackedParameter<std::string>("fileDir")),
  theAlignmentParameterStore(0), theAlignables(), theAlignableNavigator(0),
  theMonitor(0), theMille(0), thePedeLabels(0), thePedeSteer(0),
  theTrajectoryFactory(0),
  theMinNumHits(cfg.getParameter<int>("minNumHits")),
  theMaximalCor2D(cfg.getParameter<double>("max2Dcorrelation")),
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
  delete thePedeLabels;
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
  thePedeLabels = new PedeLabeler(tracker, muon);

  // 1) Create PedeSteerer: correct alignable positions for coordinate system selection
  edm::ParameterSet pedeSteerCfg(theConfig.getParameter<edm::ParameterSet>("pedeSteerer"));
  thePedeSteer = new PedeSteerer(tracker, muon, theAlignmentParameterStore, thePedeLabels,
				 pedeSteerCfg, theDir, !this->isMode(myPedeSteerBit));

  // 2) If requested, directly read in and apply result of previous pede run,
  //    assuming that correction from 1) was also applied to create the result:
  const std::vector<edm::ParameterSet> mprespset
    (theConfig.getParameter<std::vector<edm::ParameterSet> >("pedeReaderInputs"));
  for (std::vector<edm::ParameterSet>::const_iterator iSet = mprespset.begin(), iE = mprespset.end();
       iSet != iE; ++iSet) {
    edm::LogInfo("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::initialize"
                              << "Apply MillePede constants defined by VPSet 'pedeReaderInputs'.";
    this->readFromPede((*iSet), false); // false: do not erase SelectionUserVariables
    theAlignmentParameterStore->applyParameters();
    // Following needed to shut up later warning from checkAliParams? Test!
    // theAlignmentParameterStore->resetParameters();  FIXME
  }

  // 3) Now create steerings with 'final' start position:
  thePedeSteer->buildSubSteer(tracker, muon);

  // After (!) 1-3 of PedeSteerer which uses the SelectionUserVariables attached to the parameters:
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
  if (this->isMode(myPedeRunBit)) {
    thePedeSteer->runPede(masterSteer);
  }
  
  if (this->isMode(myPedeReadBit)) {
    if (!this->readFromPede(theConfig.getParameter<edm::ParameterSet>("pedeReader"), true)) {
      edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::terminate"
                                 << "Problems reading pede result, but applying!";
    }
    // FIXME: problem if what is read in does not correspond to store
    theAlignmentParameterStore->applyParameters();
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
  delete thePedeLabels;
  thePedeLabels = 0;
  delete theTrajectoryFactory;
  theTrajectoryFactory = 0;
}

// Run the algorithm on trajectories and tracks -------------------------------
//____________________________________________________
void MillePedeAlignmentAlgorithm::run(const edm::EventSetup &setup, const EventInfo &eventInfo)
{
  if (!this->isMode(myMilleBit)) return; // no theMille created...
  const ConstTrajTrackPairCollection &tracks = eventInfo.trajTrackPairs_;

  typedef TrajectoryFactoryBase::ReferenceTrajectoryCollection RefTrajColl;
  const RefTrajColl trajectories(theTrajectoryFactory->trajectories(setup, tracks));
  // Assume that same container size means same order... :-(
  const bool canUseTrack = (trajectories.size() == tracks.size());
  const bool useTrackTsosBack = theUseTrackTsos;
  if (!canUseTrack) theUseTrackTsos = false;

  // Loop over ReferenceTrajectoryCollection and possibly over tracks,
  // but in case Ref.-Traj. are not parallel to tracks, first fill monitor for tracks:
  ConstTrajTrackPairCollection::const_iterator iTrajTrack = tracks.begin();
  if (theMonitor) {
    for (; iTrajTrack != tracks.end(); ++iTrajTrack) theMonitor->fillTrack((*iTrajTrack).second);
    iTrajTrack = tracks.begin(); // set back...
  }
  // Now really loop over ReferenceTrajectoryCollection
  std::vector<TrajectoryStateOnSurface> trackTsos; // some buffer...
  for (RefTrajColl::const_iterator iRefTraj = trajectories.begin(), iRefTrajE = trajectories.end();
       iRefTraj != iRefTrajE; ++iRefTraj) {

    RefTrajColl::value_type refTrajPtr = *iRefTraj; 
    if (theMonitor) theMonitor->fillRefTrajectory(refTrajPtr);
    if (!refTrajPtr->isValid()) continue; // currently e.g. if any invalid hit (FIXME for cosmic?)
    
    if (canUseTrack) {
      if (!this->orderedTsos((*iTrajTrack).first, trackTsos)) continue; // first is Trajectory*
    } else {
      trackTsos.clear();
      trackTsos.resize((*iTrajTrack).second->recHitsSize());
    }
//     // write special data to the milleBinary.dat
//     int nPar = 6;
//     std::vector<int> integers(nPar); // filled with 0.
//     std::vector<float> floats(nPar);
//     for (int i = 0; i < nPar; ++i) {
//       floats[i] = refTrajPtr->globalPars()[i];
//     }
//     theMille->special(nPar, &(floats[0]), &(integers[0]));
             
    std::vector<AlignmentParameters*> parVec(refTrajPtr->recHits().size());//to add hits if all fine
    std::vector<bool> validHitVecY(refTrajPtr->recHits().size()); // collect hit statistics...
    int nValidHitsX = 0;                                // ...assuming that there are no y-only hits
    // Use recHits from ReferenceTrajectory (since they have the right order!):
    for (unsigned int iHit = 0; iHit < refTrajPtr->recHits().size(); ++iHit) {
      const int flagXY = this->addMeasurementData(refTrajPtr,iHit,trackTsos[iHit],parVec[iHit]);
      if (flagXY < 0) { // problem
        nValidHitsX = -1;
        break;
      } else { // hit is fine, increase x/y statistics
        if (flagXY >= 1) ++nValidHitsX;
        validHitVecY[iHit] = (flagXY >= 2);
      } 
    } // end loop on hits

// CHK add breakpoints
    for (unsigned int iBp = 0; iBp < refTrajPtr->numberOfBreakPoints(); ++iBp) { this->addBreakPoint(refTrajPtr,iBp); }
    
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
      if (theMonitor) {
        theMonitor->fillUsedTrack((canUseTrack ? iTrajTrack->second : 0), nValidHitsX, nValidHitsY);
      }
    } else {
      theMille->kill();
    }
    if (canUseTrack) ++iTrajTrack;
  } // end of reference trajectory and track loop

  theUseTrackTsos = useTrackTsosBack;
}

//____________________________________________________
void MillePedeAlignmentAlgorithm::endRun(const EndRunInfo &runInfo,
					 const edm::EventSetup &setup)
{
  if(runInfo.tkLasBeams_ && runInfo.tkLasBeamTsoses_){
    // LAS beam treatment
    this->addLaserData(*(runInfo.tkLasBeams_), *(runInfo.tkLasBeamTsoses_));
  }
}

//____________________________________________________
int MillePedeAlignmentAlgorithm::addMeasurementData
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

  // ignore invalid hits    
  if (!recHitPtr->isValid()) return 0;
  
  // get AlignableDet/Unit for this hit
  AlignableDetOrUnitPtr alidet(theAlignableNavigator->alignableFromGeomDet(recHitPtr->det()));
  if (!this->globalDerivativesHierarchy(tsos, alidet, alidet, theFloatBufferX, // 2x alidet, sic!
					theFloatBufferY, theIntBuffer, params)) {
    return -1; // problem
  } else if (theFloatBufferX.empty()) {
    return 0; // empty for X: no alignable for hit
  } else { 
    return this->callMille2D(refTrajPtr, iHit, theIntBuffer, theFloatBufferX, theFloatBufferY);
  }
}

//____________________________________________________
bool MillePedeAlignmentAlgorithm
::globalDerivativesHierarchy(const TrajectoryStateOnSurface &tsos,
                             Alignable *ali, const AlignableDetOrUnitPtr &alidet,
                             std::vector<float> &globalDerivativesX,
                             std::vector<float> &globalDerivativesY,
                             std::vector<int> &globalLabels,
                             AlignmentParameters *&lowestParams) const
{
  // derivatives and labels are recursively attached
  if (!ali) return true; // no mother might be OK

  if (false && theMonitor && alidet != ali) theMonitor->fillFrameToFrame(alidet, ali);

  AlignmentParameters *params = ali->alignmentParameters();
  if (params) {
    if (!lowestParams) lowestParams = params; // set parameters of lowest level

    const unsigned int alignableLabel = thePedeLabels->alignableLabel(ali);
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
        globalLabels.push_back(thePedeLabels->parameterLabel(alignableLabel, iSel));
	globalDerivativesY.push_back(derivs[iSel][kLocalY]
				     /thePedeSteer->cmsToPedeFactor(iSel));
      }
    }
    // Exclude mothers if Alignable selected to be no part of a hierarchy:
    if (thePedeSteer->isNoHiera(ali)) return true;
  }
  // Call recursively for mother, will stop if mother == 0:
  return this->globalDerivativesHierarchy(tsos, ali->mother(), alidet,
                                          globalDerivativesX, globalDerivativesY,
					  globalLabels, lowestParams);
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
// 		  globalDerivatives.size(), &(globalDerivatives[0]), &(globalLabels[0]),
// 		  residuum, sigma);
//   if (theMonitor) {
//     theMonitor->fillDerivatives(refTrajPtr->recHits()[iTrajHit],localDerivs, globalDerivatives,
// 				(xOrY == kLocalY));
//     theMonitor->fillResiduals(refTrajPtr->recHits()[iTrajHit],
// 			      refTrajPtr->trajectoryStates()[iTrajHit],
// 			      iTrajHit, residuum, sigma, (xOrY == kLocalY));
//   }
// }

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
      return false;
    } else {
      return true;
    }
  }
}

//__________________________________________________________________________________________________
bool MillePedeAlignmentAlgorithm::readFromPede(const edm::ParameterSet &mprespset, bool setUserVars)
{
  bool allEmpty = this->areEmptyParams(theAlignables);

  PedeReader reader(mprespset,
		    *thePedeSteer, *thePedeLabels);
  std::vector<Alignable*> alis;
  bool okRead = reader.read(alis, setUserVars);
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
    MillePedeVariables *userVars = new MillePedeVariables(params->size(), thePedeLabels->alignableLabel(*iAli));
    params->setUserVariables(userVars);
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
    return myPedeSteerBit + myPedeRunBit + myPedeReadBit; // sic! Including steering and reading of result.
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

//__________________________________________________________________________________________________
void MillePedeAlignmentAlgorithm::makeGlobDerivMatrix(const std::vector<float> &globalDerivativesx,
                                                      const std::vector<float> &globalDerivativesy,
                                                      TMatrixF &aGlobalDerivativesM)
{

  for (unsigned int i = 0; i < globalDerivativesx.size(); ++i) {
    aGlobalDerivativesM(0,i) = globalDerivativesx[i];
    aGlobalDerivativesM(1,i) = globalDerivativesy[i]; 
  }
}

//__________________________________________________________________________________________________
void MillePedeAlignmentAlgorithm::diagonalize
(TMatrixDSym &aHitCovarianceM, TMatrixF &aLocalDerivativesM, TMatrixF &aHitResidualsM,
 TMatrixF &aGlobalDerivativesM) const
{
  TMatrixDSymEigen myDiag(aHitCovarianceM);
  TMatrixD aTranfoToDiagonalSystem = myDiag.GetEigenVectors();
  TMatrixD aTranfoToDiagonalSystemInv = myDiag.GetEigenVectors( );
  TMatrixF aTranfoToDiagonalSystemInvF = myDiag.GetEigenVectors( );
  TMatrixD aMatrix = aTranfoToDiagonalSystemInv.Invert() * aHitCovarianceM * aTranfoToDiagonalSystem;
  // Tranformation of matrix M is done by A^T*M*A, not A^{-1}*M*A.
  // But here A^T == A^{-1}, so we would only save CPU by Transpose()...
  // FIXME this - I guess simply use T(), not Transpose()...
  // TMatrixD aMatrix = aTranfoToDiagonalSystemInv.Transpose() * aHitCovarianceM 
  //    * aTranfoToDiagonalSystem;
  aHitCovarianceM = TMatrixDSym(2, aMatrix.GetMatrixArray());
  aTranfoToDiagonalSystemInvF.Invert();
  //edm::LogInfo("Alignment") << "NEW HIT loca in matrix"<<aLocalDerivativesM(0,0);
  aLocalDerivativesM = aTranfoToDiagonalSystemInvF * aLocalDerivativesM;
  
  //edm::LogInfo("Alignment") << "NEW HIT loca in matrix after diag:"<<aLocalDerivativesM(0,0);
  aHitResidualsM      = aTranfoToDiagonalSystemInvF * aHitResidualsM;
  aGlobalDerivativesM = aTranfoToDiagonalSystemInvF * aGlobalDerivativesM;
}

//__________________________________________________________________________________________________
void MillePedeAlignmentAlgorithm
::addRefTrackBreakpoint1D(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
                    unsigned int iBreakPoint, TMatrixDSym &aHitCovarianceM,
                    TMatrixF &aHitResidualsM, TMatrixF &aLocalDerivativesM)
{

  // This Method is valid for 1D measurements only
  
  const unsigned int xIndex = iBreakPoint + refTrajPtr->numberOfHitMeas();
  // Covariance into a TMatrixDSym
  
  //aHitCovarianceM = new TMatrixDSym(1);
  aHitCovarianceM(0,0)=refTrajPtr->measurementErrors()[xIndex][xIndex];
  
  //theHitResidualsM= new TMatrixF(1,1);
  aHitResidualsM(0,0)= refTrajPtr->measurements()[xIndex];
  
  // Local Derivatives into a TMatrixDSym (to use matrix operations)
  const AlgebraicMatrix &locDerivMatrix = refTrajPtr->derivatives();
  //  theLocalDerivativeNumber = locDerivMatrix.num_col();
  
  //theLocalDerivativesM = new TMatrixF(1,locDerivMatrix.num_col());
  for (int i = 0; i < locDerivMatrix.num_col(); ++i) {
    aLocalDerivativesM(0,i) = locDerivMatrix[xIndex][i];
  }
}

//__________________________________________________________________________________________________
void MillePedeAlignmentAlgorithm
::addRefTrackData2D(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
                    unsigned int iTrajHit, TMatrixDSym &aHitCovarianceM,
                    TMatrixF &aHitResidualsM, TMatrixF &aLocalDerivativesM)
{

  // This Method is valid for 2D measurements only
  
  const unsigned int xIndex = iTrajHit*2;
  const unsigned int yIndex = iTrajHit*2+1;
  // Covariance into a TMatrixDSym
  
  //aHitCovarianceM = new TMatrixDSym(2);
  aHitCovarianceM(0,0)=refTrajPtr->measurementErrors()[xIndex][xIndex];
  aHitCovarianceM(0,1)=refTrajPtr->measurementErrors()[xIndex][yIndex];
  aHitCovarianceM(1,0)=refTrajPtr->measurementErrors()[yIndex][xIndex];
  aHitCovarianceM(1,1)=refTrajPtr->measurementErrors()[yIndex][yIndex];
  
  //theHitResidualsM= new TMatrixF(2,1);
  aHitResidualsM(0,0)= refTrajPtr->measurements()[xIndex] - refTrajPtr->trajectoryPositions()[xIndex];
  aHitResidualsM(1,0)= refTrajPtr->measurements()[yIndex] - refTrajPtr->trajectoryPositions()[yIndex];
  
  // Local Derivatives into a TMatrixDSym (to use matrix operations)
  const AlgebraicMatrix &locDerivMatrix = refTrajPtr->derivatives();
  //  theLocalDerivativeNumber = locDerivMatrix.num_col();
  
  //theLocalDerivativesM = new TMatrixF(2,locDerivMatrix.num_col());
  for (int i = 0; i < locDerivMatrix.num_col(); ++i) {
    aLocalDerivativesM(0,i) = locDerivMatrix[xIndex][i];
    aLocalDerivativesM(1,i) = locDerivMatrix[yIndex][i];
  }
}

//__________________________________________________________________________________________________
int MillePedeAlignmentAlgorithm
::callMille2D(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
              unsigned int iTrajHit, const std::vector<int> &globalLabels,
              const std::vector<float> &globalDerivativesx,
              const std::vector<float> &globalDerivativesy)
{
  const ConstRecHitPointer aRecHit(refTrajPtr->recHits()[iTrajHit]);
  
  if((aRecHit)->dimension() != 2) {
    edm::LogError("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::callMille2D"
                               << "You try to call method for 2D hits for a " 
                               << (aRecHit)->dimension()
                               <<  "D Hit. Hit gets ignored!";
    return -1;
  }

  TMatrixDSym aHitCovarianceM(2);
  TMatrixF aHitResidualsM(2,1);
  TMatrixF aLocalDerivativesM(2, refTrajPtr->derivatives().num_col());
  // below method fills above 3 matrices
  this->addRefTrackData2D(refTrajPtr, iTrajHit, aHitCovarianceM,aHitResidualsM,aLocalDerivativesM);
  TMatrixF aGlobalDerivativesM(2,globalDerivativesx.size());
  this->makeGlobDerivMatrix(globalDerivativesx, globalDerivativesy, aGlobalDerivativesM);
 
  // calculates correlation between Hit measurements
  // FIXME: Should take correlation (and resulting transformation) from original hit, 
  //        not 2x2 matrix from ReferenceTrajectory: That can come from error propagation etc.!
  const double corr = aHitCovarianceM(0,1) / sqrt(aHitCovarianceM(0,0) * aHitCovarianceM(1,1));
  if (theMonitor) theMonitor->fillCorrelations2D(corr, aRecHit);
  bool diag = false; // diagonalise only tracker TID, TEC
  switch(aRecHit->geographicalId().subdetId()) {
  case SiStripDetId::TID: 
  case SiStripDetId::TEC:
    if (aRecHit->geographicalId().det() == DetId::Tracker && TMath::Abs(corr) > theMaximalCor2D) {
      this->diagonalize(aHitCovarianceM, aLocalDerivativesM, aHitResidualsM, aGlobalDerivativesM);
      diag = true;
    }
    break;
  default:;
  }
  
  float newResidX = aHitResidualsM(0,0);
  float newResidY = aHitResidualsM(1,0);
  float newHitErrX = TMath::Sqrt(aHitCovarianceM(0,0));
  float newHitErrY = TMath::Sqrt(aHitCovarianceM(1,1));
  float *newLocalDerivsX = aLocalDerivativesM[0].GetPtr();
  float *newLocalDerivsY = aLocalDerivativesM[1].GetPtr();
  float *newGlobDerivsX  = aGlobalDerivativesM[0].GetPtr();
  float *newGlobDerivsY  = aGlobalDerivativesM[1].GetPtr();
  const int nLocal  = aLocalDerivativesM.GetNcols();
  const int nGlobal = aGlobalDerivativesM.GetNcols();

  if (diag && (newHitErrX > newHitErrY)) { // also for 2D hits?
    // measurement with smaller error is x-measurement (for !is2D do not fill y-measurement):
    std::swap(newResidX, newResidY);
    std::swap(newHitErrX, newHitErrY);
    std::swap(newLocalDerivsX, newLocalDerivsY);
    std::swap(newGlobDerivsX, newGlobDerivsY);
  }
  
  // &(globalLabels[0]) is valid - as long as vector is not empty 
  // cf. http://www.parashift.com/c++-faq-lite/containers.html#faq-34.3
  theMille->mille(nLocal, newLocalDerivsX, nGlobal, newGlobDerivsX,
		  &(globalLabels[0]), newResidX, newHitErrX);

//   // write special (hit) data to the milleBinary.dat
//   int nPar=3;
//   std::vector<int> integers(nPar); // filled with 0.
//   std::vector<float> floats(nPar);
//   floats[0] = aRecHit->globalPosition().x();
//   floats[1] = aRecHit->globalPosition().y();
//   floats[2] = aRecHit->globalPosition().z(); 
//   theMille->special(nPar, &(floats[0]), &(integers[0]));
    
  if (theMonitor) {
    theMonitor->fillDerivatives(aRecHit, newLocalDerivsX, nLocal, newGlobDerivsX, nGlobal);
    theMonitor->fillResiduals(aRecHit, refTrajPtr->trajectoryStates()[iTrajHit],
			      iTrajHit, newResidX, newHitErrX, false);
  }
  const bool isReal2DHit = this->is2D(aRecHit); // strip is 1D (except matched hits)
  if (isReal2DHit) {
    theMille->mille(nLocal, newLocalDerivsY, nGlobal, newGlobDerivsY,
                    &(globalLabels[0]), newResidY, newHitErrY);
    if (theMonitor) {
      theMonitor->fillDerivatives(aRecHit, newLocalDerivsY, nLocal, newGlobDerivsY, nGlobal);
      theMonitor->fillResiduals(aRecHit, refTrajPtr->trajectoryStates()[iTrajHit],
				iTrajHit, newResidY, newHitErrY, true);// true: y
    }
  }

  return (isReal2DHit ? 2 : 1);
}

//____________________________________________________
void MillePedeAlignmentAlgorithm::addLaserData(const TkFittedLasBeamCollection &lasBeams,
					       const TsosVectorCollection &lasBeamTsoses)
{
  TsosVectorCollection::const_iterator iTsoses = lasBeamTsoses.begin();
  for(TkFittedLasBeamCollection::const_iterator iBeam = lasBeams.begin(), iEnd = lasBeams.end();
      iBeam != iEnd; ++iBeam, ++iTsoses){ // beam/tsoses parallel!

    edm::LogInfo("Alignment") << "@SUB=MillePedeAlignmentAlgorithm::addLaserData"
			      << "Beam " << iBeam->getBeamId() << " with " 
			      << iBeam->parameters().size() << " parameters and " 
			      << iBeam->getData().size() << " hits.\n There are " 
			      << iTsoses->size() << " TSOSes.";
    this->addLasBeam(*iBeam, *iTsoses);

  }
}

//____________________________________________________
void MillePedeAlignmentAlgorithm::addLasBeam(const TkFittedLasBeam &lasBeam,
					     const std::vector<TrajectoryStateOnSurface> &tsoses)
{
  AlignmentParameters *dummyPtr = 0; // for globalDerivativesHierarchy()
  std::vector<float> lasLocalDerivsX; // buffer for local derivatives
  const unsigned int beamLabel = thePedeLabels->lasBeamLabel(lasBeam.getBeamId());// for global par
  
  for (unsigned int iHit = 0; iHit < tsoses.size(); ++iHit) {
    if (!tsoses[iHit].isValid()) continue;
    // clear buffer
    theFloatBufferX.clear();
    theFloatBufferY.clear();
    theIntBuffer.clear();
    lasLocalDerivsX.clear();
    // get alignables and global parameters
    const SiStripLaserRecHit2D &hit = lasBeam.getData()[iHit];
    AlignableDetOrUnitPtr lasAli(theAlignableNavigator->alignableFromDetId(hit.getDetId()));
    this->globalDerivativesHierarchy(tsoses[iHit], lasAli, lasAli, 
				     theFloatBufferX, theFloatBufferY, theIntBuffer, dummyPtr);
    // fill derivatives vector from derivatives matrix
    for (unsigned int nFitParams = 0; nFitParams < lasBeam.parameters().size(); ++nFitParams) {
      const float derivative = lasBeam.derivatives()[iHit][nFitParams];
      if (nFitParams < lasBeam.firstFixedParameter()) { // first local beam parameters
	lasLocalDerivsX.push_back(derivative);
      } else {                                          // now global ones
	const unsigned int numPar = nFitParams - lasBeam.firstFixedParameter();
	theIntBuffer.push_back(thePedeLabels->parameterLabel(beamLabel, numPar));
	theFloatBufferX.push_back(derivative);
      }

      const float residual = hit.localPosition().x() - tsoses[iHit].localPosition().x();
      // error from file or assume 0.003
      const float error = 0.003; // hit.localPositionError().xx(); sqrt???
	
      theMille->mille(lasLocalDerivsX.size(), &(lasLocalDerivsX[0]), theFloatBufferX.size(),
		      &(theFloatBufferX[0]), &(theIntBuffer[0]), residual, error);
    } // end loop over parameters
  } // end of loop over hits
  
  theMille->end();
}

//__________________________________________________________________________________________________
void MillePedeAlignmentAlgorithm
::addBreakPoint(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr, unsigned int iBreakPoint)
{
  TMatrixDSym aHitCovarianceM(1);
  TMatrixF aHitResidualsM(1,1);
  TMatrixF aLocalDerivativesM(1, refTrajPtr->derivatives().num_col());
  // below method fills above 3 'matrices'
  this->addRefTrackBreakpoint1D(refTrajPtr, iBreakPoint, aHitCovarianceM, aHitResidualsM, aLocalDerivativesM);
  
  // no global parameters (use dummy 0)
  TMatrixF aGlobalDerivativesM(1,1);
  aGlobalDerivativesM(0,0) = 0;
      
  float newResidX = aHitResidualsM(0,0);  
  float newHitErrX = TMath::Sqrt(aHitCovarianceM(0,0));
  float *newLocalDerivsX = aLocalDerivativesM[0].GetPtr();
  float *newGlobDerivsX  = aGlobalDerivativesM[0].GetPtr();
  const int nLocal  = aLocalDerivativesM.GetNcols();
  const int nGlobal = 0;
              
  theMille->mille(nLocal, newLocalDerivsX, nGlobal, newGlobDerivsX,
		  &nGlobal, newResidX, newHitErrX);  
}
