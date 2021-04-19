#include <fstream>
#include <unordered_map>

#include "TFile.h"
#include "TTree.h"
#include "TList.h"
#include "TRandom.h"
#include "TMath.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Run.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignment/interface/SurveyResidual.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"
#include "Alignment/HIPAlignmentAlgorithm/interface/HIPUserVariables.h"
#include "Alignment/HIPAlignmentAlgorithm/interface/HIPUserVariablesIORoot.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include <DataFormats/GeometrySurface/interface/LocalError.h>
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/CommonAlignment/interface/AlignableExtras.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Alignment/HIPAlignmentAlgorithm/interface/HIPAlignmentAlgorithm.h"

// Constructor ----------------------------------------------------------------
HIPAlignmentAlgorithm::HIPAlignmentAlgorithm(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
    : AlignmentAlgorithmBase(cfg, iC),
      topoToken_(iC.esConsumes<TrackerTopology, IdealGeometryRecord, edm::Transition::EndRun>()),
      topoToken2_(iC.esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::EndRun>()),
      verbose(cfg.getParameter<bool>("verbosity")),
      theMonitorConfig(cfg),
      doTrackHitMonitoring(theMonitorConfig.fillTrackMonitoring || theMonitorConfig.fillTrackHitMonitoring),
      defaultAlignableSpecs((Alignable*)nullptr),
      surveyResiduals_(cfg.getUntrackedParameter<std::vector<std::string>>("surveyResiduals")),
      theTrackHitMonitorIORootFile(nullptr),
      theTrackMonitorTree(nullptr),
      theHitMonitorTree(nullptr),
      theAlignablesMonitorIORootFile(nullptr),
      theAlignablesMonitorTree(nullptr),
      theSurveyIORootFile(nullptr),
      theSurveyTree(nullptr) {
  // parse parameters
  outpath = cfg.getParameter<std::string>("outpath");
  outfile2 = cfg.getParameter<std::string>("outfile2");
  struefile = cfg.getParameter<std::string>("trueFile");
  smisalignedfile = cfg.getParameter<std::string>("misalignedFile");
  salignedfile = cfg.getParameter<std::string>("alignedFile");
  siterationfile = cfg.getParameter<std::string>("iterationFile");
  suvarfilecore = cfg.getParameter<std::string>("uvarFile");
  suvarfile = suvarfilecore;
  sparameterfile = cfg.getParameter<std::string>("parameterFile");
  ssurveyfile = cfg.getParameter<std::string>("surveyFile");

  outfile2 = outpath + outfile2;  //Alignablewise tree
  struefile = outpath + struefile;
  smisalignedfile = outpath + smisalignedfile;
  salignedfile = outpath + salignedfile;
  siterationfile = outpath + siterationfile;
  suvarfile = outpath + suvarfile;
  sparameterfile = outpath + sparameterfile;
  ssurveyfile = outpath + ssurveyfile;

  // parameters for APE
  theApplyAPE = cfg.getParameter<bool>("applyAPE");
  theAPEParameterSet = cfg.getParameter<std::vector<edm::ParameterSet>>("apeParam");

  themultiIOV = cfg.getParameter<bool>("multiIOV");
  theIOVrangeSet = cfg.getParameter<std::vector<unsigned>>("IOVrange");

  defaultAlignableSpecs.minNHits = cfg.getParameter<int>("minimumNumberOfHits");
  ;
  defaultAlignableSpecs.minRelParError = cfg.getParameter<double>("minRelParameterError");
  defaultAlignableSpecs.maxRelParError = cfg.getParameter<double>("maxRelParameterError");
  defaultAlignableSpecs.maxHitPull = cfg.getParameter<double>("maxAllowedHitPull");
  theApplyCutsPerComponent = cfg.getParameter<bool>("applyCutsPerComponent");
  theCutsPerComponent = cfg.getParameter<std::vector<edm::ParameterSet>>("cutsPerComponent");

  // for collector mode (parallel processing)
  isCollector = cfg.getParameter<bool>("collectorActive");
  theCollectorNJobs = cfg.getParameter<int>("collectorNJobs");
  theCollectorPath = cfg.getParameter<std::string>("collectorPath");

  if (isCollector)
    edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::HIPAlignmentAlgorithm"
                              << "Collector mode";

  trackPs = cfg.getParameter<bool>("UsePreSelection");
  theDataGroup = cfg.getParameter<int>("DataGroup");
  trackWt = cfg.getParameter<bool>("UseReweighting");
  Scale = cfg.getParameter<double>("Weight");
  uniEta = trackWt && cfg.getParameter<bool>("UniformEta");
  uniEtaFormula = cfg.getParameter<std::string>("UniformEtaFormula");
  if (uniEtaFormula.empty()) {
    edm::LogWarning("Alignment") << "@SUB=HIPAlignmentAlgorithm::HIPAlignmentAlgorithm"
                                 << "Uniform eta formula is empty! Resetting to 1.";
    uniEtaFormula = "1";
  }
  theEtaFormula = std::make_unique<TFormula>(uniEtaFormula.c_str());
  rewgtPerAli = trackWt && cfg.getParameter<bool>("ReweightPerAlignable");
  IsCollision = cfg.getParameter<bool>("isCollision");
  SetScanDet = cfg.getParameter<std::vector<double>>("setScanDet");
  col_cut = cfg.getParameter<double>("CLAngleCut");
  cos_cut = cfg.getParameter<double>("CSAngleCut");

  edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::HIPAlignmentAlgorithm"
                            << "Constructed";
}

// Call at beginning of job ---------------------------------------------------
void HIPAlignmentAlgorithm::initialize(const edm::EventSetup& setup,
                                       AlignableTracker* tracker,
                                       AlignableMuon* muon,
                                       AlignableExtras* extras,
                                       AlignmentParameterStore* store) {
  edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::initialize"
                            << "Initializing...";

  alignableObjectId_ = std::make_unique<AlignableObjectId>(AlignableObjectId::commonObjectIdProvider(tracker, muon));

  for (const auto& level : surveyResiduals_)
    theLevels.push_back(alignableObjectId_->stringToId(level));

  const edm::ValidityInterval& validity = setup.get<TrackerAlignmentRcd>().validityInterval();
  const edm::IOVSyncValue first1 = validity.first();
  unsigned int firstrun = first1.eventID().run();
  if (themultiIOV) {
    if (theIOVrangeSet.size() != 1) {
      bool findMatchIOV = false;
      for (unsigned int iovl = 0; iovl < theIOVrangeSet.size(); iovl++) {
        if (firstrun == theIOVrangeSet.at(iovl)) {
          std::string iovapp = std::to_string(firstrun);
          iovapp.append(".root");
          iovapp.insert(0, "_");
          salignedfile.replace(salignedfile.end() - 5, salignedfile.end(), iovapp);
          siterationfile.replace(siterationfile.end() - 5, siterationfile.end(), iovapp);
          //sparameterfile.replace(sparameterfile.end()-5, sparameterfile.end(),iovapp);
          if (isCollector) {
            outfile2.replace(outfile2.end() - 5, outfile2.end(), iovapp);
            ssurveyfile.replace(ssurveyfile.end() - 5, ssurveyfile.end(), iovapp);
            suvarfile.replace(suvarfile.end() - 5, suvarfile.end(), iovapp);
          }

          edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::initialize"
                                    << "Found the IOV file matching IOV first run " << firstrun;
          findMatchIOV = true;
          break;
        }
      }
      if (!findMatchIOV)
        edm::LogError("Alignment") << "@SUB=HIPAlignmentAlgorithm::initialize"
                                   << "Didn't find the IOV file matching IOV first run " << firstrun
                                   << " from the validity interval";
    } else {
      std::string iovapp = std::to_string(theIOVrangeSet.at(0));
      iovapp.append(".root");
      iovapp.insert(0, "_");
      salignedfile.replace(salignedfile.end() - 5, salignedfile.end(), iovapp);
      siterationfile.replace(siterationfile.end() - 5, siterationfile.end(), iovapp);
    }
  }

  // accessor Det->AlignableDet
  theAlignableDetAccessor = std::make_unique<AlignableNavigator>(extras, tracker, muon);
  if (extras != nullptr)
    edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::initialize"
                              << "AlignableNavigator initialized with AlignableExtras";

  // set alignmentParameterStore
  theAlignmentParameterStore = store;

  // get alignables
  theAlignables = theAlignmentParameterStore->alignables();

  // Config flags that specify different detectors
  {
    AlignmentParameterSelector selector(tracker, muon, extras);

    // APE parameters, clear if necessary
    theAPEParameters.clear();
    if (theApplyAPE) {
      for (std::vector<edm::ParameterSet>::const_iterator setiter = theAPEParameterSet.begin();
           setiter != theAPEParameterSet.end();
           ++setiter) {
        align::Alignables alignables;

        selector.clear();
        edm::ParameterSet selectorPSet = setiter->getParameter<edm::ParameterSet>("Selector");
        std::vector<std::string> alignParams = selectorPSet.getParameter<std::vector<std::string>>("alignParams");
        if (alignParams.size() == 1 && alignParams[0] == std::string("selected"))
          alignables = theAlignables;
        else {
          selector.addSelections(selectorPSet);
          alignables = selector.selectedAlignables();
        }

        std::vector<double> apeSPar = setiter->getParameter<std::vector<double>>("apeSPar");
        std::vector<double> apeRPar = setiter->getParameter<std::vector<double>>("apeRPar");
        std::string function = setiter->getParameter<std::string>("function");

        if (apeSPar.size() != 3 || apeRPar.size() != 3)
          throw cms::Exception("BadConfig") << "apeSPar and apeRPar must have 3 values each" << std::endl;

        for (std::vector<double>::const_iterator i = apeRPar.begin(); i != apeRPar.end(); ++i)
          apeSPar.push_back(*i);

        if (function == std::string("linear"))
          apeSPar.push_back(0);  // c.f. note in calcAPE
        else if (function == std::string("exponential"))
          apeSPar.push_back(1);  // c.f. note in calcAPE
        else if (function == std::string("step"))
          apeSPar.push_back(2);  // c.f. note in calcAPE
        else
          throw cms::Exception("BadConfig")
              << "APE function must be \"linear\", \"exponential\", or \"step\"." << std::endl;

        theAPEParameters.push_back(std::make_pair(alignables, apeSPar));
      }
    }

    // Relative error per component instead of overall relative error
    theAlignableSpecifics.clear();
    if (theApplyCutsPerComponent) {
      for (std::vector<edm::ParameterSet>::const_iterator setiter = theCutsPerComponent.begin();
           setiter != theCutsPerComponent.end();
           ++setiter) {
        align::Alignables alignables;

        selector.clear();
        edm::ParameterSet selectorPSet = setiter->getParameter<edm::ParameterSet>("Selector");
        std::vector<std::string> alignParams = selectorPSet.getParameter<std::vector<std::string>>("alignParams");
        if (alignParams.size() == 1 && alignParams[0] == std::string("selected"))
          alignables = theAlignables;
        else {
          selector.addSelections(selectorPSet);
          alignables = selector.selectedAlignables();
        }

        double minRelParError = setiter->getParameter<double>("minRelParError");
        double maxRelParError = setiter->getParameter<double>("maxRelParError");
        int minNHits = setiter->getParameter<int>("minNHits");
        double maxHitPull = setiter->getParameter<double>("maxHitPull");
        bool applyPixelProbCut = setiter->getParameter<bool>("applyPixelProbCut");
        bool usePixelProbXYOrProbQ = setiter->getParameter<bool>("usePixelProbXYOrProbQ");
        double minPixelProbXY = setiter->getParameter<double>("minPixelProbXY");
        double maxPixelProbXY = setiter->getParameter<double>("maxPixelProbXY");
        double minPixelProbQ = setiter->getParameter<double>("minPixelProbQ");
        double maxPixelProbQ = setiter->getParameter<double>("maxPixelProbQ");
        for (auto& ali : alignables) {
          HIPAlignableSpecificParameters alispecs(ali);
          alispecs.minRelParError = minRelParError;
          alispecs.maxRelParError = maxRelParError;
          alispecs.minNHits = minNHits;
          alispecs.maxHitPull = maxHitPull;

          alispecs.applyPixelProbCut = applyPixelProbCut;
          alispecs.usePixelProbXYOrProbQ = usePixelProbXYOrProbQ;
          alispecs.minPixelProbXY = minPixelProbXY;
          alispecs.maxPixelProbXY = maxPixelProbXY;
          alispecs.minPixelProbQ = minPixelProbQ;
          alispecs.maxPixelProbQ = maxPixelProbQ;

          theAlignableSpecifics.push_back(alispecs);
          edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::initialize"
                                    << "Alignment specifics acquired for detector " << alispecs.id() << " / "
                                    << alispecs.objId() << ":\n"
                                    << " - minRelParError = " << alispecs.minRelParError << "\n"
                                    << " - maxRelParError = " << alispecs.maxRelParError << "\n"
                                    << " - minNHits = " << alispecs.minNHits << "\n"
                                    << " - maxHitPull = " << alispecs.maxHitPull << "\n"
                                    << " - applyPixelProbCut = " << alispecs.applyPixelProbCut << "\n"
                                    << " - usePixelProbXYOrProbQ = " << alispecs.usePixelProbXYOrProbQ << "\n"
                                    << " - minPixelProbXY = " << alispecs.minPixelProbXY << "\n"
                                    << " - maxPixelProbXY = " << alispecs.maxPixelProbXY << "\n"
                                    << " - minPixelProbQ = " << alispecs.minPixelProbQ << "\n"
                                    << " - maxPixelProbQ = " << alispecs.maxPixelProbQ;
        }
      }
    }
  }
}

// Call at new loop -------------------------------------------------------------
void HIPAlignmentAlgorithm::startNewLoop(void) {
  edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::startNewLoop"
                            << "Begin";

  // iterate over all alignables and attach user variables
  for (const auto& it : theAlignables) {
    AlignmentParameters* ap = it->alignmentParameters();
    int npar = ap->numSelected();
    HIPUserVariables* userpar = new HIPUserVariables(npar);
    ap->setUserVariables(userpar);
  }

  // try to read in alignment parameters from a previous iteration
  AlignablePositions theAlignablePositionsFromFile =
      theIO.readAlignableAbsolutePositions(theAlignables, salignedfile.c_str(), -1, ioerr);
  int numAlignablesFromFile = theAlignablePositionsFromFile.size();
  if (numAlignablesFromFile == 0) {  // file not there: first iteration
    // set iteration number to 1 when needed
    if (isCollector)
      theIteration = 0;
    else
      theIteration = 1;
    edm::LogWarning("Alignment") << "@SUB=HIPAlignmentAlgorithm::startNewLoop"
                                 << "IO alignables file not found for iteration " << theIteration;
  } else {  // there have been previous iterations
    edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::startNewLoop"
                              << "Alignables Read " << numAlignablesFromFile;

    // get iteration number from file
    theIteration = readIterationFile(siterationfile);
    // Where is the target for this?
    theIO.readAlignableAbsolutePositions(theAlignables, salignedfile.c_str(), theIteration, ioerr);

    // increase iteration
    if (ioerr == 0) {
      theIteration++;
      edm::LogWarning("Alignment") << "@SUB=HIPAlignmentAlgorithm::startNewLoop"
                                   << "Iteration increased by one and is now " << theIteration;
    }

    // now apply psotions of file from prev iteration
    edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::startNewLoop"
                              << "Apply positions from file ...";
    theAlignmentParameterStore->applyAlignableAbsolutePositions(theAlignables, theAlignablePositionsFromFile, ioerr);
  }

  edm::LogWarning("Alignment") << "@SUB=HIPAlignmentAlgorithm::startNewLoop"
                               << "Current Iteration number: " << theIteration;

  // book root trees
  bookRoot();

  // set alignment position error
  setAlignmentPositionError();

  // run collector job if we are in parallel mode
  if (isCollector)
    collector();

  edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::startNewLoop"
                            << "End";
}

// Call at end of job ---------------------------------------------------------
void HIPAlignmentAlgorithm::terminate(const edm::EventSetup& iSetup) {
  edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm] Terminating";

  // calculating survey residuals
  if (!theLevels.empty()) {
    edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm] Using survey constraint";

    unsigned int nAlignable = theAlignables.size();
    const TrackerTopology* const tTopo = &iSetup.getData(topoToken_);
    for (unsigned int i = 0; i < nAlignable; ++i) {
      const Alignable* ali = theAlignables[i];
      AlignmentParameters* ap = ali->alignmentParameters();
      HIPUserVariables* uservar = dynamic_cast<HIPUserVariables*>(ap->userVariables());
      int nhit = uservar->nhit;

      // get position
      std::pair<int, int> tl = theAlignmentParameterStore->typeAndLayer(ali, tTopo);
      int tmp_Type = tl.first;
      int tmp_Layer = tl.second;
      GlobalPoint pos = ali->surface().position();
      float tmpz = pos.z();
      if (nhit < 1500 ||
          (tmp_Type == 5 && tmp_Layer == 4 && fabs(tmpz) > 90)) {  // FIXME: Needs revision for hardcoded consts
        for (unsigned int l = 0; l < theLevels.size(); ++l) {
          SurveyResidual res(*ali, theLevels[l], true);

          if (res.valid()) {
            AlgebraicSymMatrix invCov = res.inverseCovariance();

            // variable for tree
            AlgebraicVector sensResid = res.sensorResidual();
            m3_Id = ali->id();
            m3_ObjId = theLevels[l];
            m3_par[0] = sensResid[0];
            m3_par[1] = sensResid[1];
            m3_par[2] = sensResid[2];
            m3_par[3] = sensResid[3];
            m3_par[4] = sensResid[4];
            m3_par[5] = sensResid[5];

            uservar->jtvj += invCov;
            uservar->jtve += invCov * sensResid;

            if (theSurveyTree != nullptr)
              theSurveyTree->Fill();
          }
        }
      }
    }
  }

  // write user variables
  HIPUserVariablesIORoot HIPIO;
  // don't store userVariable in main, to save time
  if (!isCollector)
    HIPIO.writeHIPUserVariables(theAlignables, suvarfile.c_str(), theIteration, false, ioerr);

  // now calculate alignment corrections...
  int ialigned = 0;
  // iterate over alignment parameters
  for (const auto& ali : theAlignables) {
    AlignmentParameters* par = ali->alignmentParameters();

    if (SetScanDet.at(0) != 0) {
      edm::LogWarning("Alignment") << "********Starting Scan*********";
      edm::LogWarning("Alignment") << "det ID=" << SetScanDet.at(0) << ", starting position=" << SetScanDet.at(1)
                                   << ", step=" << SetScanDet.at(2) << ", currentDet = " << ali->id();
    }

    if ((SetScanDet.at(0) != 0) && (SetScanDet.at(0) != 1) && (ali->id() != SetScanDet.at(0)))
      continue;

    bool test = calcParameters(ali, SetScanDet.at(0), SetScanDet.at(1), SetScanDet.at(2));
    if (test) {
      if (dynamic_cast<AlignableDetUnit*>(ali) != nullptr) {
        std::vector<std::pair<int, SurfaceDeformation*>> pairs;
        ali->surfaceDeformationIdPairs(pairs);
        edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::terminate"
                                  << "The alignable contains " << pairs.size() << " surface deformations";
      } else
        edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::terminate"
                                  << "The alignable cannot contain surface deformations";

      theAlignmentParameterStore->applyParameters(ali);
      // set these parameters 'valid'
      ali->alignmentParameters()->setValid(true);
      // increase counter
      ialigned++;
    } else
      par->setValid(false);
  }
  //end looping over alignables

  edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm::terminate] Aligned units: " << ialigned;

  // fill alignable wise root tree
  fillAlignablesMonitor(iSetup);

  edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm] Writing aligned parameters to file: " << theAlignables.size()
                               << ", for Iteration " << theIteration;

  // write user variables
  if (isCollector)
    HIPIO.writeHIPUserVariables(theAlignables, suvarfile.c_str(), theIteration, false, ioerr);

  // write new absolute positions to disk
  theIO.writeAlignableAbsolutePositions(theAlignables, salignedfile.c_str(), theIteration, false, ioerr);

  // write alignment parameters to disk
  //theIO.writeAlignmentParameters(theAlignables,
  //				 sparameterfile.c_str(),theIteration,false,ioerr);

  // write iteration number to file
  writeIterationFile(siterationfile, theIteration);

  // write out trees and close root file
  // Survey tree
  if (theSurveyIORootFile != nullptr) {
    theSurveyIORootFile->cd();
    if (theSurveyTree != nullptr)
      theSurveyTree->Write();
    delete theSurveyTree;
    theSurveyTree = nullptr;
    theSurveyIORootFile->Close();
  }
  // Alignable-wise tree is only filled once at iteration 1
  if (theAlignablesMonitorIORootFile != nullptr) {
    theAlignablesMonitorIORootFile->cd();
    if (theAlignablesMonitorTree != nullptr)
      theAlignablesMonitorTree->Write();
    delete theAlignablesMonitorTree;
    theAlignablesMonitorTree = nullptr;
    theAlignablesMonitorIORootFile->Close();
  }
  // Eventwise and hitwise monitoring trees
  if (theTrackHitMonitorIORootFile != nullptr) {
    theTrackHitMonitorIORootFile->cd();
    if (theTrackMonitorTree != nullptr) {
      theTrackMonitorTree->Write();
      delete theTrackMonitorTree;
      theTrackMonitorTree = nullptr;
    }
    if (theHitMonitorTree != nullptr) {
      theHitMonitorTree->Write();
      delete theHitMonitorTree;
      theHitMonitorTree = nullptr;
    }
    theTrackHitMonitorIORootFile->Close();
  }
}

bool HIPAlignmentAlgorithm::processHit1D(const AlignableDetOrUnitPtr& alidet,
                                         const Alignable* ali,
                                         const HIPAlignableSpecificParameters* alispecifics,
                                         const TrajectoryStateOnSurface& tsos,
                                         const TransientTrackingRecHit* hit,
                                         double hitwt) {
  static const unsigned int hitDim = 1;
  if (hitwt == 0.)
    return false;

  // get trajectory impact point
  LocalPoint alvec = tsos.localPosition();
  AlgebraicVector pos(hitDim);
  pos[0] = alvec.x();

  // get impact point covariance
  AlgebraicSymMatrix ipcovmat(hitDim);
  ipcovmat[0][0] = tsos.localError().positionError().xx();

  // get hit local position and covariance
  AlgebraicVector coor(hitDim);
  coor[0] = hit->localPosition().x();

  AlgebraicSymMatrix covmat(hitDim);
  covmat[0][0] = hit->localPositionError().xx();

  // add hit and impact point covariance matrices
  covmat = covmat + ipcovmat;

  // calculate the x pull of this hit
  double xpull = 0.;
  if (covmat[0][0] != 0.)
    xpull = (pos[0] - coor[0]) / sqrt(fabs(covmat[0][0]));

  // get Alignment Parameters
  AlignmentParameters* params = ali->alignmentParameters();
  HIPUserVariables* uservar = dynamic_cast<HIPUserVariables*>(params->userVariables());
  uservar->datatype = theDataGroup;
  // get derivatives
  AlgebraicMatrix derivs2D = params->selectedDerivatives(tsos, alidet);
  // calculate user parameters
  int npar = derivs2D.num_row();
  AlgebraicMatrix derivs(npar, hitDim, 0);  // This is jT

  for (int ipar = 0; ipar < npar; ipar++) {
    for (unsigned int idim = 0; idim < hitDim; idim++) {
      derivs[ipar][idim] = derivs2D[ipar][idim];
    }
  }

  // invert covariance matrix
  int ierr;
  covmat.invert(ierr);
  if (ierr != 0) {
    edm::LogError("Alignment") << "@SUB=HIPAlignmentAlgorithm::processHit1D"
                               << "Cov. matrix inversion failed!";
    return false;
  }

  double maxHitPullThreshold =
      (!theApplyCutsPerComponent ? defaultAlignableSpecs.maxHitPull : alispecifics->maxHitPull);
  bool useThisHit = (maxHitPullThreshold < 0.);
  useThisHit |= (fabs(xpull) < maxHitPullThreshold);
  if (!useThisHit) {
    edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::processHit2D"
                              << "Hit pull (x) " << xpull << " fails the cut " << maxHitPullThreshold;
    return false;
  }

  AlgebraicMatrix covtmp(covmat);
  AlgebraicMatrix jtvjtmp(derivs * covtmp * derivs.T());
  AlgebraicSymMatrix thisjtvj(npar);
  AlgebraicVector thisjtve(npar);
  thisjtvj.assign(jtvjtmp);
  thisjtve = derivs * covmat * (pos - coor);

  AlgebraicVector hitresidual(hitDim);
  hitresidual[0] = (pos[0] - coor[0]);

  AlgebraicMatrix hitresidualT;
  hitresidualT = hitresidual.T();

  uservar->jtvj += hitwt * thisjtvj;
  uservar->jtve += hitwt * thisjtve;
  uservar->nhit++;

  //for alignable chi squared
  float thischi2;
  thischi2 = hitwt * (hitresidualT * covmat * hitresidual)[0];

  if (verbose && (thischi2 / static_cast<float>(uservar->nhit)) > 10.)
    edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm::processHit1D] Added to Chi2 the number " << thischi2
                                 << " having " << uservar->nhit << " ndof"
                                 << ", X-resid " << hitresidual[0] << ", Cov^-1 matr (covmat):"
                                 << " [0][0] = " << covmat[0][0];

  uservar->alichi2 += thischi2;
  uservar->alindof += hitDim;

  return true;
}

bool HIPAlignmentAlgorithm::processHit2D(const AlignableDetOrUnitPtr& alidet,
                                         const Alignable* ali,
                                         const HIPAlignableSpecificParameters* alispecifics,
                                         const TrajectoryStateOnSurface& tsos,
                                         const TransientTrackingRecHit* hit,
                                         double hitwt) {
  static const unsigned int hitDim = 2;
  if (hitwt == 0.)
    return false;

  // get trajectory impact point
  LocalPoint alvec = tsos.localPosition();
  AlgebraicVector pos(hitDim);
  pos[0] = alvec.x();
  pos[1] = alvec.y();

  // get impact point covariance
  AlgebraicSymMatrix ipcovmat(hitDim);
  ipcovmat[0][0] = tsos.localError().positionError().xx();
  ipcovmat[1][1] = tsos.localError().positionError().yy();
  ipcovmat[0][1] = tsos.localError().positionError().xy();

  // get hit local position and covariance
  AlgebraicVector coor(hitDim);
  coor[0] = hit->localPosition().x();
  coor[1] = hit->localPosition().y();

  AlgebraicSymMatrix covmat(hitDim);
  covmat[0][0] = hit->localPositionError().xx();
  covmat[1][1] = hit->localPositionError().yy();
  covmat[0][1] = hit->localPositionError().xy();

  // add hit and impact point covariance matrices
  covmat = covmat + ipcovmat;

  // calculate the x pull and y pull of this hit
  double xpull = 0.;
  double ypull = 0.;
  if (covmat[0][0] != 0.)
    xpull = (pos[0] - coor[0]) / sqrt(fabs(covmat[0][0]));
  if (covmat[1][1] != 0.)
    ypull = (pos[1] - coor[1]) / sqrt(fabs(covmat[1][1]));

  // get Alignment Parameters
  AlignmentParameters* params = ali->alignmentParameters();
  HIPUserVariables* uservar = dynamic_cast<HIPUserVariables*>(params->userVariables());
  uservar->datatype = theDataGroup;
  // get derivatives
  AlgebraicMatrix derivs2D = params->selectedDerivatives(tsos, alidet);
  // calculate user parameters
  int npar = derivs2D.num_row();
  AlgebraicMatrix derivs(npar, hitDim, 0);  // This is jT

  for (int ipar = 0; ipar < npar; ipar++) {
    for (unsigned int idim = 0; idim < hitDim; idim++) {
      derivs[ipar][idim] = derivs2D[ipar][idim];
    }
  }

  // invert covariance matrix
  int ierr;
  covmat.invert(ierr);
  if (ierr != 0) {
    edm::LogError("Alignment") << "@SUB=HIPAlignmentAlgorithm::processHit2D"
                               << "Cov. matrix inversion failed!";
    return false;
  }

  double maxHitPullThreshold =
      (!theApplyCutsPerComponent ? defaultAlignableSpecs.maxHitPull : alispecifics->maxHitPull);
  bool useThisHit = (maxHitPullThreshold < 0.);
  useThisHit |= (fabs(xpull) < maxHitPullThreshold && fabs(ypull) < maxHitPullThreshold);
  if (!useThisHit) {
    edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::processHit2D"
                              << "Hit pull (x,y) " << xpull << " , " << ypull << " fails the cut "
                              << maxHitPullThreshold;
    return false;
  }

  AlgebraicMatrix covtmp(covmat);
  AlgebraicMatrix jtvjtmp(derivs * covtmp * derivs.T());
  AlgebraicSymMatrix thisjtvj(npar);
  AlgebraicVector thisjtve(npar);
  thisjtvj.assign(jtvjtmp);
  thisjtve = derivs * covmat * (pos - coor);

  AlgebraicVector hitresidual(hitDim);
  hitresidual[0] = (pos[0] - coor[0]);
  hitresidual[1] = (pos[1] - coor[1]);

  AlgebraicMatrix hitresidualT;
  hitresidualT = hitresidual.T();

  uservar->jtvj += hitwt * thisjtvj;
  uservar->jtve += hitwt * thisjtve;
  uservar->nhit++;

  //for alignable chi squared
  float thischi2;
  thischi2 = hitwt * (hitresidualT * covmat * hitresidual)[0];

  if (verbose && (thischi2 / static_cast<float>(uservar->nhit)) > 10.)
    edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm::processHit2D] Added to Chi2 the number " << thischi2
                                 << " having " << uservar->nhit << " ndof"
                                 << ", X-resid " << hitresidual[0] << ", Y-resid " << hitresidual[1]
                                 << ", Cov^-1 matr (covmat):"
                                 << " [0][0] = " << covmat[0][0] << " [0][1] = " << covmat[0][1]
                                 << " [1][0] = " << covmat[1][0] << " [1][1] = " << covmat[1][1];

  uservar->alichi2 += thischi2;
  uservar->alindof += hitDim;

  return true;
}

// Run the algorithm on trajectories and tracks -------------------------------
void HIPAlignmentAlgorithm::run(const edm::EventSetup& setup, const EventInfo& eventInfo) {
  if (isCollector)
    return;

  TrajectoryStateCombiner tsoscomb;

  m_datatype = theDataGroup;

  // loop over tracks
  const ConstTrajTrackPairCollection& tracks = eventInfo.trajTrackPairs();
  for (ConstTrajTrackPairCollection::const_iterator it = tracks.begin(); it != tracks.end(); ++it) {
    const Trajectory* traj = (*it).first;
    const reco::Track* track = (*it).second;

    float pt = track->pt();
    float eta = track->eta();
    float phi = track->phi();
    float p = track->p();
    float chi2n = track->normalizedChi2();
    int nhit = track->numberOfValidHits();
    float d0 = track->d0();
    float dz = track->dz();

    int nhpxb = track->hitPattern().numberOfValidPixelBarrelHits();
    int nhpxf = track->hitPattern().numberOfValidPixelEndcapHits();
    int nhtib = track->hitPattern().numberOfValidStripTIBHits();
    int nhtob = track->hitPattern().numberOfValidStripTOBHits();
    int nhtid = track->hitPattern().numberOfValidStripTIDHits();
    int nhtec = track->hitPattern().numberOfValidStripTECHits();

    if (verbose)
      edm::LogInfo("Alignment") << "New track pt,eta,phi,chi2n,hits: " << pt << "," << eta << "," << phi << "," << chi2n
                                << "," << nhit;

    double ihitwt = 1;
    double trkwt = 1;
    if (trackWt) {
      trkwt = Scale;
      // Reweight by the specified eta distribution
      if (uniEta)
        trkwt *= theEtaFormula->Eval(fabs(eta));
    }
    if (trackPs) {
      double r = gRandom->Rndm();
      if (trkwt < r)
        continue;
    } else if (trackWt)
      ihitwt = trkwt;

    // fill track parameters in root tree
    {
      theMonitorConfig.trackmonitorvars.m_Nhits.push_back(nhit);
      theMonitorConfig.trackmonitorvars.m_Pt.push_back(pt);
      theMonitorConfig.trackmonitorvars.m_P.push_back(p);
      theMonitorConfig.trackmonitorvars.m_Eta.push_back(eta);
      theMonitorConfig.trackmonitorvars.m_Phi.push_back(phi);
      theMonitorConfig.trackmonitorvars.m_Chi2n.push_back(chi2n);
      theMonitorConfig.trackmonitorvars.m_nhPXB.push_back(nhpxb);
      theMonitorConfig.trackmonitorvars.m_nhPXF.push_back(nhpxf);
      theMonitorConfig.trackmonitorvars.m_nhTIB.push_back(nhtib);
      theMonitorConfig.trackmonitorvars.m_nhTOB.push_back(nhtob);
      theMonitorConfig.trackmonitorvars.m_nhTID.push_back(nhtid);
      theMonitorConfig.trackmonitorvars.m_nhTEC.push_back(nhtec);
      theMonitorConfig.trackmonitorvars.m_d0.push_back(d0);
      theMonitorConfig.trackmonitorvars.m_dz.push_back(dz);
      theMonitorConfig.trackmonitorvars.m_wt.push_back(ihitwt);
    }

    std::vector<const TransientTrackingRecHit*> hitvec;
    std::vector<TrajectoryStateOnSurface> tsosvec;

    // loop over measurements
    std::vector<TrajectoryMeasurement> measurements = traj->measurements();
    for (std::vector<TrajectoryMeasurement>::iterator im = measurements.begin(); im != measurements.end(); ++im) {
      TrajectoryMeasurement meas = *im;

      // const TransientTrackingRecHit* ttrhit = &(*meas.recHit());
      // const TrackingRecHit *hit = ttrhit->hit();
      const TransientTrackingRecHit* hit = &(*meas.recHit());

      if (hit->isValid() && theAlignableDetAccessor->detAndSubdetInMap(hit->geographicalId())) {
        // this is the updated state (including the current hit)
        // TrajectoryStateOnSurface tsos=meas.updatedState();
        // combine fwd and bwd predicted state to get state
        // which excludes current hit

        //////////Hit prescaling part
        if (eventInfo.clusterValueMap()) {
          // check from the PrescalingMap if the hit was taken.
          // If not skip to the next TM
          // bool hitTaken=false;
          AlignmentClusterFlag myflag;

          int subDet = hit->geographicalId().subdetId();
          //take the actual RecHit out of the Transient one
          const TrackingRecHit* rechit = hit->hit();
          if (subDet > 2) {  // AM: if possible use enum instead of hard-coded value
            const std::type_info& type = typeid(*rechit);

            if (type == typeid(SiStripRecHit1D)) {
              const SiStripRecHit1D* stripHit1D = dynamic_cast<const SiStripRecHit1D*>(rechit);
              if (stripHit1D) {
                SiStripRecHit1D::ClusterRef stripclust(stripHit1D->cluster());
                // myflag=PrescMap[stripclust];
                myflag = (*eventInfo.clusterValueMap())[stripclust];
              } else
                edm::LogError("HIPAlignmentAlgorithm")
                    << "ERROR in <HIPAlignmentAlgorithm::run>: Dynamic cast of Strip RecHit failed! "
                    << "TypeId of the RecHit: " << className(*hit);
            }  //end if type = SiStripRecHit1D
            else if (type == typeid(SiStripRecHit2D)) {
              const SiStripRecHit2D* stripHit2D = dynamic_cast<const SiStripRecHit2D*>(rechit);
              if (stripHit2D) {
                SiStripRecHit2D::ClusterRef stripclust(stripHit2D->cluster());
                // myflag=PrescMap[stripclust];
                myflag = (*eventInfo.clusterValueMap())[stripclust];
              } else
                edm::LogError("HIPAlignmentAlgorithm")
                    << "ERROR in <HIPAlignmentAlgorithm::run>: Dynamic cast of Strip RecHit failed! "
                    << "TypeId of the TTRH: " << className(*hit);
            }  //end if type == SiStripRecHit2D
          }    //end if hit from strips
          else {
            const SiPixelRecHit* pixelhit = dynamic_cast<const SiPixelRecHit*>(rechit);
            if (pixelhit) {
              SiPixelClusterRefNew pixelclust(pixelhit->cluster());
              // myflag=PrescMap[pixelclust];
              myflag = (*eventInfo.clusterValueMap())[pixelclust];
            } else
              edm::LogError("HIPAlignmentAlgorithm")
                  << "ERROR in <HIPAlignmentAlgorithm::run>: Dynamic cast of Pixel RecHit failed! "
                  << "TypeId of the TTRH: " << className(*hit);
          }  //end 'else' it is a pixel hit
          if (!myflag.isTaken())
            continue;
        }  //end if Prescaled Hits
        ////////////////////////////////

        TrajectoryStateOnSurface tsos = tsoscomb.combine(meas.forwardPredictedState(), meas.backwardPredictedState());

        if (tsos.isValid()) {
          hitvec.push_back(hit);
          tsosvec.push_back(tsos);
        }

      }  //hit valid
    }

    // transform RecHit vector to AlignableDet vector
    std::vector<AlignableDetOrUnitPtr> alidetvec = theAlignableDetAccessor->alignablesFromHits(hitvec);

    // get concatenated alignment parameters for list of alignables
    CompositeAlignmentParameters aap = theAlignmentParameterStore->selectParameters(alidetvec);

    std::vector<TrajectoryStateOnSurface>::const_iterator itsos = tsosvec.begin();
    std::vector<const TransientTrackingRecHit*>::const_iterator ihit = hitvec.begin();

    // loop over vectors(hit,tsos)
    while (itsos != tsosvec.end()) {
      // get AlignableDet for this hit
      const GeomDet* det = (*ihit)->det();
      // int subDet= (*ihit)->geographicalId().subdetId();
      uint32_t nhitDim = (*ihit)->dimension();

      AlignableDetOrUnitPtr alidet = theAlignableDetAccessor->alignableFromGeomDet(det);

      // get relevant Alignable
      Alignable* ali = aap.alignableFromAlignableDet(alidet);

      if (ali != nullptr) {
        const HIPAlignableSpecificParameters* alispecifics = findAlignableSpecs(ali);
        const TrajectoryStateOnSurface& tsos = *itsos;

        //  LocalVector v = tsos.localDirection();
        //  double proj_z = v.dot(LocalVector(0,0,1));

        //In fact, sin_theta=Abs(mom_z)
        double mom_x = tsos.localDirection().x();
        double mom_y = tsos.localDirection().y();
        double mom_z = tsos.localDirection().z();
        double sin_theta = TMath::Abs(mom_z) / sqrt(pow(mom_x, 2) + pow(mom_y, 2) + pow(mom_z, 2));
        double angle = TMath::ASin(sin_theta);
        double alihitwt = ihitwt;

        //Make cut on hit impact angle, reduce collision hits perpendicular to modules
        if (IsCollision) {
          if (angle > col_cut)
            alihitwt = 0;
        } else {
          if (angle < cos_cut)
            alihitwt = 0;
        }

        // Fill hit monitor variables
        theMonitorConfig.hitmonitorvars.m_angle = angle;
        theMonitorConfig.hitmonitorvars.m_sinTheta = sin_theta;
        theMonitorConfig.hitmonitorvars.m_detId = ali->id();

        // Check pixel XY and Q probabilities
        if ((*ihit)->hit() != nullptr) {
          const SiPixelRecHit* pixhit = dynamic_cast<const SiPixelRecHit*>((*ihit)->hit());
          if (pixhit != nullptr) {
            theMonitorConfig.hitmonitorvars.m_hasHitProb = pixhit->hasFilledProb();
            if (theMonitorConfig.hitmonitorvars.m_hasHitProb) {
              // Prob X, Y are deprecated
              theMonitorConfig.hitmonitorvars.m_probXY = pixhit->probabilityXY();
              theMonitorConfig.hitmonitorvars.m_probQ = pixhit->probabilityQ();
              theMonitorConfig.hitmonitorvars.m_rawQualityWord = pixhit->rawQualityWord();
              if (alispecifics->applyPixelProbCut) {
                bool probXYgood = (theMonitorConfig.hitmonitorvars.m_probXY >= alispecifics->minPixelProbXY &&
                                   theMonitorConfig.hitmonitorvars.m_probXY <= alispecifics->maxPixelProbXY);
                bool probQgood = (theMonitorConfig.hitmonitorvars.m_probQ >= alispecifics->minPixelProbQ &&
                                  theMonitorConfig.hitmonitorvars.m_probQ <= alispecifics->maxPixelProbQ);
                bool probXYQgood;
                if (alispecifics->usePixelProbXYOrProbQ)
                  probXYQgood = (probXYgood || probQgood);
                else
                  probXYQgood = (probXYgood && probQgood);
                if (!probXYQgood)
                  alihitwt = 0;
              }
            }
          }
        }

        theMonitorConfig.hitmonitorvars.m_hitwt = alihitwt;
        bool hitProcessed = false;
        switch (nhitDim) {
          case 1:
            hitProcessed = processHit1D(alidet, ali, alispecifics, tsos, *ihit, alihitwt);
            break;
          case 2:
            hitProcessed = processHit2D(alidet, ali, alispecifics, tsos, *ihit, alihitwt);
            break;
          default:
            edm::LogError("HIPAlignmentAlgorithm")
                << "ERROR in <HIPAlignmentAlgorithm::run>: Number of hit dimensions = " << nhitDim
                << " is not supported!" << std::endl;
            break;
        }
        if (theMonitorConfig.fillTrackHitMonitoring && theMonitorConfig.checkNhits() && hitProcessed)
          theMonitorConfig.hitmonitorvars.fill();
      }

      itsos++;
      ihit++;
    }
  }  // end of track loop

  // fill eventwise root tree (with prescale defined in pset)
  if (theMonitorConfig.fillTrackMonitoring && theMonitorConfig.checkNevents())
    theMonitorConfig.trackmonitorvars.fill();
}

// ----------------------------------------------------------------------------
int HIPAlignmentAlgorithm::readIterationFile(std::string filename) {
  int result;

  std::ifstream inIterFile(filename.c_str(), std::ios::in);
  if (!inIterFile) {
    edm::LogError("Alignment") << "[HIPAlignmentAlgorithm::readIterationFile] ERROR! "
                               << "Unable to open Iteration file";
    result = -1;
  } else {
    inIterFile >> result;
    edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm::readIterationFile] "
                                 << "Read last iteration number from file: " << result;
  }
  inIterFile.close();

  return result;
}

// ----------------------------------------------------------------------------
void HIPAlignmentAlgorithm::writeIterationFile(std::string filename, int iter) {
  std::ofstream outIterFile((filename.c_str()), std::ios::out);
  if (!outIterFile)
    edm::LogError("Alignment") << "[HIPAlignmentAlgorithm::writeIterationFile] ERROR: Unable to write Iteration file";
  else {
    outIterFile << iter;
    edm::LogWarning("Alignment") << "[HIPAlignmentAlgorithm::writeIterationFile] writing iteration number to file: "
                                 << iter;
  }
  outIterFile.close();
}

// ----------------------------------------------------------------------------
// set alignment position error
void HIPAlignmentAlgorithm::setAlignmentPositionError(void) {
  // Check if user wants to override APE
  if (!theApplyAPE) {
    edm::LogInfo("Alignment") << "[HIPAlignmentAlgorithm::setAlignmentPositionError] No APE applied";
    return;  // NO APE APPLIED
  }

  edm::LogInfo("Alignment") << "[HIPAlignmentAlgorithm::setAlignmentPositionError] Apply APE!";

  double apeSPar[3], apeRPar[3];
  for (const auto& alipars : theAPEParameters) {
    const auto& alignables = alipars.first;
    const auto& pars = alipars.second;

    apeSPar[0] = pars[0];
    apeSPar[1] = pars[1];
    apeSPar[2] = pars[2];
    apeRPar[0] = pars[3];
    apeRPar[1] = pars[4];
    apeRPar[2] = pars[5];

    int function = pars[6];

    // Printout for debug
    printf("APE: %u alignables\n", (unsigned int)alignables.size());
    for (int i = 0; i < 21; ++i) {
      double apelinstest = calcAPE(apeSPar, i, 0);
      double apeexpstest = calcAPE(apeSPar, i, 1);
      double apestepstest = calcAPE(apeSPar, i, 2);
      double apelinrtest = calcAPE(apeRPar, i, 0);
      double apeexprtest = calcAPE(apeRPar, i, 1);
      double apesteprtest = calcAPE(apeRPar, i, 2);
      printf("APE: iter slin sexp sstep rlin rexp rstep: %5d %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f \n",
             i,
             apelinstest,
             apeexpstest,
             apestepstest,
             apelinrtest,
             apeexprtest,
             apesteprtest);
    }

    // set APE
    double apeshift = calcAPE(apeSPar, theIteration, function);
    double aperot = calcAPE(apeRPar, theIteration, function);
    theAlignmentParameterStore->setAlignmentPositionError(alignables, apeshift, aperot);
  }
}

// ----------------------------------------------------------------------------
// calculate APE
double HIPAlignmentAlgorithm::calcAPE(double* par, int iter, int function) {
  double diter = (double)iter;
  if (function == 0)
    return std::max(par[1], par[0] + ((par[1] - par[0]) / par[2]) * diter);
  else if (function == 1)
    return std::max(0., par[0] * (exp(-pow(diter, par[1]) / par[2])));
  else if (function == 2) {
    int ipar2 = (int)par[2];
    int step = iter / ipar2;
    double dstep = (double)step;
    return std::max(0., par[0] - par[1] * dstep);
  } else
    assert(false);  // should have been caught in the constructor
}

// ----------------------------------------------------------------------------
// book root trees
void HIPAlignmentAlgorithm::bookRoot(void) {
  // create ROOT files
  if (doTrackHitMonitoring && !isCollector) {
    theTrackHitMonitorIORootFile = TFile::Open(theMonitorConfig.outfile.c_str(), "update");
    theTrackHitMonitorIORootFile->cd();
    // book event-wise ROOT Tree
    if (theMonitorConfig.fillTrackMonitoring) {
      TString tname = Form("T1_%i", theIteration);
      theTrackMonitorTree = new TTree(tname, "Eventwise tree");
      //theTrackMonitorTree->Branch("Run",     &m_Run,     "Run/I");
      //theTrackMonitorTree->Branch("Event",   &m_Event,   "Event/I");
      theTrackMonitorTree->Branch("DataType", &m_datatype);
      theMonitorConfig.trackmonitorvars.setTree(theTrackMonitorTree);
      theMonitorConfig.trackmonitorvars.bookBranches();
    }
    // book hit-wise ROOT Tree
    if (theMonitorConfig.fillTrackHitMonitoring) {
      TString tname_hit = Form("T1_hit_%i", theIteration);
      theHitMonitorTree = new TTree(tname_hit, "Hitwise tree");
      theHitMonitorTree->Branch("DataType", &m_datatype);
      theMonitorConfig.hitmonitorvars.setTree(theHitMonitorTree);
      theMonitorConfig.hitmonitorvars.bookBranches();
    }
  }

  // book alignable-wise ROOT Tree
  if (isCollector) {
    TString tname = Form("T2_%i", theIteration);
    theAlignablesMonitorIORootFile = TFile::Open(outfile2.c_str(), "update");
    theAlignablesMonitorIORootFile->cd();
    theAlignablesMonitorTree = new TTree(tname, "Alignablewise tree");
    theAlignablesMonitorTree->Branch("Id", &m2_Id, "Id/i");
    theAlignablesMonitorTree->Branch("ObjId", &m2_ObjId, "ObjId/I");
    theAlignablesMonitorTree->Branch("Nhit", &m2_Nhit);
    theAlignablesMonitorTree->Branch("DataType", &m2_datatype);
    theAlignablesMonitorTree->Branch("Type", &m2_Type);
    theAlignablesMonitorTree->Branch("Layer", &m2_Layer);
    theAlignablesMonitorTree->Branch("Xpos", &m2_Xpos);
    theAlignablesMonitorTree->Branch("Ypos", &m2_Ypos);
    theAlignablesMonitorTree->Branch("Zpos", &m2_Zpos);
    theAlignablesMonitorTree->Branch("DeformationsType", &m2_dtype, "DeformationsType/I");
    theAlignablesMonitorTree->Branch("NumDeformations", &m2_nsurfdef);
    theAlignablesMonitorTree->Branch("Deformations", &m2_surfDef);
  }

  // book survey-wise ROOT Tree only if survey is enabled
  if (!theLevels.empty()) {
    TString tname = Form("T3_%i", theIteration);
    theSurveyIORootFile = TFile::Open(ssurveyfile.c_str(), "update");
    theSurveyIORootFile->cd();
    theSurveyTree = new TTree(tname, "Survey Tree");
    theSurveyTree->Branch("Id", &m3_Id, "Id/i");
    theSurveyTree->Branch("ObjId", &m3_ObjId, "ObjId/I");
    theSurveyTree->Branch("Par", &m3_par, "Par[6]/F");
    edm::LogInfo("Alignment") << "[HIPAlignmentAlgorithm::bookRoot] Survey trees booked.";
  }
  edm::LogInfo("Alignment") << "[HIPAlignmentAlgorithm::bookRoot] Root trees booked.";
}

// ----------------------------------------------------------------------------
// fill alignable-wise root tree
void HIPAlignmentAlgorithm::fillAlignablesMonitor(const edm::EventSetup& iSetup) {
  if (theAlignablesMonitorIORootFile == (TFile*)nullptr)
    return;
  using std::setw;
  theAlignablesMonitorIORootFile->cd();

  int naligned = 0;

  //Retrieve tracker topology from geometry
  const TrackerTopology* const tTopo = &iSetup.getData(topoToken2_);

  for (const auto& ali : theAlignables) {
    AlignmentParameters* dap = ali->alignmentParameters();

    // consider only those parameters classified as 'valid'
    if (dap->isValid()) {
      // get number of hits from user variable
      HIPUserVariables* uservar = dynamic_cast<HIPUserVariables*>(dap->userVariables());
      m2_Nhit = uservar->nhit;
      m2_datatype = uservar->datatype;

      // get type/layer
      std::pair<int, int> tl = theAlignmentParameterStore->typeAndLayer(ali, tTopo);
      m2_Type = tl.first;
      m2_Layer = tl.second;

      // get identifier (as for IO)
      m2_Id = ali->id();
      m2_ObjId = ali->alignableObjectId();

      // get position
      GlobalPoint pos = ali->surface().position();
      m2_Xpos = pos.x();
      m2_Ypos = pos.y();
      m2_Zpos = pos.z();

      m2_surfDef.clear();
      {
        std::vector<std::pair<int, SurfaceDeformation*>> dali_id_pairs;
        SurfaceDeformation* dali_obj = nullptr;
        SurfaceDeformationFactory::Type dtype = SurfaceDeformationFactory::kNoDeformations;
        std::vector<double> dali;
        if (1 == ali->surfaceDeformationIdPairs(dali_id_pairs)) {
          dali_obj = dali_id_pairs[0].second;
          dali = dali_obj->parameters();
          dtype = (SurfaceDeformationFactory::Type)dali_obj->type();
        }
        for (auto& dit : dali)
          m2_surfDef.push_back((float)dit);
        m2_dtype = dtype;
        m2_nsurfdef = m2_surfDef.size();
      }

      if (verbose) {
        AlgebraicVector pars = dap->parameters();
        edm::LogVerbatim("Alignment") << "------------------------------------------------------------------------\n"
                                      << " ALIGNABLE: " << setw(6) << naligned << '\n'
                                      << "hits: " << setw(4) << m2_Nhit << " type: " << setw(4) << m2_Type
                                      << " layer: " << setw(4) << m2_Layer << " id: " << setw(4) << m2_Id
                                      << " objId: " << setw(4) << m2_ObjId << '\n'
                                      << std::fixed << std::setprecision(5) << "x,y,z: " << setw(12) << m2_Xpos
                                      << setw(12) << m2_Ypos << setw(12) << m2_Zpos
                                      << "\nDeformations type, nDeformations: " << setw(12) << m2_dtype << setw(12)
                                      << m2_nsurfdef << '\n'
                                      << "params: " << setw(12) << pars[0] << setw(12) << pars[1] << setw(12) << pars[2]
                                      << setw(12) << pars[3] << setw(12) << pars[4] << setw(12) << pars[5];
      }

      naligned++;
      if (theAlignablesMonitorTree != nullptr)
        theAlignablesMonitorTree->Fill();
    }
  }
}

// ----------------------------------------------------------------------------
bool HIPAlignmentAlgorithm::calcParameters(Alignable* ali, int setDet, double start, double step) {
  edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::calcParameters"
                            << "Begin: Processing detector " << ali->id();

  // Alignment parameters
  AlignmentParameters* par = ali->alignmentParameters();
  const HIPAlignableSpecificParameters* alispecifics = findAlignableSpecs(ali);
  // access user variables
  HIPUserVariables* uservar = dynamic_cast<HIPUserVariables*>(par->userVariables());
  int nhit = uservar->nhit;
  // The following variable is needed for the extended 1D/2D hit fix using
  // matrix shrinkage and expansion
  // int hitdim = uservar->hitdim;

  // Test nhits
  int minHitThreshold = (!theApplyCutsPerComponent ? defaultAlignableSpecs.minNHits : alispecifics->minNHits);
  if (!isCollector)
    minHitThreshold = 1;
  if (setDet == 0 && nhit < minHitThreshold) {
    edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::calcParameters"
                              << "Skipping detector " << ali->id() << " because number of hits = " << nhit
                              << " <= " << minHitThreshold;
    par->setValid(false);
    return false;
  }

  AlgebraicSymMatrix jtvj = uservar->jtvj;
  AlgebraicVector jtve = uservar->jtve;

  // these are the alignment corrections+covariance (for selected params)
  int npar = jtve.num_row();
  AlgebraicVector params(npar);
  AlgebraicVector paramerr(npar);
  AlgebraicSymMatrix cov(npar);

  // errors of parameters
  if (isCollector) {
    if (setDet == 0) {
      int ierr;
      AlgebraicSymMatrix jtvjinv = jtvj.inverse(ierr);
      if (ierr != 0) {
        edm::LogError("Alignment") << "@SUB=HIPAlignmentAlgorithm::calcParameters"
                                   << "Matrix inversion failed!";
        return false;
      }
      params = -(jtvjinv * jtve);
      cov = jtvjinv;

      double minRelErrThreshold =
          (!theApplyCutsPerComponent ? defaultAlignableSpecs.minRelParError : alispecifics->minRelParError);
      double maxRelErrThreshold =
          (!theApplyCutsPerComponent ? defaultAlignableSpecs.maxRelParError : alispecifics->maxRelParError);
      for (int i = 0; i < npar; i++) {
        double relerr = 0;
        if (fabs(cov[i][i]) > 0.)
          paramerr[i] = sqrt(fabs(cov[i][i]));
        else
          paramerr[i] = params[i];
        if (params[i] != 0.)
          relerr = fabs(paramerr[i] / params[i]);
        if ((maxRelErrThreshold >= 0. && relerr > maxRelErrThreshold) || relerr < minRelErrThreshold) {
          edm::LogWarning("Alignment") << "@SUB=HIPAlignmentAlgorithm::calcParameters"
                                       << "RelError = " << relerr << " > " << maxRelErrThreshold << " or < "
                                       << minRelErrThreshold << ". Setting param = paramerr = 0 for component " << i;
          params[i] = 0;
          paramerr[i] = 0;
        }
      }
    } else {
      if (params.num_row() != 1) {
        edm::LogError("Alignment") << "@SUB=HIPAlignmentAlgorithm::calcParameters"
                                   << "For scanning, please only turn on one parameter! check common_cff_py.txt";
        return false;
      }
      if (theIteration == 1)
        params[0] = start;
      else
        params[0] = step;
      edm::LogWarning("Alignment") << "@SUB=HIPAlignmentAlgorithm::calcParameters"
                                   << "Parameters = " << params;
    }
  }

  uservar->alipar = params;
  uservar->alierr = paramerr;

  AlignmentParameters* parnew = par->cloneFromSelected(params, cov);
  ali->setAlignmentParameters(parnew);
  parnew->setValid(true);

  edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::calcParameters"
                            << "End: Processing detector " << ali->id();

  return true;
}

//-----------------------------------------------------------------------------
void HIPAlignmentAlgorithm::collector(void) {
  edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::collector"
                            << "Called for iteration " << theIteration;

  std::vector<std::string> monitorFileList;
  HIPUserVariablesIORoot HIPIO;

  typedef int pawt_t;
  std::unordered_map<Alignable*, std::unordered_map<int, pawt_t>> ali_datatypecountpair_map;
  if (rewgtPerAli) {
    edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::collector"
                              << "Per-alignable reweighting is turned on. Iterating over the parallel jobs to sum "
                                 "number of hits per alignable for each data type.";
    // Counting step for per-alignable reweighting
    for (int ijob = 1; ijob <= theCollectorNJobs; ijob++) {
      edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::collector"
                                << "Pre-reading uservar for job " << ijob;

      std::string str = std::to_string(ijob);
      std::string uvfile = theCollectorPath + "/job" + str + "/" + suvarfilecore;

      std::vector<AlignmentUserVariables*> uvarvec =
          HIPIO.readHIPUserVariables(theAlignables, uvfile.c_str(), theIteration, ioerr);
      if (uvarvec.size() != theAlignables.size())
        edm::LogWarning("Alignment") << "@SUB=HIPAlignmentAlgorithm::collector"
                                     << "Number of alignables = " << theAlignables.size()
                                     << " is not the same as number of user variables = " << uvarvec.size()
                                     << ". A mismatch might occur!";
      if (ioerr != 0) {
        edm::LogError("Alignment") << "@SUB=HIPAlignmentAlgorithm::collector"
                                   << "Could not read user variable files for job " << ijob << " in iteration "
                                   << theIteration;
        continue;
      }
      std::vector<AlignmentUserVariables*>::const_iterator iuvar =
          uvarvec.begin();  // This vector should have 1-to-1 correspondence with the alignables vector
      for (const auto& ali : theAlignables) {
        // No need for the user variables already attached to the alignables
        // Just count from what you read.
        HIPUserVariables* uvar = dynamic_cast<HIPUserVariables*>(*iuvar);
        if (uvar != nullptr) {
          int alijobdtype = uvar->datatype;
          pawt_t alijobnhits = uvar->nhit;
          if (ali_datatypecountpair_map.find(ali) == ali_datatypecountpair_map.end()) {  // This is a new alignable
            std::unordered_map<int, pawt_t> newmap;
            ali_datatypecountpair_map[ali] = newmap;
            ali_datatypecountpair_map[ali][alijobdtype] = alijobnhits;
          } else {  // Alignable already exists in the map
            std::unordered_map<int, pawt_t>& theMap = ali_datatypecountpair_map[ali];
            if (theMap.find(alijobdtype) == theMap.end())
              theMap[alijobdtype] = alijobnhits;
            else
              theMap[alijobdtype] += alijobnhits;
          }
          delete uvar;  // Delete new user variables as they are added
        }
        iuvar++;
      }  // End loop over alignables
    }    // End loop over subjobs
  }

  for (int ijob = 1; ijob <= theCollectorNJobs; ijob++) {
    edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::collector"
                              << "Reading uservar for job " << ijob;

    std::string str = std::to_string(ijob);
    std::string uvfile = theCollectorPath + "/job" + str + "/" + suvarfilecore;

    std::vector<AlignmentUserVariables*> uvarvec =
        HIPIO.readHIPUserVariables(theAlignables, uvfile.c_str(), theIteration, ioerr);
    if (uvarvec.size() != theAlignables.size())
      edm::LogWarning("Alignment") << "@SUB=HIPAlignmentAlgorithm::collector"
                                   << "Number of alignables = " << theAlignables.size()
                                   << " is not the same as number of user variables = " << uvarvec.size()
                                   << ". A mismatch might occur!";

    if (ioerr != 0) {
      edm::LogError("Alignment") << "@SUB=HIPAlignmentAlgorithm::collector"
                                 << "Could not read user variable files for job " << ijob << " in iteration "
                                 << theIteration;
      continue;
    }

    // add
    std::vector<AlignmentUserVariables*> uvarvecadd;
    std::vector<AlignmentUserVariables*>::const_iterator iuvarnew = uvarvec.begin();
    for (const auto& ali : theAlignables) {
      AlignmentParameters* ap = ali->alignmentParameters();

      HIPUserVariables* uvarold = dynamic_cast<HIPUserVariables*>(ap->userVariables());
      HIPUserVariables* uvarnew = dynamic_cast<HIPUserVariables*>(*iuvarnew);

      HIPUserVariables* uvar = uvarold->clone();
      uvar->datatype =
          theDataGroup;  // Set the data type of alignable to that specified for the collector job (-2 by default)

      if (uvarnew != nullptr) {
        double peraliwgt = 1;
        if (rewgtPerAli) {
          int alijobdtype = uvarnew->datatype;
          if (ali_datatypecountpair_map.find(ali) != ali_datatypecountpair_map.end() &&
              ali_datatypecountpair_map[ali].find(alijobdtype) != ali_datatypecountpair_map[ali].end()) {
            peraliwgt = ali_datatypecountpair_map[ali][alijobdtype];
            unsigned int nNonZeroTypes = 0;
            pawt_t sumwgts = 0;
            for (auto it = ali_datatypecountpair_map[ali].cbegin(); it != ali_datatypecountpair_map[ali].cend(); ++it) {
              sumwgts += it->second;
              if (it->second != pawt_t(0))
                nNonZeroTypes++;
            }
            edm::LogInfo("Alignment") << "@SUB=HIPAlignmentAlgorithm::collector"
                                      << "Reweighting detector " << ali->id() << " / " << ali->alignableObjectId()
                                      << " for data type " << alijobdtype << " by " << sumwgts << "/" << peraliwgt
                                      << "/" << nNonZeroTypes;
            peraliwgt = ((nNonZeroTypes == 0 || peraliwgt == double(0))
                             ? double(1)
                             : double((double(sumwgts)) / peraliwgt / (double(nNonZeroTypes))));
          } else if (ali_datatypecountpair_map.find(ali) != ali_datatypecountpair_map.end())
            edm::LogError("Alignment") << "@SUB=HIPAlignmentAlgorithm::collector"
                                       << "Could not find data type " << alijobdtype << " for detector " << ali->id()
                                       << " / " << ali->alignableObjectId();
          else
            edm::LogError("Alignment") << "@SUB=HIPAlignmentAlgorithm::collector"
                                       << "Could not find detector " << ali->id() << " / " << ali->alignableObjectId()
                                       << " in the map ali_datatypecountpair_map";
        }

        uvar->nhit = (uvarold->nhit) + (uvarnew->nhit);
        uvar->jtvj = (uvarold->jtvj) + peraliwgt * (uvarnew->jtvj);
        uvar->jtve = (uvarold->jtve) + peraliwgt * (uvarnew->jtve);
        uvar->alichi2 = (uvarold->alichi2) + peraliwgt * (uvarnew->alichi2);
        uvar->alindof = (uvarold->alindof) + (uvarnew->alindof);

        delete uvarnew;  // Delete new user variables as they are added
      }

      uvarvecadd.push_back(uvar);
      iuvarnew++;
    }

    theAlignmentParameterStore->attachUserVariables(theAlignables, uvarvecadd, ioerr);

    // fill Eventwise Tree
    if (doTrackHitMonitoring) {
      uvfile = theCollectorPath + "/job" + str + "/" + theMonitorConfig.outfilecore;
      monitorFileList.push_back(uvfile);
    }
  }  // end loop on jobs

  // Collect monitor (eventwise and hitwise) trees
  if (doTrackHitMonitoring)
    collectMonitorTrees(monitorFileList);
}

//------------------------------------------------------------------------------------
void HIPAlignmentAlgorithm::collectMonitorTrees(const std::vector<std::string>& filenames) {
  if (!doTrackHitMonitoring)
    return;
  if (!isCollector)
    throw cms::Exception("LogicError") << "[HIPAlignmentAlgorithm::collectMonitorTrees] Called in non-collector mode."
                                       << std::endl;

  TString theTrackMonitorTreeName = Form("T1_%i", theIteration);
  TString theHitMonitorTreeName = Form("T1_hit_%i", theIteration);

  std::vector<TFile*> finputlist;
  TList* eventtrees = new TList;
  TList* hittrees = new TList;
  for (std::string const& filename : filenames) {
    TFile* finput = TFile::Open(filename.c_str(), "read");
    if (finput != nullptr) {
      TTree* tmptree;
      if (theMonitorConfig.fillTrackMonitoring) {
        tmptree = nullptr;
        tmptree = (TTree*)finput->Get(theTrackMonitorTreeName);
        if (tmptree != nullptr)
          eventtrees->Add(tmptree);
      }
      if (theMonitorConfig.fillTrackHitMonitoring) {
        tmptree = nullptr;
        tmptree = (TTree*)finput->Get(theHitMonitorTreeName);
        if (tmptree != nullptr)
          hittrees->Add((TTree*)finput->Get(theHitMonitorTreeName));
      }
      finputlist.push_back(finput);
    }
  }

  if (theTrackHitMonitorIORootFile != nullptr) {  // This should never happen
    edm::LogError("Alignment") << "@SUB=HIPAlignmentAlgorithm::collectMonitorTrees"
                               << "Monitor file is already open while it is not supposed to be!";
    delete theTrackMonitorTree;
    theTrackMonitorTree = nullptr;
    delete theHitMonitorTree;
    theHitMonitorTree = nullptr;
    theTrackHitMonitorIORootFile->Close();
  }
  theTrackHitMonitorIORootFile = TFile::Open(theMonitorConfig.outfile.c_str(), "update");
  theTrackHitMonitorIORootFile->cd();
  if (eventtrees->GetSize() > 0)
    theTrackMonitorTree = TTree::MergeTrees(eventtrees);
  if (hittrees->GetSize() > 0)
    theHitMonitorTree = TTree::MergeTrees(hittrees);
  // Leave it to HIPAlignmentAlgorithm::terminate to write the trees and close theTrackHitMonitorIORootFile

  delete hittrees;
  delete eventtrees;
  for (TFile*& finput : finputlist)
    finput->Close();

  // Rename the trees to standard names
  if (theTrackMonitorTree != nullptr)
    theTrackMonitorTree->SetName(theTrackMonitorTreeName);
  if (theHitMonitorTree != nullptr)
    theHitMonitorTree->SetName(theHitMonitorTreeName);
}

//-----------------------------------------------------------------------------------
HIPAlignableSpecificParameters* HIPAlignmentAlgorithm::findAlignableSpecs(const Alignable* ali) {
  if (ali != nullptr) {
    for (std::vector<HIPAlignableSpecificParameters>::iterator it = theAlignableSpecifics.begin();
         it != theAlignableSpecifics.end();
         it++) {
      if (it->matchAlignable(ali))
        return &(*it);
    }
    edm::LogInfo("Alignment") << "[HIPAlignmentAlgorithm::findAlignableSpecs] Alignment object with id " << ali->id()
                              << " / " << ali->alignableObjectId() << " could not be found. Returning default.";
  }
  return &defaultAlignableSpecs;
}
