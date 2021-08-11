#include "L1Trigger/TrackFindingTMTT/plugins/TMTrackProducer.h"
#include "L1Trigger/TrackFindingTMTT/interface/InputData.h"
#include "L1Trigger/TrackFindingTMTT/interface/Sector.h"
#include "L1Trigger/TrackFindingTMTT/interface/HTrphi.h"
#include "L1Trigger/TrackFindingTMTT/interface/Make3Dtracks.h"
#include "L1Trigger/TrackFindingTMTT/interface/DupFitTrkKiller.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackFitFactory.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/ConverterToTTTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/HTcell.h"
#include "L1Trigger/TrackFindingTMTT/interface/MuxHToutputs.h"
#include "L1Trigger/TrackFindingTMTT/interface/MiniHTstage.h"
#include "L1Trigger/TrackFindingTMTT/interface/Array2D.h"
#include "L1Trigger/TrackFindingTMTT/interface/PrintL1trk.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include <iostream>
#include <vector>
#include <list>
#include <set>
#include <sstream>
#include <mutex>

using namespace std;

namespace tmtt {

  namespace {
    std::once_flag printOnce;
    std::once_flag callOnce;
  }  // namespace

  std::unique_ptr<GlobalCacheTMTT> TMTrackProducer::initializeGlobalCache(edm::ParameterSet const& iConfig) {
    return std::make_unique<GlobalCacheTMTT>(iConfig);
  }

  TMTrackProducer::TMTrackProducer(const edm::ParameterSet& iConfig, GlobalCacheTMTT const* globalCacheTMTT)
      : settings_(iConfig),                                        // Set configuration parameters
        stubWindowSuggest_(globalCacheTMTT->stubWindowSuggest()),  // For tuning FE stub window sizes
        hists_(globalCacheTMTT->hists()),                          // Initialize histograms
        htRphiErrMon_(globalCacheTMTT->htRphiErrMon()),            // rphi HT error monitoring
        debug_(true)                                               // Debug printout
  {
    using namespace edm;

    // Get tokens for ES data access.
    magneticFieldToken_ =
        esConsumes<MagneticField, IdealMagneticFieldRecord, Transition::BeginRun>(settings_.magneticFieldInputTag());
    trackerGeometryToken_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, Transition::BeginRun>(
        settings_.trackerGeometryInputTag());
    trackerTopologyToken_ =
        esConsumes<TrackerTopology, TrackerTopologyRcd, Transition::BeginRun>(settings_.trackerTopologyInputTag());
    ttStubAlgoToken_ =
        esConsumes<StubAlgorithm, TTStubAlgorithmRecord, Transition::BeginRun>(settings_.ttStubAlgoInputTag());

    // Get tokens for ED data access.
    stubToken_ = consumes<TTStubDetSetVec>(settings_.stubInputTag());
    if (settings_.enableMCtruth()) {
      // These lines use lots of CPU, even if no use of truth info is made later.
      tpToken_ = consumes<TrackingParticleCollection>(settings_.tpInputTag());
      stubTruthToken_ = consumes<TTStubAssMap>(settings_.stubTruthInputTag());
      clusterTruthToken_ = consumes<TTClusterAssMap>(settings_.clusterTruthInputTag());
      genJetToken_ = consumes<reco::GenJetCollection>(settings_.genJetInputTag());
    }

    trackFitters_ = settings_.trackFitters();
    useRZfilter_ = settings_.useRZfilter();
    runRZfilter_ = (not useRZfilter_.empty());  // Do any fitters require an r-z track filter to be run?

    // Book histograms.
    //hists_.book();

    // Create track fitting algorithm
    for (const string& fitterName : trackFitters_) {
      fitterWorkerMap_[fitterName] = trackFitFactory::create(fitterName, &settings_);
    }

    //--- Define EDM output to be written to file (if required)

    if (settings_.enableOutputIntermediateTTTracks()) {
      // L1 tracks found by Hough Transform
      produces<TTTrackCollection>("TML1TracksHT").setBranchAlias("TML1TracksHT");
      // L1 tracks found by r-z track filter.
      if (runRZfilter_)
        produces<TTTrackCollection>("TML1TracksRZ").setBranchAlias("TML1TracksRZ");
    }
    // L1 tracks after track fit by each of the fitting algorithms under study
    for (const string& fitterName : trackFitters_) {
      string edmName = string("TML1Tracks") + fitterName;
      produces<TTTrackCollection>(edmName).setBranchAlias(edmName);
    }
  }

  //=== Run every run

  void TMTrackProducer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
    // Get the B-field and store its value in the Settings class.
    const MagneticField* theMagneticField = &(iSetup.getData(magneticFieldToken_));
    float bField = theMagneticField->inTesla(GlobalPoint(0, 0, 0)).z();  // B field in Tesla.
    settings_.setMagneticField(bField);

    // Set also B field in GlobalCacheTMTT (used only for Histogramming)
    globalCache()->settings().setMagneticField(bField);

    std::stringstream text;
    text << "\n--- B field = " << bField << " Tesla ---\n";
    std::call_once(
        printOnce, [](string t) { PrintL1trk() << t; }, text.str());

    // Get tracker geometry
    trackerGeometry_ = &(iSetup.getData(trackerGeometryToken_));
    trackerTopology_ = &(iSetup.getData(trackerTopologyToken_));

    // Loop over tracker modules to get module info.

    // Identifies tracker module type for firmware.
    TrackerModule::ModuleTypeCfg moduleTypeCfg;
    moduleTypeCfg.pitchVsType = settings_.pitchVsType();
    moduleTypeCfg.spaceVsType = settings_.spaceVsType();
    moduleTypeCfg.barrelVsType = settings_.barrelVsType();
    moduleTypeCfg.psVsType = settings_.psVsType();
    moduleTypeCfg.tiltedVsType = settings_.tiltedVsType();

    listTrackerModule_.clear();
    for (const GeomDet* gd : trackerGeometry_->dets()) {
      DetId detId = gd->geographicalId();
      // Phase 2 Outer Tracker uses TOB for entire barrel & TID for entire endcap.
      if (detId.subdetId() != StripSubdetector::TOB && detId.subdetId() != StripSubdetector::TID)
        continue;
      if (trackerTopology_->isLower(detId)) {  // Select only lower of the two sensors in a module.
        // Store info about this tracker module.
        listTrackerModule_.emplace_back(trackerGeometry_, trackerTopology_, moduleTypeCfg, detId);
      }
    }

    // Takes one copy of this to GlobalCacheTMTT for later histogramming.
    globalCache()->setListTrackerModule(listTrackerModule_);

    // Get TTStubProducerAlgorithm algorithm, to adjust stub bend FE encoding.
    stubAlgo_ = dynamic_cast<const StubAlgorithmOfficial*>(&iSetup.getData(ttStubAlgoToken_));
    // Get FE stub window size from TTStub producer configuration
    const edm::ESHandle<StubAlgorithm> stubAlgoHandle = iSetup.getHandle(ttStubAlgoToken_);
    const edm::ParameterSet& pSetStubAlgo = getParameterSet(stubAlgoHandle.description()->pid_);
    stubFEWindows_ = std::make_unique<StubFEWindows>(pSetStubAlgo);
    // Initialize utilities needing FE window size.
    stubWindowSuggest_.setFEWindows(stubFEWindows_.get());
    degradeBend_ = std::make_unique<DegradeBend>(trackerTopology_, stubFEWindows_.get(), stubAlgo_);
  }

  //=== Run every event

  void TMTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // Note useful info about MC truth particles and about reconstructed stubs .
    InputData inputData(iEvent,
                        iSetup,
                        &settings_,
                        &stubWindowSuggest_,
                        degradeBend_.get(),
                        trackerGeometry_,
                        trackerTopology_,
                        listTrackerModule_,
                        tpToken_,
                        stubToken_,
                        stubTruthToken_,
                        clusterTruthToken_,
                        genJetToken_);

    const list<TP>& vTPs = inputData.getTPs();
    const list<Stub*>& vStubs = inputData.stubs();

    // Creates matrix of Sector objects, which decide which stubs are in which (eta,phi) sector
    Array2D<unique_ptr<Sector>> mSectors(settings_.numPhiSectors(), settings_.numEtaRegions());
    // Create matrix of r-phi Hough-Transform arrays, with one-to-one correspondence to sectors.
    Array2D<unique_ptr<HTrphi>> mHtRphis(settings_.numPhiSectors(), settings_.numEtaRegions());
    // Create matrix of Make3Dtracks objects, to run optional r-z track filter, with one-to-one correspondence to sectors.
    Array2D<unique_ptr<Make3Dtracks>> mMake3Dtrks(settings_.numPhiSectors(), settings_.numEtaRegions());
    // Create matrix of tracks from each fitter in each sector
    Array2D<map<string, std::list<L1fittedTrack>>> mapmFitTrks(settings_.numPhiSectors(), settings_.numEtaRegions());
    // Final tracks after duplicate removal from each track fitter in entire tracker.
    map<string, list<const L1fittedTrack*>> mapFinalTracks;

    //=== Initialization
    // Create utility for converting L1 tracks from our private format to official CMSSW EDM format.
    const ConverterToTTTrack converter(&settings_);

    // Pointers to TTTrack collections for ED output.
    auto htTTTracksForOutput = std::make_unique<TTTrackCollection>();
    auto rzTTTracksForOutput = std::make_unique<TTTrackCollection>();
    map<string, unique_ptr<TTTrackCollection>> allFitTTTracksForOutput;
    for (const string& fitterName : trackFitters_) {
      auto fitTTTracksForOutput = std::make_unique<TTTrackCollection>();
      allFitTTTracksForOutput[fitterName] = std::move(fitTTTracksForOutput);
    }

    //=== Do tracking in the r-phi Hough transform within each sector.

    // Fill Hough-Transform arrays with stubs.
    for (unsigned int iPhiSec = 0; iPhiSec < settings_.numPhiSectors(); iPhiSec++) {
      for (unsigned int iEtaReg = 0; iEtaReg < settings_.numEtaRegions(); iEtaReg++) {
        // Initialize constants for this sector.
        mSectors(iPhiSec, iEtaReg) = std::make_unique<Sector>(&settings_, iPhiSec, iEtaReg);
        Sector* sector = mSectors(iPhiSec, iEtaReg).get();

        mHtRphis(iPhiSec, iEtaReg) = std::make_unique<HTrphi>(
            &settings_, iPhiSec, iEtaReg, sector->etaMin(), sector->etaMax(), sector->phiCentre(), &htRphiErrMon_);
        HTrphi* htRphi = mHtRphis(iPhiSec, iEtaReg).get();

        // Check sector is enabled (always true, except if user disabled some for special studies).
        if (settings_.isHTRPhiEtaRegWhitelisted(iEtaReg)) {
          for (Stub* stub : vStubs) {
            // Digitize stub as would be at input to GP. This doesn't need the nonant number, since we assumed an integer number of
            // phi digitisation  bins inside an nonant. N.B. This changes the coordinates & bend stored in the stub.

            if (settings_.enableDigitize())
              stub->digitize(iPhiSec, Stub::DigiStage::GP);

            // Check if stub is inside this sector
            bool inside = sector->inside(stub);

            if (inside) {
              // Check which eta subsectors within the sector the stub is compatible with (if subsectors being used).
              const vector<bool> inEtaSubSecs = sector->insideEtaSubSecs(stub);

              // Digitize stub if as would be at input to HT, which slightly degrades its coord. & bend resolution, affecting the HT performance.
              if (settings_.enableDigitize())
                stub->digitize(iPhiSec, Stub::DigiStage::HT);

              // Store stub in Hough transform array for this sector, indicating its compatibility with eta subsectors with sector.
              htRphi->store(stub, inEtaSubSecs);
            }
          }
        }

        // Find tracks in r-phi HT array.
        htRphi->end();  // Calls htArrayRphi_.end() -> HTBase::end()
      }
    }

    if (settings_.muxOutputsHT() > 0) {
      // Multiplex outputs of several HT onto one pair of output opto-links.
      // This only affects tracking performance if option busySectorKill is enabled, so that tracks that
      // can't be sent down the link within the time-multiplexed period are killed.
      MuxHToutputs muxHT(&settings_);
      muxHT.exec(mHtRphis);
    }

    // Optionally, run 2nd stage mini HT -- WITHOUT TRUNCATION ???
    if (settings_.miniHTstage()) {
      MiniHTstage miniHTstage(&settings_);
      miniHTstage.exec(mHtRphis);
    }

    //=== Make 3D tracks, optionally running r-z track filters (such as Seed Filter) & duplicate track removal.

    for (unsigned int iPhiSec = 0; iPhiSec < settings_.numPhiSectors(); iPhiSec++) {
      for (unsigned int iEtaReg = 0; iEtaReg < settings_.numEtaRegions(); iEtaReg++) {
        const Sector* sector = mSectors(iPhiSec, iEtaReg).get();

        // Get tracks found by r-phi HT.
        const HTrphi* htRphi = mHtRphis(iPhiSec, iEtaReg).get();
        const list<L1track2D>& vecTracksRphi = htRphi->trackCands2D();

        // Initialize utility for making 3D tracks from 2D ones.
        mMake3Dtrks(iPhiSec, iEtaReg) = std::make_unique<Make3Dtracks>(
            &settings_, iPhiSec, iEtaReg, sector->etaMin(), sector->etaMax(), sector->phiCentre());
        Make3Dtracks* make3Dtrk = mMake3Dtrks(iPhiSec, iEtaReg).get();

        // Convert 2D tracks found by HT to 3D tracks (optionally by running r-z filters & duplicate track removal)
        make3Dtrk->run(vecTracksRphi);

        if (settings_.enableOutputIntermediateTTTracks()) {
          // Convert these tracks to EDM format for output (used for collaborative work outside TMTT group).
          // Do this for tracks output by HT & optionally also for those output by r-z track filter.
          const list<L1track3D>& vecTrk3D_ht = make3Dtrk->trackCands3D(false);
          for (const L1track3D& trk : vecTrk3D_ht) {
            TTTrack<Ref_Phase2TrackerDigi_> htTTTrack = converter.makeTTTrack(&trk, iPhiSec, iEtaReg);
            htTTTracksForOutput->push_back(htTTTrack);
          }

          if (runRZfilter_) {
            const list<L1track3D>& vecTrk3D_rz = make3Dtrk->trackCands3D(true);
            for (const L1track3D& trk : vecTrk3D_rz) {
              TTTrack<Ref_Phase2TrackerDigi_> rzTTTrack = converter.makeTTTrack(&trk, iPhiSec, iEtaReg);
              rzTTTracksForOutput->push_back(rzTTTrack);
            }
          }
        }
      }
    }

    //=== Do a helix fit to all the track candidates.

    // Loop over all the fitting algorithms we are trying.
    for (const string& fitterName : trackFitters_) {
      for (unsigned int iPhiSec = 0; iPhiSec < settings_.numPhiSectors(); iPhiSec++) {
        for (unsigned int iEtaReg = 0; iEtaReg < settings_.numEtaRegions(); iEtaReg++) {
          const Make3Dtracks* make3Dtrk = mMake3Dtrks(iPhiSec, iEtaReg).get();

          // Does this fitter require r-z track filter to be run before it?
          bool useRZfilt = (std::count(useRZfilter_.begin(), useRZfilter_.end(), fitterName) > 0);

          // Get 3D track candidates found by Hough transform (plus optional r-z filters/duplicate removal) in this sector.
          const list<L1track3D>& vecTrk3D = make3Dtrk->trackCands3D(useRZfilt);

          // Find list where fitted tracks will be stored.
          list<L1fittedTrack>& fitTrksInSec = mapmFitTrks(iPhiSec, iEtaReg)[fitterName];

          // Fit all tracks in this sector
          for (const L1track3D& trk : vecTrk3D) {
            // Ensure stubs assigned to this track is digitized with respect to the phi sector the track is in.
            if (settings_.enableDigitize()) {
              const vector<Stub*>& stubsOnTrk = trk.stubs();
              for (Stub* s : stubsOnTrk) {
                // Also digitize stub in way this specific track fitter uses it.
                s->digitize(iPhiSec, Stub::DigiStage::TF);
              }
            }

            L1fittedTrack fitTrk = fitterWorkerMap_[fitterName]->fit(trk);

            if (fitTrk.accepted()) {  // If fitter accepted track, then store it.
              // Optionally digitize fitted track, degrading slightly resolution.
              if (settings_.enableDigitize())
                fitTrk.digitizeTrack(fitterName);
              // Store fitted tracks, such that there is one fittedTracks corresponding to each HT tracks.
              fitTrksInSec.push_back(fitTrk);
            }
          }
        }
      }
    }

    // Run duplicate track removal on the fitted tracks if requested.

    // Initialize the duplicate track removal algorithm that can optionally be run after the track fit.
    DupFitTrkKiller killDupFitTrks(&settings_);

    // Loop over all the fitting algorithms we used.
    for (const string& fitterName : trackFitters_) {
      for (unsigned int iPhiSec = 0; iPhiSec < settings_.numPhiSectors(); iPhiSec++) {
        for (unsigned int iEtaReg = 0; iEtaReg < settings_.numEtaRegions(); iEtaReg++) {
          // Get fitted tracks in sector
          const list<L1fittedTrack>& fitTrksInSec = mapmFitTrks(iPhiSec, iEtaReg)[fitterName];

          // Run duplicate removal
          list<const L1fittedTrack*> filteredFitTrksInSec = killDupFitTrks.filter(fitTrksInSec);

          // Prepare TTTrack collection.
          for (const L1fittedTrack* fitTrk : filteredFitTrksInSec) {
            // Convert these fitted tracks to EDM format for output (used for collaborative work outside TMTT group).
            TTTrack<Ref_Phase2TrackerDigi_> fitTTTrack = converter.makeTTTrack(fitTrk, iPhiSec, iEtaReg);
            allFitTTTracksForOutput[fitterName]->push_back(fitTTTrack);
          }

          // Store fitted tracks from entire tracker.
          mapFinalTracks[fitterName].insert(
              mapFinalTracks[fitterName].end(), filteredFitTrksInSec.begin(), filteredFitTrksInSec.end());
        }
      }
    }

    // Debug printout
    if (debug_) {
      PrintL1trk() << "INPUT #TPs = " << vTPs.size() << " #STUBs = " << vStubs.size();
      unsigned int numHTtracks = 0;
      for (unsigned int iPhiSec = 0; iPhiSec < settings_.numPhiSectors(); iPhiSec++) {
        for (unsigned int iEtaReg = 0; iEtaReg < settings_.numEtaRegions(); iEtaReg++) {
          const Make3Dtracks* make3Dtrk = mMake3Dtrks(iPhiSec, iEtaReg).get();
          numHTtracks += make3Dtrk->trackCands3D(false).size();
        }
      }
      PrintL1trk() << "Number of tracks after HT = " << numHTtracks;
      for (const auto& p : mapFinalTracks) {
        const string& fitName = p.first;
        const list<const L1fittedTrack*> fittedTracks = p.second;
        PrintL1trk() << "Number of tracks after " << fitName << " track helix fit = " << fittedTracks.size();
      }
    }

    // Allow histogramming to plot undigitized variables.
    for (Stub* stub : vStubs) {
      if (settings_.enableDigitize())
        stub->setDigitizeWarningsOn(false);
    }

    // Fill histograms to monitor input data & tracking performance.
    hists_.fill(inputData, mSectors, mHtRphis, mMake3Dtrks, mapFinalTracks);

    //=== Store output EDM track and hardware stub collections.
    if (settings_.enableOutputIntermediateTTTracks()) {
      iEvent.put(std::move(htTTTracksForOutput), "TML1TracksHT");
      if (runRZfilter_)
        iEvent.put(std::move(rzTTTracksForOutput), "TML1TracksRZ");
    }
    for (const string& fitterName : trackFitters_) {
      string edmName = string("TML1Tracks") + fitterName;
      iEvent.put(std::move(allFitTTTracksForOutput[fitterName]), edmName);
    }
  }

  void TMTrackProducer::globalEndJob(GlobalCacheTMTT* globalCacheTMTT) {
    const Settings& settings = globalCacheTMTT->settings();

    // Print stub window sizes that TMTT recommends CMS uses in FE chips.
    if (settings.printStubWindows())
      globalCacheTMTT->stubWindowSuggest().printResults();

    // Print (once) info about tracker geometry.
    globalCacheTMTT->hists().trackerGeometryAnalysis(globalCacheTMTT->listTrackerModule());

    PrintL1trk() << "\n Number of (eta,phi) sectors used = (" << settings.numEtaRegions() << ","
                 << settings.numPhiSectors() << ")";

    // Print job summary
    globalCacheTMTT->hists().endJobAnalysis(&(globalCacheTMTT->htRphiErrMon()));
  }

}  // namespace tmtt

DEFINE_FWK_MODULE(tmtt::TMTrackProducer);
