#include <L1Trigger/TrackFindingTMTT/plugins/TMTrackProducer.h>
#include <L1Trigger/TrackFindingTMTT/interface/InputData.h>
#include <L1Trigger/TrackFindingTMTT/interface/Settings.h>
#include <L1Trigger/TrackFindingTMTT/interface/Histos.h>
#include <L1Trigger/TrackFindingTMTT/interface/Sector.h>
#include <L1Trigger/TrackFindingTMTT/interface/HTrphi.h>
#include <L1Trigger/TrackFindingTMTT/interface/Get3Dtracks.h>
#include <L1Trigger/TrackFindingTMTT/interface/KillDupFitTrks.h>
#include <L1Trigger/TrackFindingTMTT/interface/TrackFitGeneric.h>
#include <L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h>
#include <L1Trigger/TrackFindingTMTT/interface/ConverterToTTTrack.h>
#include "L1Trigger/TrackFindingTMTT/interface/HTcell.h"
#include "L1Trigger/TrackFindingTMTT/interface/MuxHToutputs.h"
#include "L1Trigger/TrackFindingTMTT/interface/MiniHTstage.h"
#include "L1Trigger/TrackFindingTMTT/interface/StubWindowSuggest.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "boost/numeric/ublas/matrix.hpp"
#include <iostream>
#include <vector>
#include <set>

// If this is defined, then TTTrack collections will be output using tracks after HT (and optionally r-z filter) too.
//#define OutputHT_TTracks

using namespace std;
using boost::numeric::ublas::matrix;

namespace TMTT {

TMTrackProducer::TMTrackProducer(const edm::ParameterSet& iConfig):
  stubInputTag( consumes<DetSetVec>( iConfig.getParameter<edm::InputTag>("stubInputTag") ) ),
  trackerGeometryInfo_()
{
  // Get configuration parameters
  settings_ = new Settings(iConfig);

  if (settings_->enableMCtruth()) {
    // These lines use lots of CPU, even if no use of truth info is made later.
    tpInputTag = consumes<TrackingParticleCollection>( iConfig.getParameter<edm::InputTag>("tpInputTag") );
    stubTruthInputTag = consumes<TTStubAssMap>( iConfig.getParameter<edm::InputTag>("stubTruthInputTag") );
    clusterTruthInputTag = consumes<TTClusterAssMap>( iConfig.getParameter<edm::InputTag>("clusterTruthInputTag") );
    genJetInputTag_ = consumes<reco::GenJetCollection>( iConfig.getParameter<edm::InputTag>("genJetInputTag") );
  }

  trackFitters_ = settings_->trackFitters();
  useRZfilter_ = settings_->useRZfilter();
  runRZfilter_  = (useRZfilter_.size() > 0); // Do any fitters require an r-z track filter to be run?

  // Tame debug printout.
  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(4);

  // Book histograms.
  hists_ = new Histos( settings_ );
  hists_->book();

  // Create track fitting algorithm (& internal histograms if it uses them)
  for (const string& fitterName : trackFitters_) {
    fitterWorkerMap_[ fitterName ] = TrackFitGeneric::create(fitterName, settings_);
    fitterWorkerMap_[ fitterName ]->bookHists(); 
  }

  //--- Define EDM output to be written to file (if required) 

#ifdef OutputHT_TTracks
  // L1 tracks found by Hough Transform 
  produces< TTTrackCollection >( "TML1TracksHT" ).setBranchAlias("TML1TracksHT");
  // L1 tracks found by r-z track filter.
  if (runRZfilter_) produces< TTTrackCollection >( "TML1TracksRZ" ).setBranchAlias("TML1TracksRZ");
#endif
  // L1 tracks after track fit by each of the fitting algorithms under study
  for (const string& fitterName : trackFitters_) {
    string edmName = string("TML1Tracks") + fitterName;
    produces< TTTrackCollection >(edmName).setBranchAlias(edmName);
  }
}


void TMTrackProducer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) 
{
  // Get the B-field and store its value in the Settings class.

  edm::ESHandle<MagneticField> magneticFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
  const MagneticField* theMagneticField = magneticFieldHandle.product();
  float bField = theMagneticField->inTesla(GlobalPoint(0,0,0)).z(); // B field in Tesla.
  cout<<endl<<"--- B field = "<<bField<<" Tesla ---"<<endl<<endl;

  settings_->setBfield(bField);

  // Initialize track fitting algorithm at start of run (especially with B-field dependent variables).
  for (const string& fitterName : trackFitters_) {
    fitterWorkerMap_[ fitterName ]->initRun(); 
  }

  // Print info on tilted modules
  edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get( trackerGeometryHandle );
  const TrackerGeometry*  trackerGeometry = trackerGeometryHandle.product();

  edm::ESHandle<TrackerTopology> trackerTopologyHandle;
  iSetup.get<TrackerTopologyRcd>().get(trackerTopologyHandle);
  const TrackerTopology*  trackerTopology = trackerTopologyHandle.product();

  trackerGeometryInfo_.getTiltedModuleInfo( settings_, trackerTopology, trackerGeometry );
}

void TMTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // edm::Handle<TrackingParticleCollection> tpHandle;
  // edm::EDGetToken token( consumes<edm::View<TrackingParticleCollection>>( edm::InputTag( "mix", "MergedTrackTruth" ) ) );
  // iEvent.getByToken(inputTag, tpHandle );


  // Note useful info about MC truth particles and about reconstructed stubs .
  InputData inputData(iEvent, iSetup, settings_, tpInputTag, stubInputTag, stubTruthInputTag, clusterTruthInputTag, genJetInputTag_);

  const vector<TP>&          vTPs   = inputData.getTPs();
  const vector<const Stub*>& vStubs = inputData.getStubs(); 

  // Creates matrix of Sector objects, which decide which stubs are in which (eta,phi) sector
  matrix<Sector>  mSectors(settings_->numPhiSectors(), settings_->numEtaRegions());
  // Create matrix of r-phi Hough-Transform arrays, with one-to-one correspondence to sectors.
  matrix<HTrphi>  mHtRphis(settings_->numPhiSectors(), settings_->numEtaRegions());
  // Create matrix of Get3Dtracks objects, to run optional r-z track filter, with one-to-one correspondence to sectors.
  matrix<Get3Dtracks>  mGet3Dtrks(settings_->numPhiSectors(), settings_->numEtaRegions());

  //=== Initialization
  // Create utility for converting L1 tracks from our private format to official CMSSW EDM format.
  const ConverterToTTTrack converter(settings_);
#ifdef OutputHT_TTracks
  // Storage for EDM L1 track collection to be produced directly from HT output.
  std::unique_ptr<TTTrackCollection>  htTTTracksForOutput(new TTTrackCollection);
  // Storage for EDM L1 track collection to be produced directly from r-z track filter output (if run).
  std::unique_ptr<TTTrackCollection>  rzTTTracksForOutput(new TTTrackCollection);
#endif
  // Storage for EDM L1 track collection to be produced from fitted tracks (one for each fit algorithm being used).
  // auto_ptr cant be stored in std containers, so use C one, together with map noting which element corresponds to which algorithm.
  const unsigned int nFitAlgs = trackFitters_.size();
  std::unique_ptr<TTTrackCollection> allFitTTTracksForOutput[nFitAlgs]; 
  map<string, unsigned int> locationInsideArray;
  unsigned int ialg = 0;
  for (const string& fitterName : trackFitters_) {
    std::unique_ptr<TTTrackCollection> fitTTTracksForOutput(new TTTrackCollection);
    allFitTTTracksForOutput[ialg] =  std::move( fitTTTracksForOutput );
    locationInsideArray[fitterName] = ialg++;
  }

  //=== Do tracking in the r-phi Hough transform within each sector.

  // Fill Hough-Transform arrays with stubs.
  for (unsigned int iPhiSec = 0; iPhiSec < settings_->numPhiSectors(); iPhiSec++) {
    for (unsigned int iEtaReg = 0; iEtaReg < settings_->numEtaRegions(); iEtaReg++) {

      Sector& sector = mSectors(iPhiSec, iEtaReg);
      HTrphi& htRphi = mHtRphis(iPhiSec, iEtaReg);

      // Initialize constants for this sector.
      sector.init(settings_, iPhiSec, iEtaReg); 
      htRphi.init(settings_, iPhiSec, iEtaReg, sector.etaMin(), sector.etaMax(), sector.phiCentre());

      // Check sector is enabled (always true, except if user disabled some for special studies).
      if (settings_->isHTRPhiEtaRegWhitelisted(iEtaReg)) {

	for (const Stub* stub: vStubs) {
	  // Digitize stub as would be at input to GP. This doesn't need the nonant number, since we assumed an integer number of
	  // phi digitisation  bins inside an nonant. N.B. This changes the coordinates & bend stored in the stub.
	  // The cast allows us to ignore the "const".
	  if (settings_->enableDigitize()) (const_cast<Stub*>(stub))->digitizeForGPinput(iPhiSec);

	  // Check if stub is inside this sector
	  bool inside = sector.inside( stub );

	  if (inside) {
	    // Check which eta subsectors within the sector the stub is compatible with (if subsectors being used).
	    const vector<bool> inEtaSubSecs =  sector.insideEtaSubSecs( stub );

	    // Digitize stub if as would be at input to HT, which slightly degrades its coord. & bend resolution, affecting the HT performance.
	    if (settings_->enableDigitize()) (const_cast<Stub*>(stub))->digitizeForHTinput(iPhiSec);

	    // Store stub in Hough transform array for this sector, indicating its compatibility with eta subsectors with sector.
	    htRphi.store( stub, inEtaSubSecs );
	  }
	}
      }

      // Find tracks in r-phi HT array.
      htRphi.end(); // Calls htArrayRphi_.end() -> HTBase::end()
    }
  }

  if (settings_->muxOutputsHT() > 0) {
    // Multiplex outputs of several HT onto one pair of output opto-links.
    // This only affects tracking performance if option busySectorKill is enabled, so that tracks that
    // can't be sent down the link within the time-multiplexed period are killed.
    MuxHToutputs muxHT(settings_);
    muxHT.exec(mHtRphis);
  }

  // Optionally, run 2nd stage mini HT -- WITHOUT TRUNCATION ???
  if ( settings_->miniHTstage() ) {
    MiniHTstage miniHTstage( settings_ );
    miniHTstage.exec( mHtRphis );
  }

  //=== Make 3D tracks, optionally running r-z track filters (such as Seed Filter) & duplicate track removal. 

  for (unsigned int iPhiSec = 0; iPhiSec < settings_->numPhiSectors(); iPhiSec++) {
    for (unsigned int iEtaReg = 0; iEtaReg < settings_->numEtaRegions(); iEtaReg++) {

      const Sector& sector = mSectors(iPhiSec, iEtaReg);

      // Get tracks found by r-phi HT.
      const HTrphi& htRphi = mHtRphis(iPhiSec, iEtaReg);
      const vector<L1track2D>& vecTracksRphi = htRphi.trackCands2D();

      Get3Dtracks& get3Dtrk = mGet3Dtrks(iPhiSec, iEtaReg);
      // Initialize utility for making 3D tracks from 2D ones.
      get3Dtrk.init(settings_, iPhiSec, iEtaReg, sector.etaMin(), sector.etaMax(), sector.phiCentre());

      // Convert 2D tracks found by HT to 3D tracks (optionally by running r-z filters & duplicate track removal)
      get3Dtrk.run(vecTracksRphi);

#ifdef OutputHT_TTracks
      // Convert these tracks to EDM format for output (used for collaborative work outside TMTT group).
      // Do this for tracks output by HT & optionally also for those output by r-z track filter.
      const vector<L1track3D>& vecTrk3D_ht = get3Dtrk.trackCands3D(false);
      for (const L1track3D& trk : vecTrk3D_ht) {
        TTTrack< Ref_Phase2TrackerDigi_ > htTTTrack = converter.makeTTTrack(trk, iPhiSec, iEtaReg);
        htTTTracksForOutput->push_back( htTTTrack );
      }

      if (runRZfilter_) {
        const vector<L1track3D>& vecTrk3D_rz = get3Dtrk.trackCands3D(true);
        for (const L1track3D& trk : vecTrk3D_rz) {
          TTTrack< Ref_Phase2TrackerDigi_ > rzTTTrack = converter.makeTTTrack(trk, iPhiSec, iEtaReg);
          rzTTTracksForOutput->push_back( rzTTTrack );
        }
      }
#endif
    }
  }

  // Initialize the duplicate track removal algorithm that can optionally be run after the track fit.
  KillDupFitTrks killDupFitTrks;
  killDupFitTrks.init(settings_, settings_->dupTrkAlgFit());
  
  //=== Do a helix fit to all the track candidates.

  map<string, vector<L1fittedTrack>> fittedTracks;
  // Initialize with empty vector in case no fitted tracks found.
  for (const string& fitterName : trackFitters_) { // Loop over fit algos.
    fittedTracks[fitterName] = vector<L1fittedTrack>(); 
  }

  for (unsigned int iPhiSec = 0; iPhiSec < settings_->numPhiSectors(); iPhiSec++) {
    for (unsigned int iEtaReg = 0; iEtaReg < settings_->numEtaRegions(); iEtaReg++) {

      const Get3Dtracks& get3Dtrk = mGet3Dtrks(iPhiSec, iEtaReg);

      // Loop over all the fitting algorithms we are trying.
      for (const string& fitterName : trackFitters_) {

	// Does this fitter require r-z track filter to be run before it?
	bool useRZfilt = (std::count(useRZfilter_.begin(), useRZfilter_.end(), fitterName) > 0);

        // Get 3D track candidates found by Hough transform (plus optional r-z filters/duplicate removal) in this sector.
	const vector<L1track3D>& vecTrk3D = get3Dtrk.trackCands3D(useRZfilt);

        // Fit all tracks in this sector
	vector<L1fittedTrack> fittedTracksInSec;
        for (const L1track3D& trk : vecTrk3D) {

	  // IRT
	  //bool OK = (trk.getMatchedTP() != nullptr && trk.getMatchedTP()->pt() > 50 && fabs(trk.getMatchedTP()->eta()) > 1.4 && fabs(trk.getMatchedTP()->eta()) < 1.8);
          //if (trk.getNumStubs() != trk.getNumLayers()) OK = false;
          //if (not OK) continue;

          // Ensure stubs assigned to this track is digitized with respect to the phi sector the track is in.
	  if (settings_->enableDigitize()) {
  	    const vector<const Stub*>& stubsOnTrk = trk.getStubs();
            for (const Stub* s : stubsOnTrk) {
             (const_cast<Stub*>(s))->digitizeForHTinput(iPhiSec);
	     // Also digitize stub in way this specific track fitter uses it.
             (const_cast<Stub*>(s))->digitizeForSForTFinput(fitterName);          
	    }
	  }

	  L1fittedTrack fitTrack = fitterWorkerMap_[fitterName]->fit(trk);

	  if (fitTrack.accepted()) { // If fitter accepted track, then store it.
  	    // Optionally digitize fitted track, degrading slightly resolution.
 	     if (settings_->enableDigitize()) fitTrack.digitizeTrack(fitterName);
	    // Store fitted tracks, such that there is one fittedTracks corresponding to each HT tracks.
	    // N.B. Tracks rejected by the fit are also stored, but marked.
	    fittedTracksInSec.push_back(fitTrack);
	  }
	}

	// Run duplicate track removal on the fitted tracks if requested.
	const vector<L1fittedTrack> filtFittedTracksInSec = killDupFitTrks.filter( fittedTracksInSec );

	// Store fitted tracks from entire tracker.
	for (const L1fittedTrack& fitTrk : filtFittedTracksInSec) {
	  fittedTracks[fitterName].push_back(fitTrk);
	  // Convert these fitted tracks to EDM format for output (used for collaborative work outside TMTT group).
	  TTTrack< Ref_Phase2TrackerDigi_ > fitTTTrack = converter.makeTTTrack(fitTrk, iPhiSec, iEtaReg);
	  allFitTTTracksForOutput[locationInsideArray[fitterName]]->push_back(fitTTTrack);
	}
      }
    }
  }

  // Debug printout
  unsigned int static nEvents = 0;
  nEvents++;
  if (settings_->debug() >= 1 && nEvents <= 1000) {
    cout<<"INPUT #TPs = "<<vTPs.size()<<" #STUBs = "<<vStubs.size()<<endl;
    unsigned int numHTtracks = 0;
    for (unsigned int iPhiSec = 0; iPhiSec < settings_->numPhiSectors(); iPhiSec++) {
      for (unsigned int iEtaReg = 0; iEtaReg < settings_->numEtaRegions(); iEtaReg++) {
	const Get3Dtracks& get3Dtrk = mGet3Dtrks(iPhiSec, iEtaReg);
	numHTtracks += get3Dtrk.trackCands3D(false).size();
      }
    }
    cout<<"Number of tracks after HT = "<<numHTtracks<<endl;
    for (const auto& p : fittedTracks) {
      const string& fitName = p.first;
      const vector<L1fittedTrack>& fittedTracks = p.second;
      cout<<"Number of tracks after "<<fitName<<" track helix fit = "<<fittedTracks.size()<<endl;
    }
  }


  // Allow histogramming to plot undigitized variables.
  for (const Stub* stub: vStubs) {
    if (settings_->enableDigitize()) (const_cast<Stub*>(stub))->setDigitizeWarningsOn(false);
  }

  // Fill histograms to monitor input data & tracking performance.
  hists_->fill(inputData, mSectors, mHtRphis, mGet3Dtrks, fittedTracks);

  //=== Store output EDM track and hardware stub collections.
#ifdef OutputHT_TTracks
  iEvent.put( std::move( htTTTracksForOutput ),  "TML1TracksHT");
  if (runRZfilter_) iEvent.put( std::move( rzTTTracksForOutput ),  "TML1TracksRZ");
#endif
  for (const string& fitterName : trackFitters_) {
    string edmName = string("TML1Tracks") + fitterName;
    iEvent.put(std::move( allFitTTTracksForOutput[locationInsideArray[fitterName]] ), edmName);
  }
}


void TMTrackProducer::endJob() 
{
  // Print stub window sizes that TMTT recommends CMS uses in FE chips.
  if (settings_->printStubWindows()) StubWindowSuggest::printResults();  

  // Optional debug printout from track fitters at end of job.
  for (const string& fitterName : trackFitters_) {
    fitterWorkerMap_[ fitterName ]->endJob(); 
  }

  // Print job summary
  hists_->trackerGeometryAnalysis(trackerGeometryInfo_);
  hists_->endJobAnalysis();

  for (const string& fitterName : trackFitters_) {
    //cout << "# of duplicated stubs = " << fitterWorkerMap_[fitterName]->nDupStubs() << endl;
    delete fitterWorkerMap_[ string(fitterName) ];
  }

  cout<<endl<<"Number of (eta,phi) sectors used = (" << settings_->numEtaRegions() << "," << settings_->numPhiSectors()<<")"<<endl; 

}

DEFINE_FWK_MODULE(TMTrackProducer);

}
