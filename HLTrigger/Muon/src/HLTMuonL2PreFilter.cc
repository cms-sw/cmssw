/** \class HLTMuonL2PreFilter
 *
 * See header file for documentation
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/Muon/interface/HLTMuonL2PreFilter.h"
#include "HLTrigger/Muon/interface/HLTMuonL2ToL1Map.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

//
// constructors and destructor
//

HLTMuonL2PreFilter::HLTMuonL2PreFilter(const edm::ParameterSet& iConfig): HLTFilter(iConfig),
  beamSpotTag_( iConfig.getParameter<edm::InputTag>("BeamSpotTag") ),
  beamSpotToken_(consumes<reco::BeamSpot>(beamSpotTag_)),
  candTag_( iConfig.getParameter<edm::InputTag >("CandTag") ),
  candToken_(consumes<reco::RecoChargedCandidateCollection>(candTag_)),
  previousCandTag_( iConfig.getParameter<edm::InputTag >("PreviousCandTag") ),
  previousCandToken_(consumes<trigger::TriggerFilterObjectWithRefs>(previousCandTag_)),
  seedMapTag_( iConfig.getParameter<edm::InputTag >("SeedMapTag") ),
  seedMapToken_(consumes<SeedMap>(seedMapTag_)),
  minN_( iConfig.getParameter<int>("MinN") ),
  maxEta_( iConfig.getParameter<double>("MaxEta") ),
  absetaBins_( iConfig.getParameter<std::vector<double> >("AbsEtaBins") ),
  minNstations_( iConfig.getParameter<std::vector<int> >("MinNstations") ),
  minNhits_( iConfig.getParameter<std::vector<int> >("MinNhits") ),
  cutOnChambers_( iConfig.getParameter<bool>("CutOnChambers") ),
  minNchambers_( iConfig.getParameter<std::vector<int> >("MinNchambers") ),
  maxDr_( iConfig.getParameter<double>("MaxDr") ),
  minDr_( iConfig.getParameter<double>("MinDr") ),
  maxDz_( iConfig.getParameter<double>("MaxDz") ),
  min_DxySig_(iConfig.getParameter<double> ("MinDxySig")),
  minPt_( iConfig.getParameter<double>("MinPt") ),
  nSigmaPt_( iConfig.getParameter<double>("NSigmaPt") )
{
  using namespace std;

  // check that number of eta bins matches number of nStation cuts
  if( minNstations_.size()!=absetaBins_.size() ||
      minNhits_.size()!=absetaBins_.size()     ||
      ( cutOnChambers_ && minNchambers_.size()!=absetaBins_.size() ) ) {
    throw cms::Exception("Configuration") << "Number of MinNstations, MinNhits, or MinNchambers cuts "
					  << "does not match number of eta bins." << endl;
  }

  if(absetaBins_.size()>1) {
    for(unsigned int i=0; i<absetaBins_.size()-1; ++i) {
      if(absetaBins_[i+1]<=absetaBins_[i])
	throw cms::Exception("Configuration") << "Absolute eta bins must be in increasing order." << endl;
    }
  }

  // dump parameters for debugging
  if(edm::isDebugEnabled()){
    ostringstream ss;
    ss<<"Constructed with parameters:"<<endl;
    ss<<"    BeamSpotTag = "<<beamSpotTag_.encode()<<endl;
    ss<<"    CandTag = "<<candTag_.encode()<<endl;
    ss<<"    PreviousCandTag = "<<previousCandTag_.encode()<<endl;
    ss<<"    SeedMapTag = "<<seedMapTag_.encode()<<endl;
    ss<<"    MinN = "<<minN_<<endl;
    ss<<"    MaxEta = "<<maxEta_<<endl;
    ss<<"    MinNstations = ";
    for(unsigned int j=0; j<absetaBins_.size(); ++j) {
      ss<<minNstations_[j]<<" (|eta|<"<<absetaBins_[j]<<"), ";
    }
    ss<<endl;
    ss<<"    MinNhits = ";
    for(unsigned int j=0; j<absetaBins_.size(); ++j) {
      ss<<minNhits_[j]<<" (|eta|<"<<absetaBins_[j]<<"), ";
    }
    ss<<endl;
    ss<<"    CutOnChambers = " <<cutOnChambers_<<endl;
    if ( cutOnChambers_ ) {
      ss<<"    MinNchambers = ";
      for(unsigned int j=0; j<absetaBins_.size(); ++j) {
        ss<<minNchambers_[j]<<" (|eta|<"<<absetaBins_[j]<<"), ";
      }
    }
    ss<<endl;
    ss<<"    MaxDr = "<<maxDr_<<endl;
    ss<<"    MinDr = "<<minDr_<<endl;
    ss<<"    MaxDz = "<<maxDz_<<endl;
    ss<<"    MinDxySig = "<<min_DxySig_<<endl;
    ss<<"    MinPt = "<<minPt_<<endl;
    ss<<"    NSigmaPt = "<<nSigmaPt_<<endl;
    ss<<"    saveTags= "<<saveTags();
    LogDebug("HLTMuonL2PreFilter")<<ss.str();
  }
}

HLTMuonL2PreFilter::~HLTMuonL2PreFilter()
{
}

void
HLTMuonL2PreFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("BeamSpotTag",edm::InputTag("hltOfflineBeamSpot"));
  desc.add<edm::InputTag>("CandTag",edm::InputTag("hltL2MuonCandidates"));
  desc.add<edm::InputTag>("PreviousCandTag",edm::InputTag(""));
  desc.add<edm::InputTag>("SeedMapTag",edm::InputTag("hltL2Muons"));
  desc.add<int>("MinN",1);
  desc.add<double>("MaxEta",2.5);
  desc.add<std::vector<double> >("AbsEtaBins", std::vector<double>(1, 9999.));
  desc.add<std::vector<int> >("MinNstations", std::vector<int>(1, 1));
  desc.add<std::vector<int> >("MinNhits", std::vector<int>(1, 0));
  desc.add<bool> ("CutOnChambers", 0);
  desc.add<std::vector<int> >("MinNchambers", std::vector<int>(1, 0));
  desc.add<double>("MaxDr",9999.0);
  desc.add<double>("MinDr",-1.0);
  desc.add<double>("MaxDz",9999.0);
  desc.add<double>("MinDxySig",-1.0);
  desc.add<double>("MinPt",0.0);
  desc.add<double>("NSigmaPt",0.0);
  descriptions.add("hltMuonL2PreFilter",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool HLTMuonL2PreFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;
  using namespace l1extra;

  // save Tag
  if (saveTags()) filterproduct.addCollectionTag(candTag_);

  // get hold of all muon candidates available at this level
  Handle<RecoChargedCandidateCollection> allMuons;
  iEvent.getByToken(candToken_, allMuons);

  // get hold of the beam spot
  Handle<BeamSpot> beamSpotHandle;
  iEvent.getByToken(beamSpotToken_, beamSpotHandle);
  BeamSpot::Point beamSpot = beamSpotHandle->position();

  // get the L2 to L1 map object for this event
  HLTMuonL2ToL1Map mapL2ToL1(previousCandToken_, seedMapToken_, iEvent);

  // number of eta bins for cut on number of stations
  const std::vector<double>::size_type nAbsetaBins = absetaBins_.size();

  // look at all allMuons,  check cuts and add to filter object
  int n = 0;
  for(RecoChargedCandidateCollection::const_iterator cand=allMuons->begin(); cand!=allMuons->end(); cand++){
    TrackRef mu = cand->get<TrackRef>();

    // check if this muon passed previous level
    if(!mapL2ToL1.isTriggeredByL1(mu)) continue;

    // eta cut
    if(std::abs(mu->eta()) > maxEta_) continue;

    // cut on number of stations
    bool failNstations(false), failNhits(false), failNchambers(false);
    for(unsigned int i=0; i<nAbsetaBins; ++i) {
      if( std::abs(mu->eta())<absetaBins_[i] ) {
	if(mu->hitPattern().muonStationsWithAnyHits() < minNstations_[i]) {
	  failNstations=true;
	}
	if(mu->numberOfValidHits() < minNhits_[i]) {
	  failNhits=true;
	}
	if( cutOnChambers_ &&
	    ( mu->hitPattern().dtStationsWithAnyHits() +
	      mu->hitPattern().cscStationsWithAnyHits() < minNchambers_[i]) ) {
	  failNchambers=true;
	}
	break;
      }
    }
    if(failNstations || failNhits || failNchambers) continue;

    //dr cut
    if(std::abs(mu->dxy(beamSpot)) > maxDr_) continue;

    //dr cut
    if(std::abs(mu->dxy(beamSpot)) < minDr_) continue;

    //dz cut
    if(std::abs(mu->dz(beamSpot)) > maxDz_) continue;

    // dxy significance cut (safeguard against bizarre values)
    if (min_DxySig_ > 0 && (mu->dxyError() <= 0 || std::abs(mu->dxy(beamSpot)/mu->dxyError()) < min_DxySig_)) continue;

    // Pt threshold cut
    double pt = mu->pt();
    double abspar0 = std::abs(mu->parameter(0));
    double ptLx = pt;
    // convert 50% efficiency threshold to 90% efficiency threshold
    if(abspar0 > 0) ptLx += nSigmaPt_*mu->error(0)/abspar0*pt;
    if(ptLx < minPt_) continue;

    // add the good candidate to the filter object
    filterproduct.addObject(TriggerMuon, RecoChargedCandidateRef(Ref<RecoChargedCandidateCollection>(allMuons, cand-allMuons->begin())));

    n++;
  }

  // filter decision
  const bool accept (n >= minN_);

  // dump event for debugging
  if(edm::isDebugEnabled()){
    ostringstream ss;
    ss<<"L2mu#"
      <<'\t'<<"q*pt"<<'\t' //scientific is too wide
      <<'\t'<<"q*ptLx"<<'\t' //scientific is too wide
      <<'\t'<<"eta"
      <<'\t'<<"phi"
      <<'\t'<<"nStations"
      <<'\t'<<"nHits"
      <<'\t'<<"dr"<<'\t' //scientific is too wide
      <<'\t'<<"dz"<<'\t' //scientific is too wide
      <<'\t'<<"L1seed#"
      <<'\t'<<"isPrev"
      <<'\t'<<"isFired"
      <<endl;
    ss<<"-----------------------------------------------------------------------------------------------------------------------"<<endl;
    for (RecoChargedCandidateCollection::const_iterator cand = allMuons->begin(); cand != allMuons->end(); cand++) {
      TrackRef mu = cand->get<TrackRef>();
      ss<<setprecision(2)
        <<cand-allMuons->begin()
        <<'\t'<<scientific<<mu->charge()*mu->pt()
        <<'\t'<<scientific<<mu->charge()*mu->pt()*(1. + ((mu->parameter(0) != 0) ? nSigmaPt_*mu->error(0)/std::abs(mu->parameter(0)) : 0.))
        <<'\t'<<fixed<<mu->eta()
        <<'\t'<<fixed<<mu->phi()
        <<'\t'<<mu->hitPattern().muonStationsWithAnyHits()
        <<'\t'<<mu->numberOfValidHits()
        <<'\t'<<scientific<<mu->d0()
        <<'\t'<<scientific<<mu->dz()
        <<'\t'<<mapL2ToL1.getL1Keys(mu)
        <<'\t'<<mapL2ToL1.isTriggeredByL1(mu);
      vector<RecoChargedCandidateRef> firedMuons;
      filterproduct.getObjects(TriggerMuon, firedMuons);
      ss<<'\t'<<(find(firedMuons.begin(), firedMuons.end(), RecoChargedCandidateRef(Ref<RecoChargedCandidateCollection>(allMuons, cand-allMuons->begin()))) != firedMuons.end())
        <<endl;
    }
    ss<<"-----------------------------------------------------------------------------------------------------------------------"<<endl;
    ss<<"Decision of filter is "<<accept<<", number of muons passing = "<<filterproduct.muonSize();
    LogDebug("HLTMuonL2PreFilter")<<ss.str();
  }

  return accept;
}

