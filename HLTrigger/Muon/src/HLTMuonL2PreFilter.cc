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

//
// constructors and destructor
//

HLTMuonL2PreFilter::HLTMuonL2PreFilter(const edm::ParameterSet& iConfig):
  beamSpotTag_( iConfig.getParameter<edm::InputTag>("BeamSpotTag") ),
  candTag_( iConfig.getParameter<edm::InputTag >("CandTag") ),
  previousCandTag_( iConfig.getParameter<edm::InputTag >("PreviousCandTag") ),
  seedMapTag_( iConfig.getParameter<edm::InputTag >("SeedMapTag") ),
  minN_( iConfig.getParameter<int>("MinN") ),
  maxEta_( iConfig.getParameter<double>("MaxEta") ),
  minNhits_( iConfig.getParameter<int>("MinNhits") ),
  maxDr_( iConfig.getParameter<double>("MaxDr") ),
  maxDz_( iConfig.getParameter<double>("MaxDz") ),
  minPt_( iConfig.getParameter<double>("MinPt") ),
  nSigmaPt_( iConfig.getParameter<double>("NSigmaPt") ), 
  saveTag_( iConfig.getUntrackedParameter<bool>("SaveTag", false) )
{
  using namespace std;

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
    ss<<"    MinNhits = "<<minNhits_<<endl;
    ss<<"    MaxDr = "<<maxDr_<<endl;
    ss<<"    MaxDz = "<<maxDz_<<endl;
    ss<<"    MinPt = "<<minPt_<<endl;
    ss<<"    NSigmaPt = "<<nSigmaPt_<<endl;
    ss<<"    SaveTag = "<<saveTag_;
    LogDebug("HLTMuonL2PreFilter")<<ss.str();
  }

  //register your products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTMuonL2PreFilter::~HLTMuonL2PreFilter()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool HLTMuonL2PreFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;
  using namespace l1extra;

  // The filter object
  auto_ptr<TriggerFilterObjectWithRefs> filterproduct(new TriggerFilterObjectWithRefs(path(), module()));

  // save Tag
  if(saveTag_) filterproduct->addCollectionTag(candTag_);

  // get hold of all muon candidates available at this level
  Handle<RecoChargedCandidateCollection> allMuons;
  iEvent.getByLabel(candTag_, allMuons);

  // get hold of the beam spot
  Handle<BeamSpot> beamSpotHandle;
  iEvent.getByLabel(beamSpotTag_, beamSpotHandle);
  BeamSpot::Point beamSpot = beamSpotHandle->position();

  // get the L2 to L1 map object for this event
  HLTMuonL2ToL1Map mapL2ToL1(previousCandTag_, seedMapTag_, iEvent);

  // look at all allMuons,  check cuts and add to filter object
  int n = 0;
  for(RecoChargedCandidateCollection::const_iterator cand=allMuons->begin(); cand!=allMuons->end(); cand++){
    TrackRef mu = cand->get<TrackRef>();

    // check if this muon passed previous level 
    if(!mapL2ToL1.isTriggeredByL1(mu)) continue;

    // eta cut
    if(fabs(mu->eta()) > maxEta_) continue;

    // cut on number of hits
    if(mu->numberOfValidHits() < minNhits_) continue;

    //dr cut
    if(fabs(mu->dxy(beamSpot)) > maxDr_) continue;

    //dz cut
    if(fabs(mu->dz(beamSpot)) > maxDz_) continue;

    // Pt threshold cut
    double pt = mu->pt();
    double abspar0 = fabs(mu->parameter(0));
    double ptLx = pt;
    // convert 50% efficiency threshold to 90% efficiency threshold
    if(abspar0 > 0) ptLx += nSigmaPt_*mu->error(0)/abspar0*pt;
    if(ptLx < minPt_) continue;

    // add the good candidate to the filter object
    filterproduct->addObject(TriggerMuon, RecoChargedCandidateRef(Ref<RecoChargedCandidateCollection>(allMuons, cand-allMuons->begin())));

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
        <<'\t'<<scientific<<mu->charge()*mu->pt()*(1. + ((mu->parameter(0) != 0) ? nSigmaPt_*mu->error(0)/fabs(mu->parameter(0)) : 0.))
        <<'\t'<<fixed<<mu->eta()
        <<'\t'<<fixed<<mu->phi()
        <<'\t'<<mu->numberOfValidHits()
        <<'\t'<<scientific<<mu->d0()
        <<'\t'<<scientific<<mu->dz()
        <<'\t'<<mapL2ToL1.getL1Keys(mu)
        <<'\t'<<mapL2ToL1.isTriggeredByL1(mu);
      vector<RecoChargedCandidateRef> firedMuons;
      filterproduct->getObjects(TriggerMuon, firedMuons);
      ss<<'\t'<<(find(firedMuons.begin(), firedMuons.end(), RecoChargedCandidateRef(Ref<RecoChargedCandidateCollection>(allMuons, cand-allMuons->begin()))) != firedMuons.end())
        <<endl;
    }
    ss<<"-----------------------------------------------------------------------------------------------------------------------"<<endl;
    ss<<"Decision of filter is "<<accept<<", number of muons passing = "<<filterproduct->muonSize();
    LogDebug("HLTMuonL2PreFilter")<<ss.str();
  }

  // put filter object into the Event
  iEvent.put(filterproduct);
   
  return accept;
}

