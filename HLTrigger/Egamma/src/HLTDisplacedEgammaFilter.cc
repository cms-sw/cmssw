/** \class HLTDisplacedEgammaFilter
 *
 * $Id: HLTDisplacedEgammaFilter.cc,v 1.1 2012/03/21 05:35:10 sckao Exp $
 *
 * \authors Shih-Chuan Kao, Michael Sigamani, Juliette Alimena (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTDisplacedEgammaFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoTracker/TrackProducer/plugins/TrackProducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"

//
// constructors and destructor
//
HLTDisplacedEgammaFilter::HLTDisplacedEgammaFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) 
{
  inputTag_ = iConfig.getParameter< edm::InputTag > ("inputTag");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
  relaxed_ = iConfig.getUntrackedParameter<bool> ("relaxed",true) ;
  L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand"); 
  L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand"); 

  inputTrk   = iConfig.getParameter< edm::InputTag > ("inputTrack");
  trkPtCut   = iConfig.getParameter<double> ("trackPtCut");
  trkdRCut   = iConfig.getParameter<double> ("trackdRCut");
  maxTrkCut  = iConfig.getParameter<int> ("maxTrackCut");

  rechitsEB  = iConfig.getParameter< edm::InputTag > ("RecHitsEB");
  rechitsEE  = iConfig.getParameter< edm::InputTag > ("RecHitsEE");
  
  sMin_min     = iConfig.getParameter<double> ("sMin_min");
  sMin_max     = iConfig.getParameter<double> ("sMin_max");
  seedTimeMin  = iConfig.getParameter<double> ("seedTimeMin");
  seedTimeMax  = iConfig.getParameter<double> ("seedTimeMax");

}

HLTDisplacedEgammaFilter::~HLTDisplacedEgammaFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTDisplacedEgammaFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace trigger;

  // The filter object
  if (saveTags()) {
    filterproduct.addCollectionTag(L1IsoCollTag_);
    if (relaxed_) filterproduct.addCollectionTag(L1NonIsoCollTag_);
  }

  // Ref to Candidate object to be recorded in filter object
   edm::Ref<reco::RecoEcalCandidateCollection> ref;

  // get hold of filtered candidates
  //edm::Handle<reco::HLTFilterObjectWithRefs> recoecalcands;
  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByLabel (inputTag_,PrevFilterOutput);

  // get hold of collection of objects
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByLabel( inputTrk , tracks);

  // get the EcalRecHit
  edm::Handle<EcalRecHitCollection>      rechitsEB_ ;
  edm::Handle<EcalRecHitCollection>      rechitsEE_ ;
  iEvent.getByLabel( rechitsEB,     rechitsEB_ );
  iEvent.getByLabel( rechitsEE,     rechitsEE_ );

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;   
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);
 
  // look at all candidates,  check cuts and add to filter object
  int n(0);

  for (unsigned int i=0; i<recoecalcands.size(); i++) {
    
    ref = recoecalcands[i] ;

    // S_Minor Cuts from the seed cluster
    reco::CaloClusterPtr SCseed = ref->superCluster()->seed() ;
    const EcalRecHitCollection* rechits = ( fabs( ref->eta() ) < 1.479 ) ? rechitsEB_.product() : rechitsEE_.product() ;

    Cluster2ndMoments moments = EcalClusterTools::cluster2ndMoments(*SCseed, *rechits);
    float sMin =  moments.sMin  ;
    float sMaj =  moments.sMaj  ;
    if ( sMin < sMin_min || sMin > sMin_max ) continue ;
    if ( sMaj < sMaj_min || sMaj > sMaj_max ) continue ;

    // seed Time 
    std::pair<DetId, float> maxRH = EcalClusterTools::getMaximum( *SCseed, rechits );
    DetId seedCrystalId = maxRH.first;
    EcalRecHitCollection::const_iterator seedRH = rechits->find(seedCrystalId);
    float seedTime = (float)seedRH->time();
    if ( seedTime < seedTimeMin || seedTime > seedTimeMax ) continue ;
 
    //Track Veto
    
    int nTrk = 0 ;
    for (reco::TrackCollection::const_iterator it = tracks->begin(); it != tracks->end(); it++ )  {
        if ( it->pt() < trkPtCut ) continue ;
        LorentzVector trkP4( it->px(), it->py(), it->pz(), it->p() ) ;
        double dR =  ROOT::Math::VectorUtil::DeltaR( trkP4 , ref->p4()  ) ;
        if ( dR < trkdRCut )  nTrk++ ;
        if ( nTrk > maxTrkCut ) break ;
    }
    if ( nTrk > maxTrkCut ) continue ;     
    

    n++;
    // std::cout << "Passed eta: " << ref->eta() << std::endl;
    filterproduct.addObject(TriggerCluster, ref);
  }
  
  
  // filter decision
  bool accept(n>=ncandcut_);
  
  return accept;
}

