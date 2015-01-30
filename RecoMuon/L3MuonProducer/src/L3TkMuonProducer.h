#ifndef RecoMuon_L3MuonProducer_L3TkMuonProducer_H
#define RecoMuon_L3MuonProducer_L3TkMuonProducer_H

/**  \class L3TkMuonProducer
 * 
 *    This module creates a skimed list of reco::Track (pointing to the original TrackExtra and TrackingRecHitOwnedVector
 *    One highest pT track per L1/L2 is selected, requiring some quality.
 *
 *   \author  J-R Vlimant.
 */


#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"


namespace edm {class ParameterSet; class Event; class EventSetup;}

class L3TkMuonProducer : public edm::stream::EDProducer<> {

 public:

  /// constructor with config
  L3TkMuonProducer(const edm::ParameterSet&);
  
  /// destructor
  virtual ~L3TkMuonProducer(); 
  
  /// produce candidates
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  typedef edm::Ref<L3MuonTrajectorySeedCollection> SeedRef;
  
 private:
  
  // L3/GLB Collection Label
  edm::InputTag theL3CollectionLabel; 
  edm::EDGetTokenT<reco::TrackCollection> trackToken_;



  //psuedo ref is L2 or L1 ref.
  typedef std::pair<unsigned int,unsigned int> pseudoRef;
  typedef std::map<pseudoRef, std::pair<reco::TrackRef,SeedRef> > LXtoL3sMap;

  pseudoRef makePseudoRef(const L3MuonTrajectorySeed& s){
    reco::TrackRef l2ref = s.l2Track();
    if (l2ref.isNull()){
      l1extra::L1MuonParticleRef l1ref = s.l1Particle();
      return std::make_pair(l1ref.id().id(),l1ref.key());
    }else return std::make_pair(l2ref.id().id(),l2ref.key());
  }

  bool sharedSeed(const L3MuonTrajectorySeed& s1,const L3MuonTrajectorySeed& s2);


  //ordering functions
  static  bool seedRefBypT(const SeedRef & s1, const SeedRef & s2){
    double pt1,pt2;
    reco::TrackRef l2ref1 = s1->l2Track();
    if (l2ref1.isNull()) pt1=s1->l1Particle()->pt();
    else pt1=l2ref1->pt();
    reco::TrackRef l2ref2 = s2->l2Track();
    if (l2ref2.isNull()) pt2=s2->l1Particle()->pt();
    else pt2=l2ref2->pt();
    return (pt1>pt2);
  }
  
  static bool trackRefBypT(const reco::TrackRef & t1,const reco::TrackRef & t2){
    return (t1->pt()>t2->pt());
  }



};

#endif
