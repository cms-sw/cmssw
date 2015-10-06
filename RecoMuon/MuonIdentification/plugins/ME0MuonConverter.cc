/** \file ME0MuonConverter.cc
 *
 * \author David Nash
 */


#include <FWCore/PluginManager/interface/ModuleDef.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

#include <Geometry/Records/interface/MuonGeometryRecord.h>



#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "TLorentzVector.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"


#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

/** \class ME0MuonConverter 
 * Produces a collection of ME0Segment's in endcap muon ME0s. 
 *
 *
 * \author David Nash
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"


#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>



#include <DataFormats/MuonReco/interface/ME0Muon.h>
#include <DataFormats/MuonReco/interface/ME0MuonCollection.h>



class ME0MuonConverter : public edm::stream::EDProducer<> {
public:
  /// Constructor
  explicit ME0MuonConverter(const edm::ParameterSet&);
  /// Destructor
  ~ME0MuonConverter();
  /// Produce the converted collection
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

    

private:

  //edm::EDGetTokenT<std::vector<reco::ME0Muon>> OurMuonsToken_;
  edm::EDGetTokenT<ME0MuonCollection> OurMuonsToken_;
};


ME0MuonConverter::ME0MuonConverter(const edm::ParameterSet& pas) {
	
  produces<std::vector<reco::RecoChargedCandidate> >();  
  edm::InputTag OurMuonsTag ("me0SegmentMatching");
  OurMuonsToken_ = consumes<ME0MuonCollection>(OurMuonsTag);

}

ME0MuonConverter::~ME0MuonConverter() {}

void ME0MuonConverter::produce(edm::Event& ev, const edm::EventSetup& setup) {

  using namespace edm;

  using namespace reco;

  // Handle <std::vector<ME0Muon> > OurMuons;
  // ev.getByToken <std::vector<ME0Muon> > (OurMuonsToken_, OurMuons);


  Handle <ME0MuonCollection> OurMuons;
  ev.getByToken(OurMuonsToken_, OurMuons);

  std::auto_ptr<RecoChargedCandidateCollection> oc( new RecoChargedCandidateCollection());

  for (std::vector<ME0Muon>::const_iterator thisMuon = OurMuons->begin();
       thisMuon != OurMuons->end(); ++thisMuon){
    TrackRef tkRef = thisMuon->innerTrack();
    
    Particle::Charge q = tkRef->charge();
    Particle::LorentzVector p4(tkRef->px(), tkRef->py(), tkRef->pz(), tkRef->p());
    Particle::Point vtx(tkRef->vx(),tkRef->vy(), tkRef->vz());

    int pid = 0;
    if(abs(q)==1) pid = q < 0 ? 13 : -13;
    reco::RecoChargedCandidate cand(q, p4, vtx, pid);
    cand.setTrack(thisMuon->innerTrack());
    oc->push_back(cand);
  }
    
  ev.put(oc);
}


 DEFINE_FWK_MODULE(ME0MuonConverter);
