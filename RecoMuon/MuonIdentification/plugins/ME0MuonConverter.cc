/** \file ME0MuonConverter.cc
 *
 * \author David Nash
 */

#include <RecoMuon/MuonIdentification/plugins/ME0MuonConverter.h>

#include <FWCore/PluginManager/interface/ModuleDef.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <DataFormats/MuonReco/interface/EmulatedME0Segment.h>
#include <DataFormats/MuonReco/interface/EmulatedME0SegmentCollection.h>
#include <DataFormats/MuonReco/interface/ME0Muon.h>
#include <DataFormats/MuonReco/interface/ME0MuonCollection.h>

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


ME0MuonConverter::ME0MuonConverter(const edm::ParameterSet& pas) : iev(0) {
	
  produces<std::vector<reco::RecoChargedCandidate> >();  
}

ME0MuonConverter::~ME0MuonConverter() {}

void ME0MuonConverter::produce(edm::Event& ev, const edm::EventSetup& setup) {

  using namespace edm;

  using namespace reco;

  Handle <std::vector<ME0Muon> > OurMuons;
  ev.getByLabel <std::vector<ME0Muon> > ("me0SegmentMatcher", OurMuons);
  
  std::auto_ptr<RecoChargedCandidateCollection> oc( new RecoChargedCandidateCollection());

  for (std::vector<ME0Muon>::const_iterator thisMuon = OurMuons->begin();
       thisMuon != OurMuons->end(); ++thisMuon){

    TrackRef tkRef = thisMuon->innerTrack();
    
    Particle::Charge q = tkRef->charge();
    Particle::LorentzVector p4(tkRef->px(), tkRef->py(), tkRef->pz(), tkRef->p());
    Particle::Point vtx(tkRef->vx(),tkRef->vy(), tkRef->vz());

    int pid = 13;
    if(abs(q)==1) pid = q < 0 ? 13 : -13;
    reco::RecoChargedCandidate cand(q, p4, vtx, pid);
    cand.setTrack(thisMuon->innerTrack());

    oc->push_back(cand);
  }
    
  ev.put(oc);
}


 DEFINE_FWK_MODULE(ME0MuonConverter);
