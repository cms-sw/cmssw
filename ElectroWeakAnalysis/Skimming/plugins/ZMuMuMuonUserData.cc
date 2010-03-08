#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/PatCandidates/interface/Isolation.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <vector>

using namespace edm;
using namespace std;
using namespace reco;
using namespace isodeposit;
//using namespace pat;

class ZMuMuMuonUserData : public edm::EDProducer {
public:
  ZMuMuMuonUserData( const edm::ParameterSet & );   
private:
  void produce( edm::Event &, const edm::EventSetup & );
  
  InputTag src_,zGenParticlesMatch_,beamSpot_, primaryVertices_;
  double alpha_, beta_; 
  
  template<typename T>
  double isolation(const T & t, double alpha, double beta, bool relIso);
};

template<typename T>
double ZMuMuMuonUserData::isolation(const T & t, double alpha, double beta, bool relIso = false) {

  double isovalueTrk  = t.trackIso();
  double isovalueEcal = t.ecalIso();
  double isovalueHcal = t.hcalIso();

  //double iso =  isovalueTrk + isovalueEcal + isovalueHcal;
  double iso = alpha*( ((1+beta)/2*isovalueEcal) + ((1-beta)/2*isovalueHcal) ) + ((1-alpha)*isovalueTrk);
  // inserire anche questo nell'ntupla
  if(relIso) iso /= t.pt();
  return iso;
}

ZMuMuMuonUserData::ZMuMuMuonUserData( const ParameterSet & cfg ):
  src_( cfg.getParameter<InputTag>( "src" ) ),
  alpha_(cfg.getParameter<double>("alpha")),
  beta_(cfg.getParameter<double>("beta")), 
  zGenParticlesMatch_(cfg.getParameter<InputTag>( "zGenParticlesMatch" )),
  beamSpot_(cfg.getParameter<InputTag>( "beamSpot" )),
  primaryVertices_(cfg.getParameter<InputTag>( "primaryVertices" )){
  produces<std::vector<pat::Muon> >();
}

void ZMuMuMuonUserData::produce( Event & evt, const EventSetup & ) {
  Handle<vector<pat::Muon>  > muons;
  evt.getByLabel(src_,muons);

  Handle<BeamSpot> beamSpotHandle;
  if (!evt.getByLabel(beamSpot_, beamSpotHandle)) {
    std::cout << ">>> No beam spot found !!!"<<std::endl;
  }

  Handle<VertexCollection> primaryVertices;  // Collection of primary Vertices
  if (!evt.getByLabel(primaryVertices_, primaryVertices)){
    std::cout << ">>> No primary vertices  found !!!"<<std::endl;
    }

  auto_ptr<vector<pat::Muon> > muonColl( new vector<pat::Muon> (*muons) );
  for (unsigned int i = 0; i< muonColl->size();++i){
    pat::Muon & m = (*muonColl)[i];
    float iso = isolation(m,alpha_, beta_);
    float relIso = isolation(m,alpha_, beta_, true);
    m.setIsolation(pat::User1Iso, iso);
    m.setIsolation(pat::User2Iso, relIso);
    float dummy = 44;
    m.addUserFloat("dummy",dummy);
    TrackRef muTrkRef = m.innerTrack();
    float zDaudxyFromBS = muTrkRef->dxy(beamSpotHandle->position());
    float zDaudzFromBS = muTrkRef->dz(beamSpotHandle->position());
    float zDaudxyFromPV = muTrkRef->dxy(primaryVertices->begin()->position() );
    float zDaudzFromPV = muTrkRef->dz(primaryVertices->begin()->position() );	
    cout<<"dxy from BS "<<zDaudxyFromBS<<endl;
    cout<<"dz from BS "<<zDaudzFromBS<<endl;
    cout<<"dxy from PV "<<zDaudxyFromPV<<endl;
    cout<<"dz from PV "<<zDaudzFromPV<<endl;
    m.addUserFloat("zDau_dxyFromBS", zDaudxyFromBS);
    m.addUserFloat("zDau_dzFromBS", zDaudzFromBS);
    m.addUserFloat("zDau_dxyFromPV", zDaudxyFromPV);
    m.addUserFloat("zDau_dzFromPV", zDaudzFromPV);
  }

  evt.put( muonColl);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( ZMuMuMuonUserData );

