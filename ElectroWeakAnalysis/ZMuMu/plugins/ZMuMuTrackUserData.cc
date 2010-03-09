#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
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

class ZMuMuTrackUserData : public edm::EDProducer {
public:
  ZMuMuTrackUserData( const edm::ParameterSet & );   
private:
  void produce( edm::Event &, const edm::EventSetup & );
  
  InputTag src_,beamSpot_, primaryVertices_;
  double alpha_, beta_; 
  string hltPath_; 
  template<typename T>
  double isolation(const T & t, double alpha, double beta, bool relIso);
};

template<typename T>
double ZMuMuTrackUserData::isolation(const T & t, double alpha, double beta, bool relIso = false) {

  double isovalueTrk  = t.trackIso();
  double isovalueEcal = t.ecalIso();
  double isovalueHcal = t.hcalIso();

  //double iso =  isovalueTrk + isovalueEcal + isovalueHcal;
  double iso = alpha*( ((1+beta)/2*isovalueEcal) + ((1-beta)/2*isovalueHcal) ) + ((1-alpha)*isovalueTrk);
  // inserire anche questo nell'ntupla
  if(relIso) iso /= t.pt();
  return iso;
}

ZMuMuTrackUserData::ZMuMuTrackUserData( const ParameterSet & cfg ):
  src_( cfg.getParameter<InputTag>( "src" ) ),
  alpha_(cfg.getParameter<double>("alpha") ),
  beta_(cfg.getParameter<double>("beta") ), 
  beamSpot_(cfg.getParameter<InputTag>( "beamSpot" ) ),
  primaryVertices_(cfg.getParameter<InputTag>( "primaryVertices" ) ),
  hltPath_(cfg.getParameter<std::string >("hltPath") ){
  produces<std::vector<pat::GenericParticle> >();
}

void ZMuMuTrackUserData::produce( Event & evt, const EventSetup & ) {
  Handle<vector<pat::GenericParticle>  > tracks;
  evt.getByLabel(src_,tracks);

  Handle<BeamSpot> beamSpotHandle;
  if (!evt.getByLabel(beamSpot_, beamSpotHandle)) {
    std::cout << ">>> No beam spot found !!!"<<std::endl;
  }

  Handle<VertexCollection> primaryVertices;  // Collection of primary Vertices
  if (!evt.getByLabel(primaryVertices_, primaryVertices)){
    std::cout << ">>> No primary vertices  found !!!"<<std::endl;
    }

  auto_ptr<vector<pat::GenericParticle> > tkColl( new vector<pat::GenericParticle> (*tracks) );
  for (unsigned int i = 0; i< tkColl->size();++i){
    pat::GenericParticle & tk = (*tkColl)[i];
    float iso = isolation(tk,alpha_, beta_);
    float relIso = isolation(tk,alpha_, beta_, true);
    tk.setIsolation(pat::User1Iso, iso);
    tk.setIsolation(pat::User2Iso, relIso);
    TrackRef muTrkRef = tk.track();
    float zDaudxyFromBS = muTrkRef->dxy(beamSpotHandle->position());
    float zDaudzFromBS = muTrkRef->dz(beamSpotHandle->position());
    float zDaudxyFromPV = muTrkRef->dxy(primaryVertices->begin()->position() );
    float zDaudzFromPV = muTrkRef->dz(primaryVertices->begin()->position() );	
    const pat::TriggerObjectStandAloneCollection muHLTMatches =  tk.triggerObjectMatchesByPath( hltPath_);
    float muHLTBit;
    int dimTrig = muHLTMatches.size();
    if(dimTrig !=0 ){
      muHLTBit = 1;
    } else {
      muHLTBit = 0; 
    }
    tk.addUserFloat("zDau_dxyFromBS", zDaudxyFromBS);
    tk.addUserFloat("zDau_dzFromBS", zDaudzFromBS);
    tk.addUserFloat("zDau_dxyFromPV", zDaudxyFromPV);
    tk.addUserFloat("zDau_dzFromPV", zDaudzFromPV);
    tk.addUserFloat("zDau_HLTBit",muHLTBit);

  }

  evt.put( tkColl);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( ZMuMuTrackUserData );

