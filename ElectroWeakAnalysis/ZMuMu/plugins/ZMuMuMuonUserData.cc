#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

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
  
  InputTag src_,beamSpot_, primaryVertices_;
  double alpha_, beta_; 
  double ptThreshold_, etEcalThreshold_, etHcalThreshold_ ,dRVetoTrk_, dRTrk_, dREcal_ , dRHcal_;
  string hltPath_; 
  template<typename T>
  vector<double> isolation(const T & t, double ptThreshold, double etEcalThreshold, double etHcalThreshold , double dRVetoTrk, double dRTrk, double dREcal , double dRHcal,  double alpha, double beta);
};

template<typename T>
vector<double> ZMuMuMuonUserData::isolation(const T & t, double ptThreshold, double etEcalThreshold, double etHcalThreshold , double dRVetoTrk, double dRTrk, double dREcal , double dRHcal,  double alpha, double beta) {

  vector<double> iso;
  const pat::IsoDeposit * trkIso = t.isoDeposit(pat::TrackIso);
  const pat::IsoDeposit * ecalIso = t.isoDeposit(pat::EcalIso);
  const pat::IsoDeposit * hcalIso = t.isoDeposit(pat::HcalIso);  
  
  Direction dir = Direction(t.eta(), t.phi());
   
  pat::IsoDeposit::AbsVetos vetosTrk;
  vetosTrk.push_back(new ConeVeto( dir, dRVetoTrk ));
  vetosTrk.push_back(new ThresholdVeto( ptThreshold ));
  
  pat::IsoDeposit::AbsVetos vetosEcal;
  vetosEcal.push_back(new ConeVeto( dir, 0.));
  vetosEcal.push_back(new ThresholdVeto( etEcalThreshold ));
  
  pat::IsoDeposit::AbsVetos vetosHcal;
  vetosHcal.push_back(new ConeVeto( dir, 0. ));
  vetosHcal.push_back(new ThresholdVeto( etHcalThreshold ));
  
  double isovalueTrk = (trkIso->sumWithin(dRTrk,vetosTrk));
  double isovalueEcal = (ecalIso->sumWithin(dREcal,vetosEcal));
  double isovalueHcal = (hcalIso->sumWithin(dRHcal,vetosHcal));
    
  iso.push_back(isovalueTrk);
  //cout<<"isoTrk"<<isovalueTrk<<" "<<t.trackIso()<<endl;
  iso.push_back(isovalueEcal);
  //cout<<"isoEcal"<<isovalueEcal<<" "<<t.ecalIso()<<endl;
  iso.push_back(isovalueHcal);
  //cout<<"isoHcal"<<isovalueHcal<<" "<<t.hcalIso()<<endl;
  //double isovalueTrk  = t.trackIso();
  //double isovalueEcal = t.ecalIso();
  //double isovalueHcal = t.hcalIso();

  //double iso =  isovalueTrk + isovalueEcal + isovalueHcal;
  double combIso = alpha*( ((1+beta)/2*isovalueEcal) + ((1-beta)/2*isovalueHcal) ) + ((1-alpha)*isovalueTrk);
  iso.push_back(combIso);
  //cout<<"combIso"<<iso[3]<<endl;  
// inserire anche questo nell'ntupla
  double relIso = combIso /= t.pt();
  iso.push_back(relIso);
  //cout<<"relIso"<<iso[4]<<endl;
  return iso;
}

ZMuMuMuonUserData::ZMuMuMuonUserData( const ParameterSet & cfg ):
  src_( cfg.getParameter<InputTag>( "src" ) ),
  beamSpot_(cfg.getParameter<InputTag>( "beamSpot" ) ),
  primaryVertices_(cfg.getParameter<InputTag>( "primaryVertices" ) ),
  alpha_(cfg.getParameter<double>("alpha") ),
  beta_(cfg.getParameter<double>("beta") ), 
  ptThreshold_(cfg.getParameter<double >("ptThreshold") ),
  etEcalThreshold_(cfg.getParameter<double >("etEcalThreshold") ),
  etHcalThreshold_(cfg.getParameter<double >("etHcalThreshold") ),
  dRVetoTrk_(cfg.getParameter<double >("dRVetoTrk") ),
  dRTrk_(cfg.getParameter<double >("dRTrk") ),
  dREcal_(cfg.getParameter<double >("dREcal") ),
  dRHcal_(cfg.getParameter<double >("dRHcal") ),
  hltPath_(cfg.getParameter<std::string >("hltPath") ){
  produces<std::vector<pat::Muon> >();
}

void ZMuMuMuonUserData::produce( Event & evt, const EventSetup & ) {
  Handle<vector<pat::Muon>  > muons;
  evt.getByLabel(src_,muons);

  Handle<BeamSpot> beamSpotHandle;
  evt.getByLabel(beamSpot_, beamSpotHandle);

  Handle<VertexCollection> primaryVertices;  // Collection of primary Vertices
  evt.getByLabel(primaryVertices_, primaryVertices);

  auto_ptr<vector<pat::Muon> > muonColl( new vector<pat::Muon> (*muons) );
  for (unsigned int i = 0; i< muonColl->size();++i){
    pat::Muon & m = (*muonColl)[i];
    //pat::Muon *mu = new pat::Muon(m); 
    vector<double> iso = isolation(m,ptThreshold_, etEcalThreshold_, etHcalThreshold_ ,dRVetoTrk_, dRTrk_, dREcal_ , dRHcal_, alpha_, beta_);
    m.setIsolation(pat::User1Iso, iso[0]);
    //cout<<"iso1 "<<iso[0]<<endl;
    m.setIsolation(pat::User2Iso, iso[1]);
    //cout<<"iso2 "<<iso[1]<<endl;
    m.setIsolation(pat::User3Iso, iso[2]);
    //cout<<"iso3 "<<iso[2]<<endl;
    m.setIsolation(pat::User4Iso, iso[3]);
    //cout<<"iso4 "<<iso[3]<<endl; 
    m.setIsolation(pat::User5Iso, iso[4]);
    //cout<<"iso5 "<<iso[4]<<endl;
    TrackRef muTrkRef = m.innerTrack();
    TrackRef muSaRef = m.outerTrack();
    float zDaudxyFromBS = 10000;
    float zDaudzFromBS = 10000;
    float zDaudxyFromPV = 10000;
    float zDaudzFromPV = 10000;
   
    if (muTrkRef.isNonnull() && m.isGlobalMuon() == true){
      zDaudxyFromBS = muTrkRef->dxy(beamSpotHandle->position());
      zDaudzFromBS = muTrkRef->dz(beamSpotHandle->position());
      zDaudxyFromPV = muTrkRef->dxy(primaryVertices->begin()->position() );
      zDaudzFromPV = muTrkRef->dz(primaryVertices->begin()->position() );
    }	
    else if (muSaRef.isNonnull() && m.isStandAloneMuon() == true){
      zDaudxyFromBS = muSaRef->dxy(beamSpotHandle->position());
      zDaudzFromBS = muSaRef->dz(beamSpotHandle->position());
      zDaudxyFromPV = muSaRef->dxy(primaryVertices->begin()->position() );
      zDaudzFromPV = muSaRef->dz(primaryVertices->begin()->position() );
    }
    const pat::TriggerObjectStandAloneCollection muHLTMatches =  m.triggerObjectMatchesByPath( hltPath_);
    float muHLTBit;
    int dimTrig = muHLTMatches.size();
	if(dimTrig !=0 ){
	  muHLTBit = 1;
	} else {
	  muHLTBit = 0; 
	}
    m.addUserFloat("zDau_dxyFromBS", zDaudxyFromBS);
    m.addUserFloat("zDau_dzFromBS", zDaudzFromBS);
    m.addUserFloat("zDau_dxyFromPV", zDaudxyFromPV);
    m.addUserFloat("zDau_dzFromPV", zDaudzFromPV);
    m.addUserFloat("zDau_HLTBit",muHLTBit);

  }

  evt.put( muonColl);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( ZMuMuMuonUserData );

