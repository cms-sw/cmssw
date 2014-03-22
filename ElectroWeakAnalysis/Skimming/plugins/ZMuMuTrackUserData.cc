#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
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
  void produce( edm::Event &, const edm::EventSetup & ) override;

  EDGetTokenT<vector<pat::GenericParticle> > srcToken_;
  EDGetTokenT<BeamSpot> beamSpotToken_;
  EDGetTokenT<VertexCollection> primaryVerticesToken_;
  double ptThreshold_, etEcalThreshold_, etHcalThreshold_ ,dRVetoTrk_, dRTrk_, dREcal_ , dRHcal_;
  double alpha_, beta_;
  template<typename T>
  vector<double> isolation(const T & t, double ptThreshold, double etEcalThreshold, double etHcalThreshold , double dRVetoTrk, double dRTrk, double dREcal , double dRHcal,  double alpha, double beta);

};

template<typename T>
vector<double> ZMuMuTrackUserData::isolation(const T & t, double ptThreshold, double etEcalThreshold, double etHcalThreshold , double dRVetoTrk, double dRTrk, double dREcal , double dRHcal,  double alpha, double beta) {

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
  iso.push_back(isovalueEcal);
  iso.push_back(isovalueHcal);

  //double iso =  isovalueTrk + isovalueEcal + isovalueHcal;
  double combIso = alpha*( ((1+beta)/2*isovalueEcal) + ((1-beta)/2*isovalueHcal) ) + ((1-alpha)*isovalueTrk);

  iso.push_back(combIso);
  double relIso = combIso /= t.pt();
  iso.push_back(relIso);
  return iso;
}

ZMuMuTrackUserData::ZMuMuTrackUserData( const ParameterSet & cfg ):
  srcToken_(consumes<vector<pat::GenericParticle> > ( cfg.getParameter<InputTag>( "src" ) ) ),
  beamSpotToken_(consumes<BeamSpot> (cfg.getParameter<InputTag>( "beamSpot" ) ) ),
  primaryVerticesToken_(consumes<VertexCollection> (cfg.getParameter<InputTag>( "primaryVertices" ) ) ),
  ptThreshold_(cfg.getParameter<double >("ptThreshold") ),
  etEcalThreshold_(cfg.getParameter<double >("etEcalThreshold") ),
  etHcalThreshold_(cfg.getParameter<double >("etHcalThreshold") ),
  dRVetoTrk_(cfg.getParameter<double >("dRVetoTrk") ),
  dRTrk_(cfg.getParameter<double >("dRTrk") ),
  dREcal_(cfg.getParameter<double >("dREcal") ),
  dRHcal_(cfg.getParameter<double >("dRHcal") ),
  alpha_(cfg.getParameter<double>("alpha") ),
  beta_(cfg.getParameter<double>("beta") ){
  produces<std::vector<pat::GenericParticle> >();
}

void ZMuMuTrackUserData::produce( Event & evt, const EventSetup & ) {
  Handle<vector<pat::GenericParticle>  > tracks;
  evt.getByToken(srcToken_,tracks);

  Handle<BeamSpot> beamSpotHandle;
  evt.getByToken(beamSpotToken_, beamSpotHandle);

  Handle<VertexCollection> primaryVertices;  // Collection of primary Vertices
  evt.getByToken(primaryVerticesToken_, primaryVertices);

  auto_ptr<vector<pat::GenericParticle> > tkColl( new vector<pat::GenericParticle> (*tracks) );
  for (unsigned int i = 0; i< tkColl->size();++i){
    pat::GenericParticle & tk = (*tkColl)[i];
    vector<double> iso = isolation(tk,ptThreshold_, etEcalThreshold_, etHcalThreshold_ ,dRVetoTrk_, dRTrk_, dREcal_ , dRHcal_, alpha_, beta_);
    tk.setIsolation(pat::User1Iso, iso[0]);
    //    cout << "track User1Iso " << iso[0] << endl;
    tk.setIsolation(pat::User2Iso, iso[1]);
    //cout << "track User2Iso " << iso[1] << endl;
    tk.setIsolation(pat::User3Iso, iso[2]);
    //cout << "track User3Iso " << iso[2] << endl;
    tk.setIsolation(pat::User4Iso, iso[3]);
    //cout << "track User4Iso " << iso[3] << endl;
    tk.setIsolation(pat::User5Iso, iso[4]);
    //cout << "track User5Iso " << iso[4] << endl;



    float zDaudxyFromBS = -1 ;
    float zDaudzFromBS = -1;
    float zDaudxyFromPV = -1;
    float zDaudzFromPV = -1;
    float zDauNofMuChambers = -1;
    float zDauNofMuMatches = -1;
    float zDauChi2 = -1;
    float zDauTrkChi2 = -1;
    float zDauSaChi2 = -1;
    float zDauNofMuonHits =- 1;
    float zDauNofStripHits = -1;
    float zDauNofPixelHits = -1;
    float zDauMuEnergyEm = -1;
    float zDauMuEnergyHad = -1;

    TrackRef muTrkRef = tk.track();
    if (muTrkRef.isNonnull()){
      zDaudxyFromBS = muTrkRef->dxy(beamSpotHandle->position());
      zDaudzFromBS = muTrkRef->dz(beamSpotHandle->position());
      zDaudxyFromPV = muTrkRef->dxy(primaryVertices->begin()->position() );
      zDaudzFromPV = muTrkRef->dz(primaryVertices->begin()->position() );
      zDauChi2 = muTrkRef->normalizedChi2();
      zDauTrkChi2 = muTrkRef->normalizedChi2();
      zDauNofStripHits = muTrkRef->hitPattern().numberOfValidStripHits();
      zDauNofPixelHits = muTrkRef->hitPattern().numberOfValidPixelHits();
    }
    tk.addUserFloat("zDau_dxyFromBS", zDaudxyFromBS);
    tk.addUserFloat("zDau_dzFromBS", zDaudzFromBS);
    tk.addUserFloat("zDau_dxyFromPV", zDaudxyFromPV);
    tk.addUserFloat("zDau_dzFromPV", zDaudzFromPV);
    tk.addUserFloat("zDau_NofMuonHits" , zDauNofMuonHits );
    tk.addUserFloat("zDau_TrkNofStripHits" , zDauNofStripHits );
    tk.addUserFloat("zDau_TrkNofPixelHits" , zDauNofPixelHits );
    tk.addUserFloat("zDau_NofMuChambers", zDauNofMuChambers);
    tk.addUserFloat("zDau_NofMuMatches", zDauNofMuMatches);
    tk.addUserFloat("zDau_Chi2", zDauChi2);
    tk.addUserFloat("zDau_TrkChi2", zDauTrkChi2);
    tk.addUserFloat("zDau_SaChi2", zDauSaChi2);
    tk.addUserFloat("zDau_MuEnergyEm", zDauMuEnergyEm);
    tk.addUserFloat("zDau_MuEnergyHad", zDauMuEnergyHad);


  }

  evt.put( tkColl);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( ZMuMuTrackUserData );

