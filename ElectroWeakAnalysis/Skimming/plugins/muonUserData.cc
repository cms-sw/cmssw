 
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

#include <vector>

using namespace edm;
using namespace std;
using namespace reco;
using namespace isodeposit;
//using namespace pat;

class muonUserData : public edm::EDProducer {
public:
  muonUserData( const edm::ParameterSet & );   
private:
  void produce( edm::Event &, const edm::EventSetup & );
  
  InputTag src_;
  double ptThreshold_, etEcalThreshold_, etHcalThreshold_ ,dRVetoTrk_, dRTrk_, dREcal_ , dRHcal_,  alpha_, beta_; 


template<typename T>
double isolation(const T & t, double alpha, double beta, bool relIso);
};


template<typename T>
double muonUserData::isolation(const T & t, double alpha, double beta, bool relIso = false) {
   // on 34X: 
const pat::IsoDeposit * trkIso = t.isoDeposit(pat::TrackIso);
//  const pat::IsoDeposit * trkIso = t->trackerIsoDeposit();
   // on 34X 
const pat::IsoDeposit * ecalIso = t.isoDeposit(pat::EcalIso);
//  const pat::IsoDeposit * ecalIso = t->ecalIsoDeposit();
//    on 34X 
const pat::IsoDeposit * hcalIso = t.isoDeposit(pat::HcalIso);   
//    const pat::IsoDeposit * hcalIso = t->hcalIsoDeposit();

    Direction dir = Direction(t.eta(), t.phi());
    
    pat::IsoDeposit::AbsVetos vetosTrk;
    vetosTrk.push_back(new ConeVeto( dir, dRVetoTrk_ ));
    vetosTrk.push_back(new ThresholdVeto( ptThreshold_ ));
    
    pat::IsoDeposit::AbsVetos vetosEcal;
    vetosEcal.push_back(new ConeVeto( dir, 0.));
    vetosEcal.push_back(new ThresholdVeto( etEcalThreshold_ ));
    
    pat::IsoDeposit::AbsVetos vetosHcal;
    vetosHcal.push_back(new ConeVeto( dir, 0. ));
    vetosHcal.push_back(new ThresholdVeto( etHcalThreshold_ ));

    // non salvare 
    double isovalueTrk = (trkIso->sumWithin(dRTrk_,vetosTrk));
    double isovalueEcal = (ecalIso->sumWithin(dREcal_,vetosEcal));
    double isovalueHcal = (hcalIso->sumWithin(dRHcal_,vetosHcal));
    

    double iso = alpha*( ((1+beta)/2*isovalueEcal) + ((1-beta)/2*isovalueHcal) ) + ((1-alpha)*isovalueTrk) ;
    // inserire anche questo nell'ntupla
    if(relIso) iso /= t.pt();
    return iso;
}




muonUserData::muonUserData( const ParameterSet & cfg ):
  src_( cfg.getParameter<InputTag>( "src" ) ),
  ptThreshold_(cfg.getParameter<double>("ptThreshold")),
  etEcalThreshold_(cfg.getParameter<double>("etEcalThreshold")),
  etHcalThreshold_(cfg.getParameter<double>("etHcalThreshold")),
  dRVetoTrk_(cfg.getParameter<double>("deltaRVetoTrk")),
  dRTrk_(cfg.getParameter<double>("deltaRTrk")),
  dREcal_(cfg.getParameter<double>("deltaREcal")),
  dRHcal_(cfg.getParameter<double>("deltaRHcal")),
  alpha_(cfg.getParameter<double>("alpha")),
  beta_(cfg.getParameter<double>("beta"))
  {

    produces<std::vector<pat::Muon> >();
 
  
  
  
}




void muonUserData::produce( Event & evt, const EventSetup & ) {

  Handle<vector<pat::Muon>  > muons;
  evt.getByLabel(src_,muons);

  auto_ptr<vector<pat::Muon> > muonColl( new vector<pat::Muon> (*muons) );
  //  muonColl->reserve(muons.size());

  for (unsigned int i = 0; i< muonColl->size();++i){
      pat::Muon  m = (*muonColl)[i];

    // isolation as defined by us into the analyzer
    float iso = isolation(m,alpha_, beta_);

   // relative isolation as defined by us into the analyzer
    float relIso = isolation(m,alpha_, beta_, true);
    
    // tracker isolation : alpha =0
    float trkIso = isolation(m,0.0, beta_);
    
    // ecal isolation : alpha =1, beta =1
    float ecalIso = isolation(m,1.0, 1.0);
    
    // hcal isolation : alpha =1, beta =-1
    float hcalIso = isolation(m,1.0, -1.0);

    //m.setIsolation(pat::TrackIso, trkIso);
    //m.setIsolation(pat::EcalIso, ecalIso);
    //m.setIsolation(pat::HcalIso, hcalIso);
    m.setIsolation(pat::User1Iso, iso);
    m.setIsolation(pat::User2Iso, relIso);
    //m.addUserData("combinedIso",iso);
    //muonColl->push_back(m);
  }


  evt.put( muonColl);



}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( muonUserData );

