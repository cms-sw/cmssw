// -*- C++ -*-
//
// Package:    JetPlusTracks
// Class:      JetPlusTrackProducer
// 
/**\class JetPlusTrackProducer JetPlusTrackProducer.cc JetPlusTrackProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Olga Kodolova,40 R-A12,+41227671273,
//         Created:  Fri Feb 19 10:14:02 CET 2010
//
//


// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetPlusTracks/plugins/JetPlusTrackAddonSeedProducer.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/TrackJetCollection.h"
#include "DataFormats/JetReco/interface/TrackJet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <string>

using namespace std;

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
JetPlusTrackAddonSeedProducer::JetPlusTrackAddonSeedProducer(const edm::ParameterSet& iConfig)
{
   //register your products
   srcCaloJets = iConfig.getParameter<edm::InputTag>("srcCaloJets");
   srcTrackJets = iConfig.getParameter<edm::InputTag>("srcTrackJets");
   srcPVs_ = iConfig.getParameter<edm::InputTag>("srcPVs");
   ptCUT = iConfig.getParameter<double>("ptCUT");
   dRcone = iConfig.getParameter<double>("dRcone");
   usePAT = iConfig.getParameter<bool>("UsePAT");
 
   produces<reco::CaloJetCollection>("ak4CaloJetsJPTSeed"); 

   input_jets_token_ = consumes<edm::View<reco::CaloJet> >(srcCaloJets);
   input_trackjets_token_ = consumes<edm::View<reco::TrackJet> >(srcTrackJets);
   input_vertex_token_ = consumes<reco::VertexCollection>(srcPVs_);
   tokenPFCandidates_ = consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("PFCandidates"));      
   input_ctw_token_ = consumes<CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("towerMaker"));
}


JetPlusTrackAddonSeedProducer::~JetPlusTrackAddonSeedProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
JetPlusTrackAddonSeedProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm; 


 // std::cout<<" RecoJets::JetPlusTrackAddonSeedProducer::produce "<<std::endl;


// get stuff from Event
  edm::Handle <edm::View <reco::CaloJet> > jets_h;
  iEvent.getByToken (input_jets_token_, jets_h);

  edm::Handle<edm::View <reco::TrackJet> > jetsTrackJets;
  iEvent.getByToken(input_trackjets_token_, jetsTrackJets);

  auto pCaloOut = std::make_unique<reco::CaloJetCollection>();

   if (jetsTrackJets.isValid()) {
     if(!jetsTrackJets->empty() ) {
     // std::cout<<" AddonSeed::The size of trackjets "<<jetsTrackJets->size()<<" "<<jets_h->size()<<std::endl;
      for (unsigned ijet = 0; ijet < jetsTrackJets->size(); ++ijet) {
          const reco::TrackJet* jet = &(*(jetsTrackJets->refAt(ijet)));
          int iflag = 0;
         for (unsigned i = 0; i < jets_h->size(); ++i) {
          const reco::CaloJet* oldjet = &(*(jets_h->refAt(i)));
          double deta = fabs(jet->eta()-oldjet->eta());
          double dphi = fabs(jet->phi()-oldjet->phi()); 
          if(dphi>4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
          double dr = sqrt(dphi*dphi+deta*deta);
          if(dr<dRcone) iflag = 1;  
         } // Calojets

     if(iflag == 1) continue;
       // std::cout<<" AddonSeed::There is the additional trackjet seed "<<jet->pt()<<" "<<jet->eta()<<
     // " "<<jet->phi()<<std::endl;
      double caloen= 0.;
      double hadinho = 0.;
      double hadinhb = 0.;
      double hadinhe = 0.;
      double hadinhf = 0.;
      double emineb = 0.;
      double eminee = 0.;
      double eminhf = 0.;
      double eefraction = 0.;
      double hhfraction = 0.;
      int ncand = 0;

      if(usePAT) {
      edm::Handle<pat::PackedCandidateCollection> pfCandidates;
      iEvent.getByToken( tokenPFCandidates_, pfCandidates);
      if(!pfCandidates.isValid()) {
      //  std::cout<<" No PFCandidate collection "<<std::endl;
        return;
      } else {
      for(unsigned int i = 0, n = pfCandidates->size(); i < n; ++i) {
        const pat::PackedCandidate &pf = (*pfCandidates)[i];
        double  deta=(*jet).eta()-pf.eta();
        double  dphi=(*jet).phi()-pf.phi();

        if(dphi > 4.*atan(1.) ) dphi = dphi-8.*atan(1.);
        if(dphi < -1.*4.*atan(1.) ) dphi = dphi+8.*atan(1.);
        double dr = sqrt(dphi*dphi+deta*deta);
        if(dr>0.4) continue;
        // jetconstit
        caloen = caloen + pf.energy()*pf.caloFraction();
         hadinho += 0.;
         if(fabs(pf.eta())<=1.4) hadinhb += pf.energy()*pf.caloFraction()*pf.hcalFraction();
         if(fabs(pf.eta())<3.&&(fabs(pf.eta())>3.)>1.4) hadinhe += pf.energy()*pf.caloFraction()*pf.hcalFraction();
         if(fabs(pf.eta())>=3.) hadinhf += pf.energy()*pf.caloFraction()*pf.hcalFraction();
         if(fabs(pf.eta())<=1.4) emineb += pf.energy()*pf.caloFraction()*(1.-pf.hcalFraction());
         if(fabs(pf.eta())<3.&&(fabs(pf.eta())>3.)>1.4) eminee += pf.energy()*pf.caloFraction()*(1.-pf.hcalFraction());
         if(fabs(pf.eta())>=3.) eminhf += pf.energy()*pf.caloFraction()*(1.-pf.hcalFraction());
         ncand++;
      } // Calojet
     }
    } else {
  //    std::cout<<" RECO "<<std::endl;
      edm::Handle<CaloTowerCollection> ct;
      iEvent.getByToken(input_ctw_token_, ct);
      if(ct.isValid()) {
      for(CaloTowerCollection::const_iterator it = ct->begin();it != ct->end(); it++) {  
        double  deta=(*jet).eta()-(*it).eta();
        double  dphi=(*jet).phi()-(*it).phi();
        if(dphi > 4.*atan(1.) ) dphi = dphi-8.*atan(1.);
        if(dphi < -1.*4.*atan(1.) ) dphi = dphi+8.*atan(1.);
        double dr = sqrt(dphi*dphi+deta*deta);
        if(dr>0.4) continue;
        caloen = caloen + (*it).energy();
         hadinho += (*it).energyInHO();
         hadinhb += (*it).energyInHB();
         hadinhe += (*it).energyInHE();
         hadinhf += 0.5*(*it).energyInHF();
         emineb += (*it).energy()-(*it).energyInHB();
         eminee += (*it).energy()-(*it).energyInHE();
         eminhf += 0.5*(*it).energyInHF();
         ncand++;
      }
      } 
     } 
         eefraction = (emineb+eminee)/caloen;
         hhfraction = (hadinhb+hadinhe+hadinhf+hadinho)/caloen;
      
        double trackp = sqrt(pow(jet->pt(),2)+pow(jet->pz(),2)); 
  //      std::cout<<" Caloenergy "<<caloen<<"area"<<jet->jetArea()<<std::endl;
        if(caloen <= 0.) caloen = 0.001;
        math::XYZTLorentzVector pcalo4(caloen*jet->px()/trackp,
                                        caloen*jet->py()/trackp,
                                        caloen*jet->pz()/trackp,
                                        caloen);
       reco::CaloJet::Specific calospe;
       calospe.mTowersArea = -1*ncand;
       calospe.mHadEnergyInHO=hadinho;
       calospe.mHadEnergyInHB=hadinhb;
       calospe.mHadEnergyInHE=hadinhe;
       calospe.mHadEnergyInHF=hadinhf;
       calospe.mEmEnergyInEB=emineb;
       calospe.mEmEnergyInEE=eminee;
       calospe.mEmEnergyInHF=eminhf;
       calospe.mEnergyFractionEm=eefraction/caloen;
       calospe.mEnergyFractionHadronic=hhfraction/caloen;
        
       reco::CaloJet mycalo(pcalo4,jet->primaryVertex()->position(),calospe);
       mycalo.setJetArea(0.5024);

       //std::cout<<" AddonSeed::New Calojet "<<mycalo.pt()<<std::endl;
//       std::cout<<" Caloenergy "<<caloen<<"area"<<jet->jetArea()<<std::endl;
//       std::cout<<" CaloJetArea "<<mycalo.jetArea()<<std::endl;

       pCaloOut->push_back(mycalo);

     } // trackjets
   } // jets
  } // There is trackjet collection

 // std::cout<<" AddonSeed::size of collection "<<pCaloOut->size()<<std::endl;
  iEvent.put(std::move(pCaloOut),"ak4CaloJetsJPTSeed");   
 // theEvent.put(std::move(outputTColl),"tracksFromPF"
}

//define this as a plug-in
DEFINE_FWK_MODULE(JetPlusTrackAddonSeedProducer);
