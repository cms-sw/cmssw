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

#include "RecoJets/JetPlusTracks/plugins/JetPlusTrackProducer.h"
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
using namespace jpt;

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
JetPlusTrackProducer::JetPlusTrackProducer(const edm::ParameterSet& iConfig)
{
   //register your products
   src = iConfig.getParameter<edm::InputTag>("src");
   srcTrackJets = iConfig.getParameter<edm::InputTag>("srcTrackJets");
   alias = iConfig.getUntrackedParameter<string>("alias");
   srcPVs_ = iConfig.getParameter<edm::InputTag>("srcPVs");
   vectorial_ = iConfig.getParameter<bool>("VectorialCorrection");
   useZSP = iConfig.getParameter<bool>("UseZSP");
   ptCUT = iConfig.getParameter<double>("ptCUT");

   mJPTalgo  = new JetPlusTrackCorrector(iConfig, consumesCollector());
   if(useZSP) mZSPalgo  = new ZSPJPTJetCorrector(iConfig);
   
   produces<reco::JPTJetCollection>().setBranchAlias(alias); 
   produces<reco::CaloJetCollection>().setBranchAlias("ak4CaloJetsJPT");

   input_jets_token_ = consumes<edm::View<reco::CaloJet> >(src);
   input_addjets_token_ = consumes<edm::View<reco::CaloJet> >(iConfig.getParameter<edm::InputTag>("srcAddCaloJets"));
   input_trackjets_token_ = consumes<edm::View<reco::TrackJet> >(srcTrackJets);
   input_vertex_token_ = consumes<reco::VertexCollection>(srcPVs_);
   mExtrapolations  = consumes<std::vector<reco::TrackExtrapolation> >
                                  (iConfig.getParameter<edm::InputTag> ("extrapolations"));   
}


JetPlusTrackProducer::~JetPlusTrackProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//
bool sort_by_pt (reco::JPTJet a,reco::JPTJet b) { return (a.pt()>b.pt());}

// ------------ method called to produce the data  ------------
void
JetPlusTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm; 

//  std::cout<<" RecoJets::JetPlusTrackProducer::produce "<<std::endl;

// get stuff from Event
  edm::Handle <edm::View <reco::CaloJet> > jets_h;
  iEvent.getByToken (input_jets_token_, jets_h);

//JetPlusTrackAddonSeed
  edm::Handle <edm::View <reco::CaloJet> > addjets_h;
  iEvent.getByToken (input_addjets_token_, addjets_h);

  //std::cout<<" Additional Calojets "<<addjets_h->size()<<std::endl;

  edm::Handle<edm::View <reco::TrackJet> > jetsTrackJets;
  iEvent.getByToken(input_trackjets_token_, jetsTrackJets);

  //std::cout<<" Additional Trackjets "<<jetsTrackJets->size()<<std::endl;

  edm::Handle <std::vector<reco::TrackExtrapolation> > iExtrapolations;
  iEvent.getByToken (mExtrapolations, iExtrapolations);

   edm::RefProd<reco::CaloJetCollection> pOut1RefProd = 
           iEvent.getRefBeforePut<reco::CaloJetCollection>();
   edm::Ref<reco::CaloJetCollection>::key_type idxCaloJet = 0;
 
  auto pOut = std::make_unique<reco::JPTJetCollection>();
  auto pOut1 = std::make_unique<reco::CaloJetCollection>();

    double scaleJPT = 1.;
    std::vector<reco::JPTJet> theJPTJets;
   if (jetsTrackJets.isValid()) {
     if(jetsTrackJets->size() > 0 ) {
      for (unsigned ijet = 0; ijet < jetsTrackJets->size(); ++ijet) {
         const reco::TrackJet* jet = &(*(jetsTrackJets->refAt(ijet)));
         int icalo = -1;
         if(addjets_h.isValid()) {
          for (unsigned i = 0; i < addjets_h->size(); ++i) {
           const reco::CaloJet* oldjet = &(*(addjets_h->refAt(i)));
           double deta = fabs(jet->eta()-oldjet->eta());
           double dphi = fabs(jet->phi()-oldjet->phi()); 
           if(dphi>4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
           double dr = sqrt(dphi*dphi+deta*deta);
           if(dr<=0.4) {icalo=i;}  
          } // Calojets
         }
     if(icalo < 0) continue;
       const reco::CaloJet* mycalo = &(*(addjets_h->refAt(icalo)));
  //     std::cout<<" Additional CaloJet "<<mycalo->pt()<<" "<<mycalo->eta()<<
  //                                   " "<<mycalo->phi()<<std::endl;
        std::vector<edm::Ptr<reco::Track> > tracksinjet = jet->tracks();
        reco::TrackRefVector tracksincalo;
        reco::TrackRefVector tracksinvert;
        for(std::vector<edm::Ptr<reco::Track> >::iterator itrack = tracksinjet.begin();
                       itrack != tracksinjet.end(); itrack++) {
       for ( std::vector<reco::TrackExtrapolation>::const_iterator xtrpBegin = iExtrapolations->begin(),
                  xtrpEnd = iExtrapolations->end(), ixtrp = xtrpBegin;
                                 ixtrp != xtrpEnd; ++ixtrp )
            {
                 if ( ixtrp->positions().empty() ) continue;
                double mydphi =fabs(ixtrp->track()->phi()-(**itrack).phi());
                if(mydphi>4.*atan(1.)) mydphi = 8.*atan(1)-mydphi;
                  if(fabs(ixtrp->track()->pt()-(**itrack).pt()) > 0.001 ||
                     fabs(ixtrp->track()->eta()-(**itrack).eta()) > 0.001 || mydphi > 0.001) continue;
                       tracksinvert.push_back(ixtrp->track());
                reco::TrackBase::Point const & point = ixtrp->positions().at(0);

                double dr = reco::deltaR<double>( jet->eta(), jet->phi(), point.eta(), point.phi() );
                if(dr <= 0.4) {
                       /*std::cout<<" TrackINcalo "<<std::endl;*/
                         tracksincalo.push_back(ixtrp->track());
                }
            } // Track extrapolations
        } // tracks

  reco::TrackJet corrected = *jet;
  math::XYZTLorentzVector p4;
  jpt::MatchedTracks pions;
  jpt::MatchedTracks muons;
  jpt::MatchedTracks elecs;

  scaleJPT = mJPTalgo->correction( corrected, *mycalo, iEvent, iSetup, 
                                 tracksinvert, tracksincalo, p4, pions, muons, elecs);
 // std::cout<<" JetPlusTrackProducer::AddSeedJet "<< (*mycalo).pt()<<" "<< (*mycalo).eta()<<" "<<
 //        (*mycalo).phi()<<" "<<(*mycalo).jetArea()<<" Corr "<<
 //       p4.pt()<<" "<<p4.eta()<<" "<<p4.phi()<<std::endl;
  if(p4.pt() > ptCUT) {
    reco::JPTJet::Specific jptspe;
    jptspe.pionsInVertexInCalo = pions.inVertexInCalo_;
    jptspe.pionsInVertexOutCalo = pions.inVertexOutOfCalo_;
    jptspe.pionsOutVertexInCalo = pions.outOfVertexInCalo_;
    jptspe.muonsInVertexInCalo = muons.inVertexInCalo_;
    jptspe.muonsInVertexOutCalo = muons.inVertexOutOfCalo_;
    jptspe.muonsOutVertexInCalo = muons.outOfVertexInCalo_;
    jptspe.elecsInVertexInCalo = elecs.inVertexInCalo_;
    jptspe.elecsInVertexOutCalo = elecs.inVertexOutOfCalo_;
    jptspe.elecsOutVertexInCalo = elecs.outOfVertexInCalo_;
    reco::CaloJetRef myjet(pOut1RefProd, idxCaloJet++);
    jptspe.theCaloJetRef = edm::RefToBase<reco::Jet>(myjet);
    jptspe.mZSPCor = 1.;
    reco::JPTJet fJet(p4, jet->primaryVertex()->position(), jptspe, mycalo->getJetConstituents());
    pOut->push_back(fJet);
    pOut1->push_back(*mycalo);
    theJPTJets.push_back(fJet);
  }
     } // trackjets
   } // jets
  } // There is trackjet collection

  for (unsigned i = 0; i < jets_h->size(); ++i) {

   const reco::CaloJet* oldjet = &(*(jets_h->refAt(i)));
   reco::CaloJet corrected = *oldjet; 

// ZSP corrections    

   double factorZSP = 1.;
   if(useZSP) factorZSP = mZSPalgo->correction(corrected, iEvent, iSetup);
   corrected.scaleEnergy (factorZSP);

// JPT corrections 
   scaleJPT = 1.; 

   math::XYZTLorentzVector p4;

  jpt::MatchedTracks pions;
  jpt::MatchedTracks muons;
  jpt::MatchedTracks elecs;
  bool ok=false;

   if ( !vectorial_ ) {
            
   scaleJPT = mJPTalgo->correction ( corrected, *oldjet, iEvent, iSetup, pions, muons, elecs,ok );
   p4 = math::XYZTLorentzVector( corrected.px()*scaleJPT, 
                                 corrected.py()*scaleJPT,
                                 corrected.pz()*scaleJPT, 
                                 corrected.energy()*scaleJPT );
    // std::cout<<" JetPlusTrackProducer "<< (*oldjet).pt()<<" "<< (*oldjet).eta()<<" "<< (*oldjet).phi()<<
    //      " "<<scaleJPT<<" "<<(*oldjet).jetArea()<<std::endl;
   } else {
     scaleJPT = mJPTalgo->correction( corrected, *oldjet, iEvent, iSetup, p4, pions, muons, elecs,ok );
    // std::cout<<" JetPlusTrackProducer "<< (*oldjet).pt()<<" "<< (*oldjet).eta()<<" "<< 
    //     (*oldjet).phi()<<" "<<(*oldjet).jetArea()<<" Corr "<<
    //    p4.pt()<<" "<<p4.eta()<<" "<<p4.phi()<<std::endl; 
  }         

   
  reco::JPTJet::Specific specific;

  if(ok) {
    specific.pionsInVertexInCalo = pions.inVertexInCalo_;
    specific.pionsInVertexOutCalo = pions.inVertexOutOfCalo_;
    specific.pionsOutVertexInCalo = pions.outOfVertexInCalo_;
    specific.muonsInVertexInCalo = muons.inVertexInCalo_;
    specific.muonsInVertexOutCalo = muons.inVertexOutOfCalo_;
    specific.muonsOutVertexInCalo = muons.outOfVertexInCalo_;
    specific.elecsInVertexInCalo = elecs.inVertexInCalo_;
    specific.elecsInVertexOutCalo = elecs.inVertexOutOfCalo_;
    specific.elecsOutVertexInCalo = elecs.outOfVertexInCalo_;
  }

// Fill JPT Specific
    edm::RefToBase<reco::Jet> myjet = (edm::RefToBase<reco::Jet>)jets_h->refAt(i);
    specific.theCaloJetRef = myjet;
    specific.mZSPCor = factorZSP;
    specific.mResponseOfChargedWithEff = (float)mJPTalgo->getResponseOfChargedWithEff();
    specific.mResponseOfChargedWithoutEff = (float)mJPTalgo->getResponseOfChargedWithoutEff();
    specific.mSumPtOfChargedWithEff = (float)mJPTalgo->getSumPtWithEff();
    specific.mSumPtOfChargedWithoutEff = (float)mJPTalgo->getSumPtWithoutEff();
    specific.mSumEnergyOfChargedWithEff = (float)mJPTalgo->getSumEnergyWithEff();
    specific.mSumEnergyOfChargedWithoutEff = (float)mJPTalgo->getSumEnergyWithoutEff();
    specific.mChargedHadronEnergy = (float)mJPTalgo->getSumEnergyWithoutEff();

// Fill Charged Jet shape parameters
   double deR2Tr = 0.;
   double deEta2Tr = 0.;
   double dePhi2Tr = 0.;
   double Zch = 0.;
   double Pout2 = 0.;
   double Pout = 0.;
   double denominator_tracks = 0.;
   int ntracks = 0;

   for( reco::TrackRefVector::const_iterator it = pions.inVertexInCalo_.begin(); it != pions.inVertexInCalo_.end(); it++) { 
    double deR = deltaR((*it)->eta(), (*it)->phi(), p4.eta(), p4.phi());
    double deEta = (*it)->eta() - p4.eta();
    double dePhi = deltaPhi((*it)->phi(), p4.phi());
     if((**it).ptError()/(**it).pt() < 0.1) {
       deR2Tr   =  deR2Tr + deR*deR*(*it)->pt();
       deEta2Tr = deEta2Tr + deEta*deEta*(*it)->pt();
       dePhi2Tr = dePhi2Tr + dePhi*dePhi*(*it)->pt();
       denominator_tracks = denominator_tracks + (*it)->pt();
       Zch    =  Zch + (*it)->pt();
       
       Pout2 = Pout2 + (**it).p()*(**it).p() - (Zch*p4.P())*(Zch*p4.P());
       ntracks++;
     }
   }

   for( reco::TrackRefVector::const_iterator it = muons.inVertexInCalo_.begin(); it != muons.inVertexInCalo_.end(); it++) {
    double deR = deltaR((*it)->eta(), (*it)->phi(), p4.eta(), p4.phi());
    double deEta = (*it)->eta() - p4.eta();
    double dePhi = deltaPhi((*it)->phi(), p4.phi());
     if((**it).ptError()/(**it).pt() < 0.1) {
       deR2Tr   =  deR2Tr + deR*deR*(*it)->pt();
       deEta2Tr = deEta2Tr + deEta*deEta*(*it)->pt();
       dePhi2Tr = dePhi2Tr + dePhi*dePhi*(*it)->pt();
       denominator_tracks = denominator_tracks + (*it)->pt();
       Zch    = Zch + (*it)->pt();
       
       Pout2 = Pout2 + (**it).p()*(**it).p() - (Zch*p4.P())*(Zch*p4.P());
       ntracks++;
     }
   }
   for( reco::TrackRefVector::const_iterator it = elecs.inVertexInCalo_.begin(); it != elecs.inVertexInCalo_.end(); it++) {
    double deR = deltaR((*it)->eta(), (*it)->phi(), p4.eta(), p4.phi());
    double deEta = (*it)->eta() - p4.eta();
    double dePhi = deltaPhi((*it)->phi(), p4.phi());
     if((**it).ptError()/(**it).pt() < 0.1) {
       deR2Tr   =  deR2Tr + deR*deR*(*it)->pt();
       deEta2Tr = deEta2Tr + deEta*deEta*(*it)->pt();
       dePhi2Tr = dePhi2Tr + dePhi*dePhi*(*it)->pt();
       denominator_tracks = denominator_tracks + (*it)->pt();
       Zch    = Zch + (*it)->pt();
       
       Pout2 = Pout2 + (**it).p()*(**it).p() - (Zch*p4.P())*(Zch*p4.P());
       ntracks++;
     }
   }
   for( reco::TrackRefVector::const_iterator it = pions.inVertexOutOfCalo_.begin(); it != pions.inVertexOutOfCalo_.end(); it++) { 
     Zch    =  Zch + (*it)->pt();
   }
   for( reco::TrackRefVector::const_iterator it = muons.inVertexOutOfCalo_.begin(); it != muons.inVertexOutOfCalo_.end(); it++) { 
     Zch    =  Zch + (*it)->pt();
   }
   for( reco::TrackRefVector::const_iterator it = elecs.inVertexOutOfCalo_.begin(); it != elecs.inVertexOutOfCalo_.end(); it++) { 
     Zch    =  Zch + (*it)->pt();
   }

     if(mJPTalgo->getSumPtForBeta()> 0.) Zch = Zch/mJPTalgo->getSumPtForBeta();

//     std::cout<<" Zch "<< Zch<<" "<<mJPTalgo->getSumPtForBeta()<<std::endl;

        if(ntracks > 0) {
          Pout   = sqrt(fabs(Pout2))/ntracks;          
        }
          if (denominator_tracks!=0){
            deR2Tr  = deR2Tr/denominator_tracks;
            deEta2Tr= deEta2Tr/denominator_tracks;
            dePhi2Tr= dePhi2Tr/denominator_tracks;
          }
   
      specific.R2momtr = deR2Tr;
      specific.Eta2momtr = deEta2Tr;
      specific.Phi2momtr = dePhi2Tr;
      specific.Pout = Pout;
      specific.Zch = Zch;


//       std::cout<<" Moments for charged component "<<deR2_Tr<<" "<<deEta2_Tr<<" "<<dePhi2_Tr<<std::endl;


// Create JPT jet

   reco::Particle::Point vertex_=reco::Jet::Point(0,0,0);
   
// If we add primary vertex
   edm::Handle<reco::VertexCollection> pvCollection;
   iEvent.getByToken(input_vertex_token_, pvCollection);
   if ( pvCollection.isValid() && !pvCollection->empty() ) vertex_=pvCollection->begin()->position();

   reco::JPTJet fJet(p4, vertex_, specific, corrected.getJetConstituents()); 

   //   fJet.printJet();

// Output module
    if(fJet.pt()>ptCUT) pOut->push_back(fJet); 
  }

  std::sort(pOut->begin(),pOut->end(),sort_by_pt);  
   //std::cout<<"Size of the additional jets "<<pOut1->size()<<std::endl;
  iEvent.put(std::move(pOut1));
  iEvent.put(std::move(pOut));

}

//define this as a plug-in
//DEFINE_FWK_MODULE(JetPlusTrackProducer);
