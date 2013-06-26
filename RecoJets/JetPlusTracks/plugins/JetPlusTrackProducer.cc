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
// $Id: JetPlusTrackProducer.cc,v 1.9 2013/04/30 09:02:46 kodolova Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetPlusTracks/plugins/JetPlusTrackProducer.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
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
   alias = iConfig.getUntrackedParameter<string>("alias");
   srcPVs_ = iConfig.getParameter<edm::InputTag>("srcPVs");
   vectorial_ = iConfig.getParameter<bool>("VectorialCorrection");
   useZSP = iConfig.getParameter<bool>("UseZSP");
   ptCUT = iConfig.getParameter<double>("ptCUT");
   mJPTalgo  = new JetPlusTrackCorrector(iConfig);
   if(useZSP) mZSPalgo  = new ZSPJPTJetCorrector(iConfig);
 
   produces<reco::JPTJetCollection>().setBranchAlias(alias); 
   
}


JetPlusTrackProducer::~JetPlusTrackProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
JetPlusTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm; 


//  std::cout<<" RecoJets::JetPlusTrackProducer::produce "<<std::endl;


// get stuff from Event
  edm::Handle <edm::View <reco::CaloJet> > jets_h;
  iEvent.getByLabel (src, jets_h);

//  std::auto_ptr<reco::CaloJetCollection> pOut(new reco::CaloJetCollection());
  std::auto_ptr<reco::JPTJetCollection> pOut(new reco::JPTJetCollection());

  for (unsigned i = 0; i < jets_h->size(); ++i) {

   const reco::CaloJet* oldjet = &(*(jets_h->refAt(i)));
   
   reco::CaloJet corrected = *oldjet; 
   
// ZSP corrections    

   double factorZSP = 1.;
   if(useZSP) factorZSP = mZSPalgo->correction(corrected, iEvent, iSetup);

   corrected.scaleEnergy (factorZSP);

// JPT corrections 

   double scaleJPT = 1.; 

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
   } else {
     scaleJPT = mJPTalgo->correction( corrected, *oldjet, iEvent, iSetup, p4, pions, muons, elecs,ok );
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
   iEvent.getByLabel(srcPVs_,pvCollection);
   if ( pvCollection.isValid() && pvCollection->size()>0 ) vertex_=pvCollection->begin()->position();

   reco::JPTJet fJet(p4, vertex_, specific, corrected.getJetConstituents()); 

   //   fJet.printJet();

// Output module
    if(fJet.pt()>ptCUT) pOut->push_back(fJet); 
          
  }
  
  iEvent.put(pOut);
   
}

// ------------ method called once each job just before starting event loop  ------------
void 
JetPlusTrackProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
JetPlusTrackProducer::endJob() {
}

//define this as a plug-in
//DEFINE_FWK_MODULE(JetPlusTrackProducer);
