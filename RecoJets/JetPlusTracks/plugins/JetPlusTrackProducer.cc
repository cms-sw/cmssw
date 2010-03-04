// -*- C++ -*-
//
// Package:    JetPlusTracks
// Class:      JetPlusTrackProducer
// 
/**\class JetPlusTrackProducer JetPlusTrackProducer.cc RecoJets/JetPlusTracks/src/JetPlusTrackProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Olga Kodolova,40 R-A12,+41227671273,
//         Created:  Fri Feb 19 10:14:02 CET 2010
// $Id$
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
   mJPTalgo  = new JetPlusTrackCorrector(iConfig);
   mZSPalgo  = new ZSPJPTJetCorrector(iConfig);
 
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
   
   if ( !vectorial_ ) {
            
   scaleJPT = mJPTalgo->correction ( corrected, *oldjet, iEvent, iSetup );
   p4 = math::XYZTLorentzVector( corrected.px()*scaleJPT, 
                                 corrected.py()*scaleJPT,
                                 corrected.pz()*scaleJPT, 
                                 corrected.energy()*scaleJPT );
   } else {
   scaleJPT = mJPTalgo->correction( corrected, *oldjet, iEvent, iSetup, p4 );
  }         

// Construct JPTJet constituent
  jpt::MatchedTracks pions;
  jpt::MatchedTracks muons;
  jpt::MatchedTracks elecs;
  
  bool ok = mJPTalgo->matchTracks( *oldjet, 
		                    iEvent, 
		                    iSetup,
		                     pions, 
		                     muons, 
		                     elecs );

   
  reco::JPTJet::Specific specific;

  if(ok) {
    specific.pions_inVertexInCalo = pions.inVertexInCalo_;
    specific.pions_inVertexOutCalo = pions.inVertexOutOfCalo_;
    specific.pions_OutVertexInCalo = pions.outOfVertexInCalo_;
    specific.muons_inVertexInCalo = muons.inVertexInCalo_;
    specific.muons_inVertexOutCalo = muons.inVertexOutOfCalo_;
    specific.muons_OutVertexInCalo = muons.outOfVertexInCalo_;
    specific.elecs_inVertexInCalo = elecs.inVertexInCalo_;
    specific.elecs_inVertexOutCalo = elecs.inVertexOutOfCalo_;
    specific.elecs_OutVertexInCalo = elecs.outOfVertexInCalo_;
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
   double deR2_Tr = 0.;
   double deEta2_Tr = 0.;
   double dePhi2_Tr = 0.;
   double Z_ch = 0.;
   double P_out_2 = 0.;
   double P_out = 0.;
   double denominator_tracks = 0.;

   for( reco::TrackRefVector::const_iterator it = pions.inVertexInCalo_.begin(); it != pions.inVertexInCalo_.end(); it++) { 
    double deR = deltaR((*it)->eta(), (*it)->phi(), p4.eta(), p4.phi());
    double deEta = (*it)->eta() - p4.eta();
    double dePhi = deltaPhi((*it)->phi(), p4.phi());
     if((**it).ptError()/(**it).pt() < 0.1) {
       deR2_Tr   =  deR2_Tr + deR*deR*(*it)->pt();
       deEta2_Tr = deEta2_Tr + deEta*deEta*(*it)->pt();
       dePhi2_Tr = dePhi2_Tr + dePhi*dePhi*(*it)->pt();
       denominator_tracks = denominator_tracks + (*it)->pt();
       Z_ch    = 
       ((*it)->px()*p4.Px()+(*it)->py()*p4.Py()+(*it)->pz()*p4.Pz())/(p4.P()*p4.P());
       P_out_2 = (**it).p()*(**it).p() - (Z_ch*p4.P())*(Z_ch*p4.P());
       P_out   = sqrt(fabs(P_out_2)); 
     }
   }
   for( reco::TrackRefVector::const_iterator it = muons.inVertexInCalo_.begin(); it != muons.inVertexInCalo_.end(); it++) {
    double deR = deltaR((*it)->eta(), (*it)->phi(), p4.eta(), p4.phi());
    double deEta = (*it)->eta() - p4.eta();
    double dePhi = deltaPhi((*it)->phi(), p4.phi());
     if((**it).ptError()/(**it).pt() < 0.1) {
       deR2_Tr   =  deR2_Tr + deR*deR*(*it)->pt();
       deEta2_Tr = deEta2_Tr + deEta*deEta*(*it)->pt();
       dePhi2_Tr = dePhi2_Tr + dePhi*dePhi*(*it)->pt();
       denominator_tracks = denominator_tracks + (*it)->pt();
       Z_ch    =
       ((*it)->px()*p4.Px()+(*it)->py()*p4.Py()+(*it)->pz()*p4.Pz())/(p4.P()*p4.P());
       P_out_2 = (**it).p()*(**it).p() - (Z_ch*p4.P())*(Z_ch*p4.P());
       P_out   = sqrt(fabs(P_out_2));
     }
   }
   for( reco::TrackRefVector::const_iterator it = elecs.inVertexInCalo_.begin(); it != elecs.inVertexInCalo_.end(); it++) {
    double deR = deltaR((*it)->eta(), (*it)->phi(), p4.eta(), p4.phi());
    double deEta = (*it)->eta() - p4.eta();
    double dePhi = deltaPhi((*it)->phi(), p4.phi());
     if((**it).ptError()/(**it).pt() < 0.1) {
       deR2_Tr   =  deR2_Tr + deR*deR*(*it)->pt();
       deEta2_Tr = deEta2_Tr + deEta*deEta*(*it)->pt();
       dePhi2_Tr = dePhi2_Tr + dePhi*dePhi*(*it)->pt();
       denominator_tracks = denominator_tracks + (*it)->pt();
       Z_ch    =
       ((*it)->px()*p4.Px()+(*it)->py()*p4.Py()+(*it)->pz()*p4.Pz())/(p4.P()*p4.P());
       P_out_2 = (**it).p()*(**it).p() - (Z_ch*p4.P())*(Z_ch*p4.P());
       P_out   = sqrt(fabs(P_out_2));
     }
   }

          if (denominator_tracks!=0){
            deR2_Tr  = deR2_Tr/denominator_tracks;
            deEta2_Tr= deEta2_Tr/denominator_tracks;
            dePhi2_Tr= dePhi2_Tr/denominator_tracks;
          }
   
      specific.R2mom_tr = deR2_Tr;
      specific.Eta2mom_tr = deEta2_Tr;
      specific.Phi2mom_tr = dePhi2_Tr;
      specific.P_out = P_out;
      specific.Z_ch = Z_ch;


//       std::cout<<" Moments for charged component "<<deR2_Tr<<" "<<deEta2_Tr<<" "<<dePhi2_Tr<<std::endl;


// Create JPT jet

   reco::Particle::Point vertex_=reco::Jet::Point(0,0,0);
   
// If we add primary vertex
   edm::Handle<reco::VertexCollection> pvCollection;
   iEvent.getByLabel(srcPVs_,pvCollection);
   if ( pvCollection.isValid() && pvCollection->size()>0 ) vertex_=pvCollection->begin()->position();

   reco::JPTJet fJet(p4, vertex_, specific, corrected.getJetConstituents()); 

  // fJet.printJet();

// Output module
    pOut->push_back(fJet); 
          
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
