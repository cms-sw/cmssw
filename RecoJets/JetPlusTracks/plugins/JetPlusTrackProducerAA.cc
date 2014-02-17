// -*- C++ -*-
//
// Package:    JetPlusTrack
// Class:      JetPlusTrack
// 
/**\class JetPlusTrackProducer JetPlusTrackProducer.cc JetPlusTrackProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Olga Kodolova,40 R-A12,+41227671273,
//         Created:  Fri Feb 19 10:14:02 CET 2010
// $Id: JetPlusTrackProducerAA.cc,v 1.10 2012/10/18 08:46:42 eulisse Exp $
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

#include "RecoJets/JetPlusTracks/plugins/JetPlusTrackProducerAA.h"
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

//=>
#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationXtrpCalo.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
//=>

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
JetPlusTrackProducerAA::JetPlusTrackProducerAA(const edm::ParameterSet& iConfig)
{
   //register your products
   src = iConfig.getParameter<edm::InputTag>("src");
   alias = iConfig.getUntrackedParameter<string>("alias");
   mTracks = iConfig.getParameter<edm::InputTag> ("tracks");
   srcPVs_ = iConfig.getParameter<edm::InputTag>("srcPVs"); 
   vectorial_ = iConfig.getParameter<bool>("VectorialCorrection");
   useZSP = iConfig.getParameter<bool>("UseZSP");
   std::string tq = iConfig.getParameter<std::string>("TrackQuality");
   trackQuality_ = reco::TrackBase::qualityByName(tq);
   mConeSize = iConfig.getParameter<double> ("coneSize");
//=>
   mExtrapolations = iConfig.getParameter<edm::InputTag> ("extrapolations");
//=>
   mJPTalgo  = new JetPlusTrackCorrector(iConfig);
   if(useZSP) mZSPalgo  = new ZSPJPTJetCorrector(iConfig);

   produces<reco::JPTJetCollection>().setBranchAlias(alias); 
     
}


JetPlusTrackProducerAA::~JetPlusTrackProducerAA()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
JetPlusTrackProducerAA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm; 


//  std::cout<<" RecoJets::JetPlusTrackProducerAA::produce "<<std::endl;

// get stuff from Event
  edm::Handle <edm::View <reco::CaloJet> > jets_h;
  iEvent.getByLabel (src, jets_h);
  
  edm::Handle <reco::TrackCollection> tracks_h;
  iEvent.getByLabel (mTracks, tracks_h);
  
  std::vector <reco::TrackRef> fTracks;
  fTracks.reserve (tracks_h->size());
  for (unsigned i = 0; i < tracks_h->size(); ++i) {
             fTracks.push_back (reco::TrackRef (tracks_h, i));
  } 

//=>
  edm::Handle <std::vector<reco::TrackExtrapolation> > extrapolations_h;
  iEvent.getByLabel (mExtrapolations, extrapolations_h);

//  std::cout<<"JetPlusTrackProducerAA::produce, extrapolations_h="<<extrapolations_h->size()<<std::endl;  
//=>

  std::auto_ptr<reco::JPTJetCollection> pOut(new reco::JPTJetCollection());
  
  reco::JPTJetCollection tmpColl;

  for (unsigned i = 0; i < jets_h->size(); ++i) {

   const reco::CaloJet* oldjet = &(*(jets_h->refAt(i)));
   
   reco::CaloJet corrected = *oldjet; 
   
// ZSP corrections    

   double factorZSP = 1.;
   if(useZSP) factorZSP = mZSPalgo->correction(corrected, iEvent, iSetup);

//   std::cout << " UseZSP = "<<useZSP<<std::endl;


   corrected.scaleEnergy (factorZSP);

// JPT corrections 

   double scaleJPT = 1.; 

   math::XYZTLorentzVector p4;

// Construct JPTJet constituent
  jpt::MatchedTracks pions;
  jpt::MatchedTracks muons;
  jpt::MatchedTracks elecs;
  bool ok=false;

   if ( !vectorial_ ) {
            
     scaleJPT = mJPTalgo->correction ( corrected, *oldjet, iEvent, iSetup, pions, muons, elecs, ok );
   p4 = math::XYZTLorentzVector( corrected.px()*scaleJPT, 
                                 corrected.py()*scaleJPT,
                                 corrected.pz()*scaleJPT, 
                                 corrected.energy()*scaleJPT );
   } else {
     scaleJPT = mJPTalgo->correction( corrected, *oldjet, iEvent, iSetup, p4, pions, muons, elecs, ok );
  }         

  
   
  reco::JPTJet::Specific specific;

  if(ok) {
//    std::cout<<" Size of Pion in-in "<<pions.inVertexInCalo_.size()<<" in-out "<<pions.inVertexOutOfCalo_.size()
//             <<" out-in "<<pions.outOfVertexInCalo_.size()<<" Oldjet "<<oldjet->et()<<" factorZSP "<<factorZSP
//             <<"  "<<corrected.et()<<" scaleJPT "<<scaleJPT<<" after JPT "<<p4.pt()<<std::endl;
    

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
       Zch    = Zch + (*it)->pt();
       
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

 //    std::cout<<" Zch "<< Zch<<" "<<mJPTalgo->getSumPtForBeta()<<std::endl;

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

// Create JPT jet

   reco::Particle::Point vertex_=reco::Jet::Point(0,0,0);
   
// If we add primary vertex
   edm::Handle<reco::VertexCollection> pvCollection;
   iEvent.getByLabel(srcPVs_,pvCollection);
   if ( pvCollection.isValid() && pvCollection->size()>0 ) vertex_=pvCollection->begin()->position();
 
   reco::JPTJet fJet(p4, vertex_, specific, corrected.getJetConstituents()); 
  // fJet.printJet();

// Temporarily collection before correction for background
 
   tmpColl.push_back(fJet);     

  }

//=======================================================================================================>
// Correction for background

  reco::TrackRefVector trBgOutOfCalo;
  reco::TrackRefVector trBgOutOfVertex = calculateBGtracksJet(tmpColl,fTracks,extrapolations_h,trBgOutOfCalo);

//===> Area without Jets 
    std::map<reco::JPTJetCollection::iterator, double> AreaNonJet;
    
    for(reco::JPTJetCollection::iterator ij1=tmpColl.begin(); ij1!=tmpColl.end(); ij1++) 
    {
      int nj1 = 1;
      for(reco::JPTJetCollection::iterator ij2=tmpColl.begin(); ij2!=tmpColl.end(); ij2++) 
      {
       if(ij2 == ij1) continue;
       if(fabs((*ij1).eta() - (*ij2).eta()) > 0.5 ) continue;
       nj1++;

      }

      AreaNonJet[ij1] = 4*M_PI*mConeSize - nj1*4*mConeSize*mConeSize;

//      std::cout<<"+++AreaNonJet[ij1]="<<AreaNonJet[ij1]<<" nj1="<<nj1<<std::endl;
    }

//===>

//  std::cout<<" The size of BG tracks: trBgOutOfVertex= "<<trBgOutOfVertex.size()
//           <<" trBgOutOfCalo= "<<trBgOutOfCalo.size()<<std::endl;
//
//  std::cout<<" The size of JPT jet collection "<<tmpColl.size()<<std::endl;
  
  for(reco::JPTJetCollection::iterator ij=tmpColl.begin(); ij!=tmpColl.end(); ij++)
  {    
// Correct JPTjet for background tracks

   const reco::TrackRefVector pioninin  = (*ij).getPionsInVertexInCalo();
   const reco::TrackRefVector pioninout = (*ij).getPionsInVertexOutCalo();
   
   double ja = (AreaNonJet.find(ij))->second;

//    std::cout<<"+++ ja="<<ja<<" pioninout="<<pioninout.size()<<std::endl;

   double factorPU = mJPTalgo->correctAA(*ij,trBgOutOfVertex,mConeSize,pioninin,pioninout,ja,trBgOutOfCalo);

   (*ij).scaleEnergy (factorPU);
   
//   std::cout<<" FactorPU "<<factorPU<<std::endl;
   
// Output module
    pOut->push_back(*ij);
    
//    std::cout<<" New JPT energy "<<(*ij).et()<<" "<<(*ij).pt()<<" "<<(*ij).eta()<<" "<<(*ij).phi()<<std::endl;
    
  }
  
   iEvent.put(pOut);
   
}
// -----------------------------------------------
// ------------ calculateBGtracksJet  ------------
// ------------ Tracks not included in jets ------
// -----------------------------------------------
reco::TrackRefVector  JetPlusTrackProducerAA::calculateBGtracksJet(reco::JPTJetCollection& fJets, std::vector <reco::TrackRef>& fTracks,
                                                                edm::Handle <std::vector<reco::TrackExtrapolation> > & extrapolations_h, 
                                                                                                   reco::TrackRefVector& trBgOutOfCalo){ 

  
  reco::TrackRefVector trBgOutOfVertex;
  
  for (unsigned t = 0; t < fTracks.size(); ++t) {

     int track_bg = 0;
     
    // if(!(*fTracks[t]).quality(trackQuality_))
    // {
      // cout<<"BG, BAD trackQuality, ptBgV="<<fTracks[t]->pt()<<" etaBgV = "<<fTracks[t]->eta()<<" phiBgV = "<<fTracks[t]->phi()<<endl;
      // continue;
    // }

    const reco::Track* track = &*(fTracks[t]);
    double trackEta = track->eta();
    double trackPhi = track->phi();

  //  std::cout<<"++++++++++++++++>  track="<<t<<" trackEta="<<trackEta<<" trackPhi="<<trackPhi
  //           <<" coneSize="<<mConeSize<<std::endl;
   
   //loop on jets
    for (unsigned j = 0; j < fJets.size(); ++j) {

     const reco::Jet* jet = &(fJets[j]);
     double jetEta = jet->eta();
     double jetPhi = jet->phi();

    //  std::cout<<"-jet="<<j<<" jetEt ="<<jet->pt()
    //  <<" jetE="<<jet->energy()<<" jetEta="<<jetEta<<" jetPhi="<<jetPhi<<std::endl;

      if(fabs(jetEta - trackEta) < mConeSize) {
       double dphiTrackJet = fabs(trackPhi - jetPhi);
       if(dphiTrackJet > M_PI) dphiTrackJet = 2*M_PI - dphiTrackJet;

       if(dphiTrackJet < mConeSize) 
        {
         track_bg = 1;
     //    std::cout<<"===>>>> Track inside jet at vertex, track_bg="<< track_bg <<" track="<<t<<" jet="<<j
      //            <<" trackEta="<<trackEta<<" trackPhi="<<trackPhi
      //            <<" jetEta="<<jetEta<<" jetPhi="<<jetPhi<<std::endl;
        }
      }      
    } //jets

    if( track_bg == 0 ) 
     {
       trBgOutOfVertex.push_back (fTracks[t]);
    
//       std::cout<<"------Track outside jet at vertex, track_bg="<< track_bg<<" track="<<t
//               <<" trackEta="<<trackEta<<" trackPhi="<<trackPhi <<std::endl;    
     }

  } //tracks    

//=====> Propagate BG tracks to calo 
    int nValid = 0;
    for ( std::vector<reco::TrackExtrapolation>::const_iterator xtrpBegin = extrapolations_h->begin(),
          xtrpEnd = extrapolations_h->end(), ixtrp = xtrpBegin;
          ixtrp != xtrpEnd; ++ixtrp ) {

//    std::cout<<"JetPlusTrackProducerAA::calculateBGtracksJet: initial track pt= "<<ixtrp->track()->pt()
//             <<" eta= "<<ixtrp->track()->eta()<<" phi="<<ixtrp->track()->phi()
//             <<" Valid? "<<ixtrp->isValid().at(0)<<std::endl;

          //if( ixtrp->isValid().at(0) == 0 ) continue;
          //in DF change in 4.2, all entries are valid.
          nValid++;

          reco::TrackRefVector::iterator it = find(trBgOutOfVertex.begin(),trBgOutOfVertex.end(),(*ixtrp).track() );

          if ( it != trBgOutOfVertex.end() ){
             trBgOutOfCalo.push_back (*it);

//          std::cout<<"+++trBgOutOfCalo, pt= "<<ixtrp->track()->pt()<<" eta= "<<ixtrp->track()->eta()<<" phi="<<ixtrp->track()->phi()
//                   <<" Valid? "<<ixtrp->isValid().at(0)<<std::endl;
          }

    }

//     std::cout<<"calculateBGtracksJet, trBgOutOfCalo="<<trBgOutOfCalo.size()
//              <<" trBgOutOfVertex="<<trBgOutOfVertex.size()<<" nValid="<<nValid<<endl;
//=====>

  return trBgOutOfVertex;
}
// ------------ method called once each job just before starting event loop  ------------
void 
JetPlusTrackProducerAA::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
JetPlusTrackProducerAA::endJob() {
}

//define this as a plug-in
//DEFINE_FWK_MODULE(JetPlusTrackProducerAA);
