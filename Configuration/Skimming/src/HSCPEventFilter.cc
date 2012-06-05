// -*- C++ -*-
//
// Package:    HSCPEventFilter
// Class:      HSCPEventFilter
// 
/**\class HSCPEventFilter HSCPEventFilter.cc HSCPEventFilter/HSCPEventFilter/src/HSCPEventFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jie Chen
//         Created:  Thu Apr 29 16:32:10 CDT 2010
// $Id: HSCPEventFilter.cc,v 1.4 2011/05/10 22:07:21 jiechen Exp $
// modified by Loic Quertenmont Apr 24 2012
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class HSCPEventFilter : public edm::EDFilter {
   public:
      explicit HSCPEventFilter(const edm::ParameterSet&);
      ~HSCPEventFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      bool filterFlag;
      edm::InputTag input_muon_collection, input_track_collection,input_dedx_collection;
      int ndedxHits;
      double dedxMin, dedxMaxLeft, trkPtMin,SAMuPtMin,etaMin,etaMax,chi2nMax,dxyMax,dzMax, Mu1PtMin, Mu2PtMin;


int NSAMu, N2Mu, NTk;
};

HSCPEventFilter::HSCPEventFilter(const edm::ParameterSet& iConfig)
{
     filterFlag = iConfig.getParameter< bool >("filter");
     input_muon_collection = iConfig.getParameter< edm::InputTag >("inputMuonCollection");    
     input_track_collection = iConfig.getParameter< edm::InputTag >("inputTrackCollection");    
     input_dedx_collection =  iConfig.getParameter< edm::InputTag >("inputDedxCollection");
     dedxMin = iConfig.getParameter< double >("dedxMin");
     dedxMaxLeft = iConfig.getParameter< double >("dedxMaxLeft");
     trkPtMin = iConfig.getParameter< double >("trkPtMin");
     etaMin =  iConfig.getParameter< double >("etaMin");
     etaMax =  iConfig.getParameter< double >("etaMax");
     ndedxHits = iConfig.getParameter< int >("ndedxHits");
     chi2nMax = iConfig.getParameter< double >("chi2nMax");
     dxyMax = iConfig.getParameter< double >("dxyMax");
     dzMax = iConfig.getParameter< double >("dzMax");
     SAMuPtMin = iConfig.getParameter< double >("SAMuPtMin");
     Mu1PtMin = iConfig.getParameter< double >("Mu1PtMin");
     Mu2PtMin = iConfig.getParameter< double >("Mu2PtMin");
}


HSCPEventFilter::~HSCPEventFilter(){}
void HSCPEventFilter::beginJob(){}
void HSCPEventFilter::endJob() {}

bool HSCPEventFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){
   using namespace edm;
   using namespace reco;

   if(!filterFlag) return true;

   edm::Handle<reco::VertexCollection> recoVertexHandle;
   iEvent.getByLabel("offlinePrimaryVertices", recoVertexHandle);
   reco::VertexCollection recoVertex = *recoVertexHandle;
   if(recoVertex.size()<1) return false;

   //CHECK IF WE HAVE A DIMUON EVENT AND/OR A HIGH PT SA MUON
   double mu1pt=0, mu2pt=0;
   double SAmupt=0;
   using reco::MuonCollection;
   Handle<MuonCollection> muTracks;
   iEvent.getByLabel(input_muon_collection,muTracks);
   const reco::MuonCollection muonC = *(muTracks.product());
   for(unsigned int i=0; i<muonC.size(); i++){
      reco::MuonRef muon  = reco::MuonRef( muTracks, i );
      if(muon->pt()>mu1pt){mu2pt=mu1pt;mu1pt=muon->pt();
      }else if(muon->pt()>mu2pt){ mu2pt=muon->pt();}

      if(!muon->standAloneMuon().isNull()) {
         TrackRef SATrack = muon->standAloneMuon();         
         if(SATrack->pt()>SAmupt)SAmupt = SATrack->pt();
      }
   }
   if(SAmupt>=SAMuPtMin){ return true;}
   if(mu1pt>=Mu1PtMin && mu2pt>=Mu2PtMin){ return true;}



   //CHECK IF WE HAVE A HIGH PT / HIGH DEDX TRACK
   using reco::TrackCollection;
   Handle<TrackCollection> tkTracks;
   iEvent.getByLabel(input_track_collection,tkTracks);
   const reco::TrackCollection tkTC = *(tkTracks.product());

   Handle<ValueMap<DeDxData> >          dEdxTrackHandle;
   iEvent.getByLabel(input_dedx_collection, dEdxTrackHandle);
   const ValueMap<DeDxData> dEdxTrack = *dEdxTrackHandle.product();

   for(size_t i=0; i<tkTracks->size(); i++){
      reco::TrackRef trkRef = reco::TrackRef(tkTracks, i);
           
      if(trkRef->pt()>=trkPtMin && trkRef->eta()<=etaMax && trkRef->eta()>=etaMin && trkRef->normalizedChi2()<=chi2nMax){
           double dz  = trkRef->dz (recoVertex[0].position());
           double dxy = trkRef->dxy(recoVertex[0].position());
           double distancemin =sqrt(dxy*dxy+dz*dz);
           int closestvertex=0;
           for(unsigned int i=1;i<recoVertex.size();i++){
              dz  = trkRef->dz (recoVertex[i].position());
              dxy = trkRef->dxy(recoVertex[i].position());
              double distance = sqrt(dxy*dxy+dz*dz);
              if(distance < distancemin ){
                 distancemin = distance;
                 closestvertex=i;
              }
           }          
           dz  = trkRef->dz (recoVertex[closestvertex].position());
           dxy = trkRef->dxy(recoVertex[closestvertex].position());
           
           if(fabs(dz)<=dzMax && fabs(dxy)<=dxyMax ){              
              double dedx = dEdxTrack[trkRef].dEdx();
              int dedxnhits  = dEdxTrack[trkRef].numberOfMeasurements();
              if((dedx >=dedxMin || dedx<=dedxMaxLeft) && dedxnhits>=ndedxHits){return true;}
           }
      }
   }
   return false;

}

//define this as a plug-in
DEFINE_FWK_MODULE(HSCPEventFilter);
