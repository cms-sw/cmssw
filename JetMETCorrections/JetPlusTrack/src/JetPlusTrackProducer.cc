// -*- C++ -*-
//
// Package:    MCJet
// Class:      MCJet
// 
/**\class MCJet MCJet.cc JetMETCorrections/MCJet/src/MCJet.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Olga Kodolova
//         Created:  Wed Feb  1 17:04:23 CET 2006
// $Id: MCJetProducer.cc,v 1.3 2006/04/02 10:18:20 kodolova Exp $
//
//


// system include files
#include <memory>
#include <iostream>
#include <iomanip>
#include <string>
// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "JetMETCorrections/JetPlusTrack/interface/JetPlusTrackProducer.h"
#include "JetMETCorrections/JetPlusTrack/interface/JetPlusTrackAlgorithm.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/Candidate/interface/Candidate.h"

using namespace std;
using namespace reco;
namespace cms 
{

//
// constructors and destructor
//
JetPlusTrack::JetPlusTrack(const edm::ParameterSet& iConfig): 
                                           mAlgorithm(),
					   mInputJets(iConfig.getParameter<edm::InputTag>("src1")),
					   mInputCaloTower(iConfig.getParameter<edm::InputTag>("src2")),
					   mInputPVfCTF(iConfig.getParameter<edm::InputTag>("src3")),
					   theRcalo(iConfig.getParameter<double>("rcalo")),
					   theRvert(iConfig.getParameter<double>("rvert")),
					   theResponse(iConfig.getParameter<int>("respalgo"))

{
    m_inputTrackLabel = iConfig.getUntrackedParameter<std::string>("inputTrackLabel","ctfWithMaterialTracks");
   //now do what ever other initialization is needed
   // Fill data labels
    std::vector<std::string> theLabels = iConfig.getParameter<std::vector<std::string> >("labels");
    produces<CaloJetCollection>();
    mAlgorithm.setParameters(theRcalo,theRvert,theResponse,theLabels);
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
JetPlusTrack::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
// Take jet collection
   edm::Handle<CaloJetCollection> jets;                              //Define Inputs
   iEvent.getByLabel(mInputJets, jets);                              //Get Inputs
// Take Vertex collection   
   edm::Handle<reco::VertexCollection> primary_vertices;                 //Define Inputs (vertices)
   iEvent.getByLabel(mInputPVfCTF, primary_vertices);                  //Get Inputs    (vertices)
// Take Track Collection
   edm::Handle<reco::TrackCollection> trackCollection;
   iEvent.getByLabel(m_inputTrackLabel,trackCollection);
   const reco::TrackCollection tC = *(trackCollection.product());
   
   
   auto_ptr<CaloJetCollection> result (new CaloJetCollection);       //Corrected jets
   
   CaloJetCollection::const_iterator jet = jets->begin ();
   
      cout<<" Size of jets "<<jets->size()<<endl;
   VertexCollection::const_iterator pv = primary_vertices->begin(); 
   
   cout<<" Size of jets "<<jets->size()<<
         " Number of tracks "<<tC.size()<<
	 " Number of vertices "<<primary_vertices->size()<<
	 endl;
   VertexCollection::const_iterator pvmax = pv;
   
   double ptmax = -1000.;
   vector<Track> trPV;
   
   for (; pv != primary_vertices->end(); pv++ )
   {
      double pto = 0.;
      
      vector<Track>  tmp;
      for (reco::track_iterator track = (*pv).tracks_begin();
                track != (*pv).tracks_end(); track++)
		{
		   pto = pto + (*track)->pt();
		   tmp.push_back((**track));
		}
       if ( ptmax < pto )
       {  
           ptmax = pto;
	   pvmax = pv;
	   trPV = tmp;
       }		
   } 
    
    cout<<" Vertex with pt= "<<ptmax<<endl;
    
      mAlgorithm.setPrimaryVertex((*pvmax)); 
      mAlgorithm.setTracksFromPrimaryVertex(trPV);
      
      
   if(jets->size() > 0 )
   { 
   for (; jet != jets->end (); jet++) {
//      result->push_back (mAlgorithm.applyCorrection (*jet));
      result->push_back (mAlgorithm.applyCorrection (*jet, iEvent, iSetup)); 
      cout<<" Size of the result "<<result->size()<<endl;
   }
   }
      cout<<" Put result "<<result->size()<<endl;
   iEvent.put(result);  //Puts Corrected Jet Collection into event
   
}

}//end namespace cms
