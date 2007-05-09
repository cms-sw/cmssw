// -*- C++ -*-
//
// Package:    TrackIPProducer
// Class:      TrackIPProducer
// 
/**\class TrackIPProducer TrackIPProducer.cc RecoBTau/TrackIPProducer/src/TrackIPProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Thu Apr  6 09:56:23 CEST 2006
// $Id: TrackIPProducer.cc,v 1.2 2007/01/30 16:12:40 arizzi Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "RecoBTag/ImpactParameter/interface/TrackIPProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

//#include "MagneticField/Engine/interface/MagneticField.h"
//#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
//#include "RecoBTau/TrackTools/interface/SignedImpactParameter3D.h"
//#include "RecoBTau/TrackTools/interface/SignedTransverseImpactParameter.h"
//#include "RecoBTau/TrackTools/interface/SignedDecayLength3D.h"
//#include "RecoBTag/BTagTools/interface/SignedImpactParameter3D.h"
//#include "RecoBTag/BTagTools/interface/SignedTransverseImpactParameter.h"
//#include "RecoBTag/BTagTools/interface/SignedDecayLength3D.h"

//#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace std;
using namespace reco;
using namespace edm;

//
// constructors and destructor
//
TrackIPProducer::TrackIPProducer(const edm::ParameterSet& iConfig) : 
  m_config(iConfig)  {

  produces<reco::TrackIPTagInfoCollection>(); 
  
  m_associator = m_config.getParameter<string>("jetTracks");
  m_primaryVertexProducer = m_config.getParameter<string>("primaryVertex");

  m_computeProbabilities = m_config.getParameter<bool>("ComputeProbabilities"); //FIXME: use or remove
  
  m_cutPixelHits     =  m_config.getParameter<int>("MinimumNumberOfPixelHits"); //FIXME: use or remove
  m_cutTotalHits     =  m_config.getParameter<int>("MinimumNumberOfHits"); // used
  m_cutMaxTIP        =  m_config.getParameter<double>("MaximumTransverseImpactParameter"); // used
  m_cutMinPt         =  m_config.getParameter<double>("MinimumTransverseMomentum"); // used
  m_cutMaxDecayLen   =  m_config.getParameter<double>("MaximumDecayLength"); //used
  m_cutMaxChiSquared =  m_config.getParameter<double>("MaximumChiSquared"); //used
  m_cutMaxLIP        =  m_config.getParameter<double>("MaximumLongitudinalImpactParameter"); //used
  m_cutMaxDistToAxis =  m_config.getParameter<double>("MaximumDistanceToJetAxis"); //used

}

TrackIPProducer::~TrackIPProducer()
{
}

//
// member functions
//
// ------------ method called to produce the data  ------------
void
TrackIPProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   //input objects
   Handle<reco::JetTracksAssociationCollection> jetTracksAssociation;
   iEvent.getByLabel(m_associator,jetTracksAssociation);
   
   Handle<reco::VertexCollection> primaryVertex;
   iEvent.getByLabel(m_primaryVertexProducer,primaryVertex);
   
   edm::ESHandle<TransientTrackBuilder> builder;
   iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",builder);
  //  m_algo.setTransientTrackBuilder(builder.product());

  

   //output collections 
   reco::TrackIPTagInfoCollection * outCollection = new reco::TrackIPTagInfoCollection();

   //use first pv of the collection
   //FIXME: use BeamSpot when pv is missing
   const  Vertex  *pv;

   bool pvFound = (primaryVertex->size() != 0);
   if(pvFound)
   {
    pv = &(*primaryVertex->begin());
   }
    else 
   { // create a dummy PV
     Vertex::Error e;
     e(0,0)=0.0015*0.0015;
      e(1,1)=0.0015*0.0015;
     e(2,2)=15.*15.;
     Vertex::Point p(0,0,0);
     pv=  new Vertex(p,e,1,1,1);
   }
   
   double pvZ=pv->z();
 
   int i=0;
   JetTracksAssociationCollection::const_iterator it = jetTracksAssociation->begin();
   for(; it != jetTracksAssociation->end(); it++, i++)
     {

        GlobalVector direction(it->first->px(),it->first->py(),it->first->pz());
        TrackRefVector tracks = it->second;
        
        TrackRefVector selectedTracks;
        vector<Measurement1D> ip3Dv,ip2Dv,dLenv,jetDistv;
        vector<float> prob2D,prob3D;
       for (TrackRefVector::const_iterator itTrack = tracks.begin(); itTrack != tracks.end(); ++itTrack) {
             const Track & track = **itTrack;
             const TransientTrack & transientTrack = builder->build(&(**itTrack));
             //FIXME: this stuff is computed twice. transienttrack like container in IPTools for caching? 
             //       is it needed? does it matter at HLT?
             float distToAxis = IPTools::jetTrackDistance(transientTrack,direction,*pv).second.value();
             float dLen = IPTools::signedDecayLength3D(transientTrack,direction,*pv).second.value();

         if( track.pt() > m_cutMinPt  &&                          // minimum pt
                 fabs(track.d0()) < m_cutMaxTIP &&                // max transverse i.p.
                 track.recHitsSize() >= m_cutTotalHits &&         // min num tracker hits
                 fabs(track.dz()-pvZ) < m_cutMaxLIP &&            // z-impact parameter
                 track.normalizedChi2() < m_cutMaxChiSquared &&   // normalized chi2
                 fabs(distToAxis) < m_cutMaxDistToAxis  &&        // distance to JetAxis
                 fabs(dLen) < m_cutMaxDecayLen &&                 // max decay len
                 track.hitPattern().numberOfValidPixelHits() >= m_cutPixelHits //min # pix hits 
           )     // quality cuts
        { 
         //Fill vectors
         selectedTracks.push_back(*itTrack);
         ip3Dv.push_back(IPTools::signedImpactParameter3D(transientTrack,direction,*pv).second);
         ip2Dv.push_back(IPTools::signedTransverseImpactParameter(transientTrack,direction,*pv).second);
         dLenv.push_back(IPTools::signedDecayLength3D(transientTrack,direction,*pv).second);
         jetDistv.push_back(IPTools::jetTrackDistance(transientTrack,direction,*pv).second);

         if(m_computeProbabilities) {
              prob2D.push_back(-1.); 
              prob3D.push_back(-1.); 
//             pair<bool,double> prob3d =  m_probabilityEstimator->probability(0,sip3D.apply(transientTrack,direction,pv).second.significance(),track,*(jetTracks->key),pv);
//             pair<bool,double> prob2d =  m_probabilityEstimator->probability(1,stip.apply(transientTrack,direction,pv).second.significance(),track,*(jetTracks->key),pv);
          } 
    
         } // quality cuts if
     
      } //track loop
       TrackIPTagInfo tagInfo(ip2Dv,ip3Dv,dLenv,jetDistv,prob2D,prob3D,selectedTracks,edm::Ref<JetTracksAssociationCollection>(jetTracksAssociation,i));
       outCollection->push_back(tagInfo); 
     }
  // reco::TrackIPTagInfoCollection * outCollection = new reco::TrackIPTagInfoCollection();
 
    std::auto_ptr<reco::TrackIPTagInfoCollection> result(outCollection);
   iEvent.put(result);
   cout << "done"  << endl;
 
   if(!pvFound) delete pv; //dummy pv deleted

}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackIPProducer);

