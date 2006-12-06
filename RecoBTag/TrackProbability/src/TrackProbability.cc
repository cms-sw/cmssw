// -*- C++ -*-
//
// Package:    TrackProbability
// Class:      TrackProbability
// 
/**\class TrackProbability TrackProbability.cc RecoBTag/TrackProbability/src/TrackProbability.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Thu Apr  6 09:56:23 CEST 2006
// $Id: TrackProbability.cc,v 1.20 2006/10/27 01:35:35 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "RecoBTag/TrackProbability/interface/TrackProbability.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

//#include "MagneticField/Engine/interface/MagneticField.h"
//#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoBTag/BTagTools/interface/SignedImpactParameter3D.h"
#include "RecoBTag/BTagTools/interface/SignedTransverseImpactParameter.h"
#include "RecoBTag/TrackProbability/interface/TrackClassFilterCategory.h"
#include "RecoBTag/XMLCalibration/interface/CalibratedHistogram.h"

//#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace reco;
using namespace edm;

//
// constructors and destructor
//
TrackProbability::TrackProbability(const edm::ParameterSet& iConfig) : 
  m_config(iConfig), 
  m_algo(iConfig.getParameter<edm::ParameterSet>("AlgorithmPSet") ) {

  //FIXME: delete the histogram probability estimator !!
  m_algo.setProbabilityEstimator(new HistogramProbabilityEstimator( new AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogram>("3DHisto.xml"),
new AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogram>("2DHisto.xml")) );

  //outputInstanceName_( iConfig.getParameter<std::string>( "outputInstanceName" ) ) {
  //produces<reco::JetTagCollection>( outputInstanceName_ );  //Several producer so I put a label
   produces<reco::JetTagCollection>();  //get rid of this label it is useless if name come  from .cfg
  produces<reco::TrackProbabilityTagInfoCollection>();       //Only one producer
  m_associator = m_config.getParameter<string>("jetTracks");
  m_primaryVertexProducer = m_config.getParameter<string>("primaryVertex");
}

TrackProbability::~TrackProbability()
{
}

//
// member functions
//
// ------------ method called to produce the data  ------------
void
TrackProbability::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
   //input objects
   Handle<reco::JetTracksAssociationCollection> jetTracksAssociation;
   iEvent.getByLabel(m_associator,jetTracksAssociation);
   Handle<reco::VertexCollection> primaryVertex;
   iEvent.getByLabel(m_primaryVertexProducer,primaryVertex);
   

    edm::ESHandle<TransientTrackBuilder> builder;
    iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",builder);
    m_algo.setTransientTrackBuilder(builder.product());
  //  ESHandle<MagneticField> magneticField;
    //iSetup.get<IdealMagneticFieldRecord>().get(magneticField);
//    m_algo.setMagneticField(magneticField.product());



   //output collections 
   reco::JetTagCollection * baseCollection = new reco::JetTagCollection();
   reco::TrackProbabilityTagInfoCollection * extCollection = new reco::TrackProbabilityTagInfoCollection();

   //use first pv of the collection
   //FIXME: handle missing PV with a dummy PV
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
   
   JetTracksAssociationCollection::const_iterator it = jetTracksAssociation->begin();
   for(; it != jetTracksAssociation->end(); it++)
     {
      int i=it->key.key();
      pair<JetTag,TrackProbabilityTagInfo>  result=m_algo.tag(edm::Ref<JetTracksAssociationCollection>(jetTracksAssociation,i),*pv);
      baseCollection->push_back(result.first);    
      extCollection->push_back(result.second);    
   
   }
  
    std::auto_ptr<reco::JetTagCollection> resultBase(baseCollection);
    edm::OrphanHandle <reco::JetTagCollection > jetTagHandle =  iEvent.put(resultBase); //, outputInstanceName_ );
    reco::TrackProbabilityTagInfoCollection::iterator it_ext =extCollection->begin();
    int cc=0;
    for(;it_ext!=extCollection->end();it_ext++)
      {
        it_ext->setJetTag(JetTagRef(jetTagHandle,cc));
        cc++;
      }
  
   std::auto_ptr<reco::TrackProbabilityTagInfoCollection> resultExt(extCollection);
   iEvent.put(resultExt);
   
   if(!pvFound) delete pv; //dummy pv deleted
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackProbability);

