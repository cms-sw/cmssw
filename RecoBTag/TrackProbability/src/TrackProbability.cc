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
// $Id: TrackProbability.cc,v 1.4 2007/01/23 14:11:02 arizzi Exp $
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
#include "RecoBTag/XMLCalibration/interface/AlgorithmCalibration.h"

#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"
//#include "RecoBTag/TrackProbability/interface/CalibrationInterface.h"
#include "RecoBTag/XMLCalibration/interface/CalibrationInterface.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability2DRcd.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability3DRcd.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

//#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace reco;
using namespace edm;
using namespace edm::eventsetup;

//
// constructors and destructor
//
TrackProbability::TrackProbability(const edm::ParameterSet& iConfig) : 
  m_config(iConfig), 
  m_algo(iConfig.getParameter<edm::ParameterSet>("AlgorithmPSet") ) {
  m_useDB=iConfig.getParameter<bool>("UseDB");
  if(!m_useDB)
  {
   edm::FileInPath f2d("RecoBTag/TrackProbability/data/2DHisto.xml");
   edm::FileInPath f3d("RecoBTag/TrackProbability/data/3DHisto.xml");
   m_algo.setProbabilityEstimator(new HistogramProbabilityEstimator( new AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogramXML>((f3d.fullPath()).c_str()),
               new AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogramXML>((f2d.fullPath()).c_str())) );
  }

  produces<reco::JetTagCollection>();  //get rid of this label it is useless if name come  from .cfg
  produces<reco::TrackProbabilityTagInfoCollection>();       //Only one producer
  m_associator = m_config.getParameter<string>("jetTracks");
  m_primaryVertexProducer = m_config.getParameter<string>("primaryVertex");
  m_calibrationCacheId3D= 0;
  m_calibrationCacheId2D= 0;
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
  
  if(m_useDB)
  {
   const EventSetupRecord & re2D= iSetup.get<BTagTrackProbability2DRcd>();
   const EventSetupRecord & re3D= iSetup.get<BTagTrackProbability3DRcd>();
   unsigned long long cacheId2D= re2D.cacheIdentifier();
   unsigned long long cacheId3D= re3D.cacheIdentifier();

   if(cacheId2D!=m_calibrationCacheId2D || cacheId3D!=m_calibrationCacheId3D  )  //Calibration changed
   {
     cout<< "Calibration data changed" << endl;
     //iSetup.get<BTagTrackProbabilityRcd>().get(calib);
     ESHandle<TrackProbabilityCalibration> calib2DHandle;
     iSetup.get<BTagTrackProbability2DRcd>().get(calib2DHandle);
     ESHandle<TrackProbabilityCalibration> calib3DHandle;
     iSetup.get<BTagTrackProbability3DRcd>().get(calib3DHandle);

     const TrackProbabilityCalibration *  ca2D= calib2DHandle.product();
     const TrackProbabilityCalibration *  ca3D= calib3DHandle.product();

     CalibrationInterface<TrackClassFilterCategory,CalibratedHistogramXML> * calib3d =  new CalibrationInterface<TrackClassFilterCategory,CalibratedHistogramXML>;
     CalibrationInterface<TrackClassFilterCategory,CalibratedHistogramXML> * calib2d =  new CalibrationInterface<TrackClassFilterCategory,CalibratedHistogramXML>;

     for(size_t i=0;i<ca3D->data.size(); i++)    
     {
        cout <<  "  Adding category" << endl;
        calib3d->addEntry(TrackClassFilterCategory(ca3D->data[i].category),ca3D->data[i].histogram); // convert category data to filtering category
     }
    
     for(size_t i=0;i<ca2D->data.size(); i++)    
     {
        cout <<  "  Adding category" << endl;
        calib2d->addEntry(TrackClassFilterCategory(ca2D->data[i].category),ca2D->data[i].histogram); // convert category data to filtering category
     }
  
     if(m_algo.probabilityEstimator()) delete m_algo.probabilityEstimator();  //this should delete also old calib via estimator destructor
     m_algo.setProbabilityEstimator(new HistogramProbabilityEstimator(calib3d,calib2d));

   }
   m_calibrationCacheId3D=cacheId3D;
   m_calibrationCacheId2D=cacheId2D;
   }
   

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

