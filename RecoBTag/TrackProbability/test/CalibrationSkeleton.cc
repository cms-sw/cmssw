// -*- C++ -*-
//
// Package:    TrackProbabilityXMLtoDB
// Class:      TrackProbabilityXMLtoDB
// 
/**\class TrackProbabilityXMLtoDB TrackProbabilityXMLtoDB.cc RecoBTag/TrackProbabilityXMLtoDB/src/TrackProbabilityXMLtoDB.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Wed Apr 12 11:12:49 CEST 2006
// $Id: CalibrationSkeleton.cc,v 1.12 2010/02/20 21:00:44 wmtan Exp $
//
//




// system include files
#include <memory>
#include <string>
#include <iostream>
using namespace std;

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"

//#include "RecoBTag/TrackProbability/interface/TrackClassFilterCategory.h"

#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
//#include "TrackProbabilityCalibratedHistogram.h"

#include "RecoBTag/BTagTools/interface/SignedTransverseImpactParameter.h"
#include "RecoBTag/BTagTools/interface/SignedImpactParameter3D.h"
#include "RecoBTag/BTagTools/interface/SignedDecayLength3D.h"

//CondFormats
#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

#include "RecoBTag/XMLCalibration/interface/AlgorithmCalibration.h"
#include "RecoBTag/XMLCalibration/interface/CalibratedHistogramXML.h"
#include "RecoBTag/TrackProbability/interface/TrackClassFilterCategory.h"
#include "TrackingTools/IPTools/interface/IPTools.h"

//#include "TH1F.h"
//#include "TFile.h"


#include <fstream>
#include <iostream>

using namespace reco;

//
// class decleration
//

class CalibrationSkeleton : public edm::EDAnalyzer {
   public:
  explicit CalibrationSkeleton(const edm::ParameterSet&);

  virtual void beginJob()
  {
    bool resetData=true;
    bool newBinning=false;
    edm::FileInPath f2d("RecoBTag/TrackProbability/data/2DHisto.xml");
    edm::FileInPath f3d("RecoBTag/TrackProbability/data/3DHisto.xml");
    calibrationNew   =  new AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogramXML>((f3d.fullPath()).c_str());
    calibration2dNew =  new AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogramXML>((f2d.fullPath()).c_str());
    if(resetData)
      {
	vector<pair<TrackClassFilterCategory, CalibratedHistogramXML> > data = calibrationNew->categoriesWithData();
	vector<pair<TrackClassFilterCategory, CalibratedHistogramXML> > data2d = calibration2dNew->categoriesWithData();
	for(unsigned int i = 0 ; i < data.size();i++)
          {
            if(newBinning) data[i].second = CalibratedHistogram(1000, 0, 50);
            else           data[i].second = CalibratedHistogram();
          }
	for(unsigned int i = 0 ; i < data2d.size();i++)
          {
            if(newBinning) data2d[i].second = CalibratedHistogram(1000, 0, 50);
            else           data2d[i].second = CalibratedHistogram();
          }
	
      }
//    calibrationNew->startCalibration();
  //  calibration2dNew->startCalibration();
  }    
  
  virtual void endJob()
  {
    
    edm::Service<cond::service::PoolDBOutputService> mydbservice;
    if( !mydbservice.isAvailable() ) return;
    
    vector<pair<TrackClassFilterCategory, CalibratedHistogramXML> > data = calibrationNew->categoriesWithData();
    vector<pair<TrackClassFilterCategory, CalibratedHistogramXML> > data2d = calibration2dNew->categoriesWithData();
    TrackProbabilityCalibration * calibration= new TrackProbabilityCalibration();
    TrackProbabilityCalibration * calibration2d= new TrackProbabilityCalibration();
    for(unsigned int i = 0 ; i < data.size();i++)
      {
	TrackProbabilityCalibration::Entry entry;
	entry.category=data[i].first.categoryData();
	entry.histogram=data[i].second;
	calibration->data.push_back(entry);
      }
    for(unsigned int i = 0 ; i < data2d.size();i++)
      {
	TrackProbabilityCalibration::Entry entry;
	entry.category=data2d[i].first.categoryData();
	entry.histogram=data2d[i].second;
	calibration2d->data.push_back(entry);
      }
    
    
    mydbservice->createNewIOV<TrackProbabilityCalibration>(calibration,  mydbservice->endOfTime(),"BTagTrackProbability3DRcd");
    
    mydbservice->createNewIOV<TrackProbabilityCalibration>(calibration2d,  mydbservice->endOfTime(),"BTagTrackProbability2DRcd");
    
    
  }
  
  
  ~CalibrationSkeleton() 
  {
  }

  
  
  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  
private:
  AlgorithmCalibration<TrackClassFilterCategory, CalibratedHistogramXML>*  calibrationNew;
  AlgorithmCalibration<TrackClassFilterCategory, CalibratedHistogramXML>* calibration2dNew;
  
  
  int count;
  int ntracks;
  int  m_cutPixelHits;
  int  m_cutTotalHits;
  double  m_cutMaxTIP;
  double  m_cutMinPt;
  double  m_cutMaxDecayLen;
  double  m_cutMaxChiSquared;
  double  m_cutMaxLIP;
  double m_cutMaxDistToAxis;
  double m_cutMinProb;

  edm::InputTag m_assoc;
  edm::InputTag m_jets;
  edm::InputTag m_primaryVertexProducer;
};

//
// constructors and destructor
//
CalibrationSkeleton::CalibrationSkeleton(const edm::ParameterSet& parameters)
{
}

void
CalibrationSkeleton::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{



  using namespace edm;
  using namespace reco;
  using namespace std;
 
  Handle<reco::VertexCollection> primaryVertex;
  iEvent.getByLabel("offlinePrimaryVertices",primaryVertex);
  
  //*********************************************************************************** 
  //look at reco vertices 
  const  reco::Vertex  *pv;
  bool newvertex = false;
  
  bool pvFound = (primaryVertex->size() != 0);
  if(pvFound)
    {
      pv = &(*primaryVertex->begin());
    }
  else 
    { 
      reco::Vertex::Error e;
      e(0,0)=0.0015*0.0015;
      e(1,1)=0.0015*0.0015;
      e(2,2)=15.*15.;
      reco::Vertex::Point p(0,0,0);
      pv=  new reco::Vertex(p,e,1,1,1);
      newvertex=true;
    }
  
  //*************************************************************************
  //look at JetTracks
  edm::Handle<JetTracksAssociationCollection> associationHandle;
  iEvent.getByLabel("ak5JetTracksAssociatorAtVertex", associationHandle);
  reco::JetTracksAssociationCollection::const_iterator it = associationHandle->begin();
  
  
  
  //***********************************************************************************
  //mandatory for ip significance reco
  const TransientTrackBuilder * m_transientTrackBuilder_producer;
  edm::ESHandle<TransientTrackBuilder> builder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",builder);
  m_transientTrackBuilder_producer = builder.product();
  //***********************************************************************************
  
    int i = 0;
  for(; it != associationHandle->end(); it++, i++)
    { 
      
      const  JetTracksAssociationRef & jetTracks = Ref<JetTracksAssociationCollection>(associationHandle,i);
      
      
      //GlobalVector direction(jetTracks->key->px(),jetTracks->key->py(),jetTracks->key->pz());
       
 //     if(jetTracks->second.size() <2 ) continue;
       
       bool directionWithTracks = false;
       TrackRefVector tracks = it->second;
       math::XYZVector jetMomentum=it->first->momentum()/2.;
        if(directionWithTracks)
         {
           for (TrackRefVector::const_iterator itTrack = tracks.begin(); itTrack != tracks.end(); ++itTrack) {
               if((**itTrack).numberOfValidHits() >= m_cutTotalHits )  //minimal quality cuts
                  jetMomentum+=(**itTrack).momentum();
             }
         }
          else
         {
            jetMomentum=it->first->momentum();
         }
        GlobalVector direction(jetMomentum.x(),jetMomentum.y(),jetMomentum.z());
 
      
      
      
      for(RefVector<reco::TrackCollection>::const_iterator itt=tracks.begin() ; itt!=tracks.end(); itt++ )
	 {
	   
	   //loop on jets and tracks 
	   
	   TrackClassFilterInput input(**itt,*(it->first), *pv);
	   
           const TransientTrack & transientTrack = builder->build(&(**itt));
	   const_cast<CalibratedHistogramXML*>(calibrationNew->getCalibData(input))->fill(IPTools::signedImpactParameter3D(transientTrack,direction,*pv).second.significance());
//  calibration2dNew->getCalibData(input)->Fill(significance2D);
	   
//	   calibrationNew->updateCalibration(input);
//	   calibration2dNew->updateCalibration(input);
	   
	 }
       
       
    }
  
}
//define this as a plug-in
DEFINE_FWK_MODULE(CalibrationSkeleton);
