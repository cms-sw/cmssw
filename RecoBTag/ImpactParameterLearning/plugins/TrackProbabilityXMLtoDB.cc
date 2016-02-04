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
// $Id: TrackProbabilityXMLtoDB.cc,v 1.3 2010/02/25 00:32:04 wmtan Exp $
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

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"

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

//#include "TH1F.h"
//#include "TFile.h"


#include <fstream>
#include <iostream>

using namespace reco;

//
// class decleration
//

class TrackProbabilityXMLtoDB : public edm::EDAnalyzer {
   public:
      explicit TrackProbabilityXMLtoDB(const edm::ParameterSet&);

    virtual void endJob()
    {
              edm::Service<cond::service::PoolDBOutputService> mydbservice;
              if( !mydbservice.isAvailable() ) return;
              
          edm::FileInPath f2d("RecoBTag/TrackProbability/data/2DHisto.xml");
          edm::FileInPath f3d("RecoBTag/TrackProbability/data/3DHisto.xml");
	  AlgorithmCalibration<TrackClassFilterCategory, CalibratedHistogramXML>*  calibrationOld=     new AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogramXML>((f3d.fullPath()).c_str());
	  AlgorithmCalibration<TrackClassFilterCategory, CalibratedHistogramXML>* calibration2dOld=       new AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogramXML>((f2d.fullPath()).c_str());

	  vector<pair<TrackClassFilterCategory, CalibratedHistogramXML> > data = calibrationOld->categoriesWithData();
	  vector<pair<TrackClassFilterCategory, CalibratedHistogramXML> > data2d = calibration2dOld->categoriesWithData();
          TrackProbabilityCalibration * calibration= new TrackProbabilityCalibration();
          TrackProbabilityCalibration * calibration2d= new TrackProbabilityCalibration();
	  for(int i = 0 ; i < data.size();i++)
    	  {
            TrackProbabilityCalibration::Entry entry;
            entry.category=data[i].first.categoryData();
            entry.histogram=data[i].second;
            calibration->data.push_back(entry);   
          }
	  for(int i = 0 ; i < data2d.size();i++)
    	  {
            TrackProbabilityCalibration::Entry entry;
            entry.category=data2d[i].first.categoryData();
            entry.histogram=data2d[i].second;
            calibration2d->data.push_back(entry);   
          }


         mydbservice->createNewIOV<TrackProbabilityCalibration>(calibration,  mydbservice->endOfTime(),"BTagTrackProbability3DRcd");    

         mydbservice->createNewIOV<TrackProbabilityCalibration>(calibration2d,  mydbservice->endOfTime(),"BTagTrackProbability2DRcd");   
               

    }    
      ~TrackProbabilityXMLtoDB() 
    {
    }



      virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

   private:
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
TrackProbabilityXMLtoDB::TrackProbabilityXMLtoDB(const edm::ParameterSet& parameters)
{
}

void
TrackProbabilityXMLtoDB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}

//define this as a plug-in

DEFINE_FWK_MODULE(TrackProbabilityXMLtoDB);
