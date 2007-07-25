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
// $Id: TrackProbabilityXMLtoDB.cc,v 1.3 2007/06/28 17:28:20 fwyzard Exp $
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
#include "FWCore/ParameterSet/interface/InputTag.h"

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

class CalibrationSkeleton : public edm::EDAnalyzer {
   public:
      explicit CalibrationSkeleton(const edm::ParameterSet&);

   virtual vois beginJob()
    {
      edm::FileInPath f2d("RecoBTag/TrackProbability/data/2DHisto.xml");
      edm::FileInPath f3d("RecoBTag/TrackProbability/data/3DHisto.xml");
      calibrationNew   =  new AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogramXML>((f3d.fullPath()).c_str());
      calibration2dNew =  new AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogramXML>((f2d.fullPath()).c_str());
      vector<float> * bins =0;
      if(m_resetData)
          {
              if(m_newBinning)
                bins = new  vector<float>(CalibratedHistogram::constantBinning(1000,0,50));
          vector<pair<TrackClassFilterCategory, CalibratedHistogramXML> > data = calibrationNew->categoriesWithData();
          vector<pair<TrackClassFilterCategory, CalibratedHistogramXML> > data2d = calibration2dNew>categoriesWithData();
          for(int i = 0 ; i < data.size();i++)
          {
            data[i].second->reset();
            if(bins)  data2d[i].second->setUpperLimits(*bins)
          }
          for(int i = 0 ; i < data2d.size();i++)
          {
            data2d[i].second->reset();
            if(bins)  data2d[i].second->setUpperLimits(*bins)
          }

          }
     if(bins) delete bins;

    }    

    virtual void endJob()
    {
          vector<pair<TrackClassFilterCategory, CalibratedHistogramXML> > data = calibrationNew->categoriesWithData();
          vector<pair<TrackClassFilterCategory, CalibratedHistogramXML> > data2d = calibration2dNew>categoriesWithData();
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
CalibrationSkeleton::TrackProbabilityXMLtoDB(const edm::ParameterSet& parameters)
{
}

void
CalibrationSkeleton::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

//loop on jets and tracks 
{
//  TrackClassFilterInput input(track,jet,vertex);
//  m_calibrationNew->getCalibData(input)->Fill(significance3D);
//  m_calibration2dNew->getCalibData(input)->Fill(significance2D);
}


}

//define this as a plug-in
DEFINE_FWK_MODULE(CalibrationSkeleton);
