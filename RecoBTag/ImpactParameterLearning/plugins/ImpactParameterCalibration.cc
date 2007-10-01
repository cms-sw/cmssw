// -*- C++ -*-
//
// Package:    ImpactParameterCalibration
// Class:      ImpactParameterCalibration
// 
/**\class ImpactParameterCalibration ImpactParameterCalibration.cc RecoBTag/ImpactParameterCalibration/src/ImpactParameterCalibration.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremy Andrea/Andrea Rizzi
//         Created:  Mon Aug  6 16:10:38 CEST 2007
// $Id$
//
//
// system include files
#include <memory>
#include <string>
#include <iostream>
using namespace std;

// user include files
/*#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
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
#include "TrackingTools/IPTools/interface/IPTools.h"

//#include "TH1F.h"
//#include "TFile.h"


#include <fstream>
#include <iostream>
*/
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"


#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"

#include "CondFormats/DataRecord/interface/BTagTrackProbability2DRcd.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability3DRcd.h"

#include <TClass.h>
#include <TBuffer.h>
#include <TBufferXML.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>

#include "TrackClassMatch.h"


using namespace reco;
using namespace std;
//
// class decleration
//



class ImpactParameterCalibration : public edm::EDAnalyzer {
   public:
      explicit ImpactParameterCalibration(const edm::ParameterSet&);
      ~ImpactParameterCalibration();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      edm::ParameterSet config;

   static TrackProbabilityCategoryData createCategory(double  pmin,double  pmax,
                 double  etamin,  double  etamax,
                  int  nhitmin,  int  nhitmax,
                 int  npixelhitsmin, int  npixelhitsmax,
                  double cmin, double cmax, int withFirst)
  {
  TrackProbabilityCategoryData c;
  c.pMin=pmin;
  c.pMax=pmax;
  c.etaMin=etamin;
  c.etaMax=etamax;
  c.nHitsMin=nhitmin;
  c.nHitsMax=nhitmax;
  c.nPixelHitsMin=npixelhitsmin;
  c.nPixelHitsMax=npixelhitsmax;
  c.chiMin=cmin;
  c.chiMax=cmax;
  c.withFirstPixel=withFirst;
  return c;
 }

   TrackProbabilityCalibration * m_calibration[2];
   edm::InputTag m_iptaginfo;
   edm::InputTag m_pv;


};

ImpactParameterCalibration::ImpactParameterCalibration(const edm::ParameterSet& iConfig):config(iConfig)
{
  m_iptaginfo = iConfig.getParameter<edm::InputTag>("tagInfoSrc");
  m_pv = iConfig.getParameter<edm::InputTag>("primaryVertexSrc");

}


ImpactParameterCalibration::~ImpactParameterCalibration()
{
}


void
ImpactParameterCalibration::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;

  Handle<TrackIPTagInfoCollection> ipHandle;
  iEvent.getByLabel(m_iptaginfo, ipHandle);
  const TrackIPTagInfoCollection & ip = *(ipHandle.product());

  cout << "Found " << ip.size() << " TagInfo" << endl;

  Handle<reco::VertexCollection> primaryVertex;
  iEvent.getByLabel(m_pv,primaryVertex);

  vector<TrackProbabilityCalibration::Entry>::iterator found;
  vector<TrackProbabilityCalibration::Entry>::iterator it_begin;
  vector<TrackProbabilityCalibration::Entry>::iterator it_end;

      
   TrackIPTagInfoCollection::const_iterator it = ip.begin();
   for(; it != ip.end(); it++)
     {
      TrackRefVector selTracks=it->selectedTracks();
//      if(it->primaryVertex().isNull()) continue;
      if(primaryVertex.product()->size() == 0) continue;
      const Vertex & pv = *(primaryVertex.product()->begin());
           
      for(int i=0; i < 2;i++)
      { 
        vector<Measurement1D> ip = it->impactParameters(i);
        it_begin=m_calibration[i]->data.begin();
        it_end=m_calibration[i]->data.end();
  
      for(int j=0;j<ip.size(); j++)
        {
          TrackClassMatch::Input input(*selTracks[j],*it->jet(),pv);
          if(ip[j].significance() < 0) 
           {
            found = std::find_if(it_begin,it_end,bind1st(TrackClassMatch(),input));

            if(found!=it_end) 
              found->histogram.fill(-ip[j].significance());
            else
              std::cout << "No category for this track!!" << std::endl;
           }
         }
      } 
     }  
      
      
  
}










// ------------ method called once each job just before starting event loop  ------------
void 
ImpactParameterCalibration::beginJob(const edm::EventSetup & iSetup)
{
 using namespace edm;
   m_calibration[0] =   new TrackProbabilityCalibration();
   m_calibration[1] =   new TrackProbabilityCalibration();

/*        m_xmlfilename3D = iConfig.getParameter<std::string>("xmlfilename3D");
        m_xmlfilename2D = iConfig.getParameter<std::string>("xmlfilename2D");
        m_assoc         = iConfig.getParameter<edm::InputTag>("JetTrackTag");
        m_primaryVertexProducer = iConfig.getParameter<edm::InputTag>("primaryVertexProducer");
        m_resetData     = iConfig.getParameter<bool>("resetData");
        m_newBinning    = iConfig.getParameter<bool>("newBinning");
 */


  CalibratedHistogram hist(config.getParameter<int>("nBins"),0,config.getParameter<double>("maxSignificance"));

  std::string categories = config.getParameter<std::string>("inputCategories");

  if(categories == "HardCoded")
  {
   vector<TrackProbabilityCategoryData> v;
  v.push_back(createCategory(0,5000,0,2.4,8,50,1,1,0,5,0));
  v.push_back(createCategory(0,5000,0,2.4,8,50,2,5,2.5,5,0));
  v.push_back(createCategory(0,8,0,0.8,8,50,3,5,0,2.5,0));
  v.push_back(createCategory(0,8,0.8,1.6,8,50,3,5,0,2.5,0));
  v.push_back(createCategory(0,8,1.6,2.4,8,50,3,5,0,2.5,0));
  v.push_back(createCategory(0,8,0,2.4,8,50,2,2,0,2.5,0));
  v.push_back(createCategory(8,5000,0,0.8,8,50,3,5,0,2.5,0));
  v.push_back(createCategory(8,5000,0.8,1.6,8,50,3,5,0,2.5,0));
  v.push_back(createCategory(8,5000,1.6,2.4,8,50,3,5,0,2.5,0));
  v.push_back(createCategory(8,5000,0,2.4,8,50,2,2,0,2.5,0));
  for(int i=0;i <2 ;i++)
   for(int j=0;j<v.size() ; j++)
    {
     TrackProbabilityCalibration::Entry e;
     e.category=v[j];
     e.histogram=hist;
     m_calibration[i]->data.push_back(e);
    }

  }

  if(categories == "EventSetup")
   {
    bool resetHistogram = config.getParameter<bool>("resetHistograms");
    ESHandle<TrackProbabilityCalibration> calib2DHandle;
    iSetup.get<BTagTrackProbability2DRcd>().get(calib2DHandle);
    ESHandle<TrackProbabilityCalibration> calib3DHandle;
    iSetup.get<BTagTrackProbability3DRcd>().get(calib3DHandle);
    const TrackProbabilityCalibration * ca[2];
    ca[0]  = calib3DHandle.product();
    ca[1]  = calib2DHandle.product();
    for(int i=0;i <2 ;i++)
    for(int j=0;j<ca[i]->data.size() ; j++)
    {
     TrackProbabilityCalibration::Entry e;
     e.category=ca[i]->data[j].category;

     if(resetHistogram)
      e.histogram=hist;
     else 
      e.histogram=ca[i]->data[j].histogram;

     m_calibration[i]->data.push_back(e);
    }

  }



/*  edm::FileInPath f2d(m_xmlfilename2D);
    edm::FileInPath f3d(m_xmlfilename3D);
    calibrationNew   =  new AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogramXML>((f3d.fullPath()).c_str());
    calibration2dNew =  new AlgorithmCalibration<TrackClassFilterCategory,CalibratedHistogramXML>((f2d.fullPath()).c_str());
    vector<float> * bins =0;
    if(m_resetData)
      {
	if(m_newBinning)  bins = new  vector<float>(CalibratedHistogram::constantBinning(m_nBin,0,m_range));
	vector<pair<TrackClassFilterCategory, CalibratedHistogramXML> > data = calibrationNew->categoriesWithData();
	vector<pair<TrackClassFilterCategory, CalibratedHistogramXML> > data2d = calibration2dNew->categoriesWithData();
      	std::cout <<  data.size() <<  std::endl;
	for(unsigned int i = 0 ; i < data.size();i++)
          {
            data[i].second.reset();
            if(bins)  data[i].second.setUpperLimits(*bins);
          }
	for(unsigned int i = 0 ; i < data2d.size();i++)
          {
            data2d[i].second.reset();
            if(bins)  data2d[i].second.setUpperLimits(*bins);
          }
	
      }
    if(bins) delete bins;
    
*/


}

// ------------ method called once each job just after ending the event loop  ------------
void 
ImpactParameterCalibration::endJob() {

  if(config.getParameter<bool>("writeToDB"))
  {
    edm::Service<cond::service::PoolDBOutputService> mydbservice;
    if( !mydbservice.isAvailable() ) return;
    mydbservice->createNewIOV<TrackProbabilityCalibration>(m_calibration[0],  mydbservice->endOfTime(),"BTagTrackProbability3DRcd");
    mydbservice->createNewIOV<TrackProbabilityCalibration>(m_calibration[1],  mydbservice->endOfTime(),"BTagTrackProbability2DRcd");
  } 
    

  if(config.getParameter<bool>("writeToRootXML"))
  {
   std::ofstream of2("2d.xml");
   TBufferXML b2(TBuffer::kWrite);
   of2 << b2.ConvertToXML(const_cast<void*>(static_cast<const void*>(m_calibration[1])),
                                                  TClass::GetClass("TrackProbabilityCalibration"),
                                                  kTRUE, kFALSE);
   of2.close();
   std::ofstream of3("3d.xml");
   TBufferXML b3(TBuffer::kWrite);
   of3 << b3.ConvertToXML(const_cast<void*>(static_cast<const void*>(m_calibration[0])),
                                                  TClass::GetClass("TrackProbabilityCalibration"),
                                                  kTRUE, kFALSE);
   of3.close();
  }

 
  if(config.getParameter<bool>("writeToBinary"))
  {
    std::ofstream ofile("2d.dat");
    TBuffer buffer(TBuffer::kWrite);
    buffer.StreamObject(const_cast<void*>(static_cast<const void*>(m_calibration[1])),
                                                  TClass::GetClass("TrackProbabilityCalibration"));
    Int_t size = buffer.Length();
    ofile.write(buffer.Buffer(),size);
    ofile.close();

    std::ofstream ofile3("3d.dat");
    TBuffer buffer3(TBuffer::kWrite);
    buffer3.StreamObject(const_cast<void*>(static_cast<const void*>(m_calibration[0])),
                                                  TClass::GetClass("TrackProbabilityCalibration"));
    Int_t size3 = buffer3.Length();
    ofile3.write(buffer3.Buffer(),size3);
    ofile3.close();
  }





}

//define this as a plug-in
DEFINE_FWK_MODULE(ImpactParameterCalibration);
