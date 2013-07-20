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
// $Id: ImpactParameterCalibration.cc,v 1.15 2012/07/05 14:48:01 eulisse Exp $
//
//
// system include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Utilities/General/interface/FileInPath.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"


#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"
#include "CondFormats/BTauObjects/interface/CalibratedHistogram.h"

#include "CondFormats/DataRecord/interface/BTagTrackProbability2DRcd.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability3DRcd.h"

#include "TClass.h"

#include "TBufferFile.h"

#include "TBufferXML.h"
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
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      virtual void initFromFirstES(const edm::EventSetup&);
      edm::ParameterSet config;
      bool m_needInitFromES;
   TrackProbabilityCalibration * fromXml(edm::FileInPath xmlCalibration);

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
  unsigned int minLoop, maxLoop;

};

ImpactParameterCalibration::ImpactParameterCalibration(const edm::ParameterSet& iConfig):config(iConfig)
{
  m_needInitFromES = false;
  m_iptaginfo = iConfig.getParameter<edm::InputTag>("tagInfoSrc");
  m_pv = iConfig.getParameter<edm::InputTag>("primaryVertexSrc");
  bool createOnlyOne = iConfig.getUntrackedParameter<bool>("createOnlyOneCalibration", false);
  minLoop=0;
  maxLoop=1;
  if (createOnlyOne == true){
    int whichCalib = iConfig.getUntrackedParameter<int>("dimension", 2);
    if (whichCalib==2){
      std::cout <<" Writing only 2D calibrations"<<std::endl;
      minLoop=1;
      maxLoop=1;
    }else if (whichCalib==3){
      std::cout <<" Writing only 3D calibrations"<<std::endl;
      minLoop=0;
      maxLoop=0;
    }else {
      std::cout <<" Dimension not found: "<<whichCalib<<"; it must be either 2 or 3"<<std::endl;
    }
  }

}


ImpactParameterCalibration::~ImpactParameterCalibration()
{
}


void
ImpactParameterCalibration::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  if(m_needInitFromES) initFromFirstES(iSetup);
  using namespace edm;
  using namespace reco;

  Handle<TrackIPTagInfoCollection> ipHandle;
  iEvent.getByLabel(m_iptaginfo, ipHandle);
  const TrackIPTagInfoCollection & ip = *(ipHandle.product());

//  cout << "Found " << ip.size() << " TagInfo" << endl;

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
      if(primaryVertex.product()->size() == 0) 
       {
       std::cout << "No PV in the event!!" << std::endl;
         continue;
        }
       const Vertex & pv = *(primaryVertex.product()->begin());
           
      for(unsigned int i=minLoop; i <= maxLoop;i++)
      { 
        it_begin=m_calibration[i]->data.begin();
        it_end=m_calibration[i]->data.end();
  
      for(unsigned int j=0;j<selTracks.size(); j++)
        {
          double ipsig;
          if (i==0) ipsig  = it->impactParameterData()[j].ip3d.significance();
          else  ipsig  = it->impactParameterData()[j].ip2d.significance();
          TrackClassMatch::Input input(*selTracks[j],*it->jet(),pv);
          if(ipsig < 0) 
           {
            found = std::find_if(it_begin,it_end,bind1st(TrackClassMatch(),input));
//            std::cout << ip[j].significance() << std::endl; 
            if(found!=it_end) 
              found->histogram.fill(-ipsig);
            else
              {std::cout << "No category for this track!!" << std::endl;
              std::cout << "p       "  <<(*selTracks[j]).p ()  << std::endl;
              std::cout << "eta     " << (*selTracks[j]).eta() << std::endl;
              std::cout << "NHit    " << (*selTracks[j]).numberOfValidHits() << std::endl;
              std::cout << "NPixHit " << (*selTracks[j]).hitPattern().numberOfValidPixelHits() << std::endl;
              std::cout << "FPIXHIT " << (*selTracks[j]).hitPattern().hasValidHitInFirstPixelBarrel() << std::endl;}
	      
           }
         }
      } 
     }  
      
         
  
}







void ImpactParameterCalibration::initFromFirstES(const edm::EventSetup& iSetup)
{
  using namespace edm;

    CalibratedHistogram hist(config.getParameter<int>("nBins"),0,config.getParameter<double>("maxSignificance"));
    bool resetHistogram = config.getParameter<bool>("resetHistograms");
    ESHandle<TrackProbabilityCalibration> calib2DHandle;
    iSetup.get<BTagTrackProbability2DRcd>().get(calib2DHandle);
    ESHandle<TrackProbabilityCalibration> calib3DHandle;
    iSetup.get<BTagTrackProbability3DRcd>().get(calib3DHandle);
    const TrackProbabilityCalibration * ca[2];
    ca[0]  = calib3DHandle.product();
    ca[1]  = calib2DHandle.product();
    for(unsigned int i=minLoop;i <=maxLoop ;i++)
    for(unsigned int j=0;j<ca[i]->data.size() ; j++)
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


// ------------ method called once each job just before starting event loop  ------------
void 
ImpactParameterCalibration::beginJob()
{
  using namespace edm;
  m_calibration[0] =   new TrackProbabilityCalibration();
  m_calibration[1] =   new TrackProbabilityCalibration();

  CalibratedHistogram hist(config.getParameter<int>("nBins"),0,config.getParameter<double>("maxSignificance"));

  std::string categories = config.getParameter<std::string>("inputCategories");

  if(categories == "HardCoded")
  {
   vector<TrackProbabilityCategoryData> v;
    //TrackProbabilityCategoryData {pMin, pMax, etaMin, etaMax,
    //nHitsMin, nHitsMax, nPixelHitsMin, nPixelHitsMax, chiMin,chiMax, withFirstPixel;
    //trackQuality;
  v.push_back(createCategory(0, 5000, 0  , 2.5, 8 , 50, 1, 1, 0  , 5  , 0));
  v.push_back(createCategory(0, 5000, 0  , 2.5, 8 , 50, 2, 8, 2.5, 5  , 0));
  v.push_back(createCategory(0, 8   , 0  , 0.8, 8 , 50, 3, 8, 0  , 2.5, 0));
  v.push_back(createCategory(0, 8   , 0.8, 1.6, 8 , 50, 3, 8, 0  , 2.5, 0));
  v.push_back(createCategory(0, 8   , 1.6, 2.5, 8 , 50, 3, 8, 0  , 2.5, 0));
  v.push_back(createCategory(0, 8   , 0  , 2.5, 8 , 50, 2, 8, 0  , 2.5, 0));
  v.push_back(createCategory(8, 5000, 0  , 0.8, 8 , 50, 3, 8, 0  , 2.5, 0));
  v.push_back(createCategory(8, 5000, 0.8, 1.6, 8 , 50, 3, 8, 0  , 2.5, 0));
  v.push_back(createCategory(8, 5000, 1.6, 2.5, 8 , 50, 3, 8, 0  , 2.5, 0));
  v.push_back(createCategory(8, 5000, 0  , 2.5, 8 , 50, 2 ,2, 0  , 2.5, 0));
  for(unsigned int i=minLoop;i <=maxLoop ;i++)
   for(unsigned int j=0;j<v.size() ; j++)
    {
     TrackProbabilityCalibration::Entry e;
     e.category=v[j];
     e.histogram=hist;
     m_calibration[i]->data.push_back(e);
    }

  }
  if(categories == "RootXML")
  {
    bool resetHistogram = config.getParameter<bool>("resetHistograms");
    const TrackProbabilityCalibration * ca[2];
    ca[0]  = fromXml(config.getParameter<edm::FileInPath>("calibFile3d"));
    ca[1]  = fromXml(config.getParameter<edm::FileInPath>("calibFile2d"));
  
    for(unsigned int i=minLoop;i <=maxLoop ;i++)
     for(unsigned int j=0;j<ca[i]->data.size() ; j++)
     {
      TrackProbabilityCalibration::Entry e;
      e.category=ca[i]->data[j].category;

      if(resetHistogram)
       e.histogram=hist;
      else
       e.histogram=ca[i]->data[j].histogram;

      m_calibration[i]->data.push_back(e);
     }

   delete ca[0];
   delete ca[1];

   }
  if(categories == "EventSetup")
   {
    m_needInitFromES=true;
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

TrackProbabilityCalibration * ImpactParameterCalibration::fromXml(edm::FileInPath xmlCalibration)   
{
     std::ifstream xmlFile(xmlCalibration.fullPath().c_str());
        if (!xmlFile.good())
                throw cms::Exception("BTauFakeMVAJetTagConditions")
                        << "File \"" << xmlCalibration.fullPath()
                        << "\" could not be opened for reading."
                        << std::endl;
        std::ostringstream ss;
        ss << xmlFile.rdbuf();
        xmlFile.close();
        TClass *classType = 0;
        void *ptr = TBufferXML(TBuffer::kRead).ConvertFromXMLAny(
                                ss.str().c_str(), &classType, kTRUE, kFALSE);
        if (!ptr)
                throw cms::Exception("ImpactParameterCalibration")
                        << "Unknown error parsing XML serialization"
                        << std::endl;

        if (std::strcmp(classType->GetName(),
                "TrackProbabilityCalibration")) {
                classType->Destructor(ptr);
                throw cms::Exception("ImpactParameterCalibration")
                        << "Serialized object has wrong C++ type."
                        << std::endl;
        }

        return static_cast<TrackProbabilityCalibration*>(ptr);
}




// ------------ method called once each job just after ending the event loop  ------------
void 
ImpactParameterCalibration::endJob() {

  if(config.getParameter<bool>("writeToDB"))
  {
    edm::Service<cond::service::PoolDBOutputService> mydbservice;
    if( !mydbservice.isAvailable() ) return;
    if(minLoop == 0 )  mydbservice->createNewIOV<TrackProbabilityCalibration>(m_calibration[0], mydbservice->beginOfTime(), mydbservice->endOfTime(),"BTagTrackProbability3DRcd");
    if(maxLoop == 1)   mydbservice->createNewIOV<TrackProbabilityCalibration>(m_calibration[1],  mydbservice->beginOfTime(), mydbservice->endOfTime(),"BTagTrackProbability2DRcd");
  } 
    

  if(config.getParameter<bool>("writeToRootXML"))
  {
    if(maxLoop == 1 ){
      std::ofstream of2("2d.xml");
      TBufferXML b2(TBuffer::kWrite);
      of2 << b2.ConvertToXML(static_cast<void*>(m_calibration[1]),
			     TClass::GetClass("TrackProbabilityCalibration"),
			     kTRUE, kFALSE);
      of2.close();
    }
    if(minLoop == 0 ){
      std::ofstream of3("3d.xml");
      TBufferXML b3(TBuffer::kWrite);
      of3 << b3.ConvertToXML(static_cast<void*>(m_calibration[0]),
			     TClass::GetClass("TrackProbabilityCalibration"),
			     kTRUE, kFALSE);
      of3.close();
    }
  }

 
  if(config.getParameter<bool>("writeToBinary"))
  {
    if(maxLoop == 1 ){
      
      std::ofstream ofile("2d.dat");
      TBufferFile buffer(TBuffer::kWrite);
      buffer.StreamObject(static_cast<void*>(m_calibration[1]),
			  TClass::GetClass("TrackProbabilityCalibration"));
      Int_t size = buffer.Length();
      ofile.write(buffer.Buffer(),size);
      ofile.close();
    }
    if(minLoop == 0 ){
      std::ofstream ofile3("3d.dat");
      TBufferFile buffer3(TBuffer::kWrite);
      buffer3.StreamObject(static_cast<void*>(m_calibration[0]),
			   TClass::GetClass("TrackProbabilityCalibration"));
      Int_t size3 = buffer3.Length();
      ofile3.write(buffer3.Buffer(),size3);
      ofile3.close();
    }
  }
  
  



}

//define this as a plug-in
DEFINE_FWK_MODULE(ImpactParameterCalibration);
