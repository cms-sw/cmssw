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
// Original Author:  Jeremy Andrea
//         Created:  Wed Mar  5 19:17:38 CEST 2008
// $Id: SumHistoCalibration.cc,v 1.9 2012/07/05 15:07:03 eulisse Exp $
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


#include "RecoBTag/XMLCalibration/interface/AlgorithmCalibration.h"
#include "RecoBTag/XMLCalibration/interface/CalibratedHistogramXML.h"
#include "RecoBTag/TrackProbability/interface/TrackClassFilter.h"

#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"



#include "RVersion.h"
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,15,0)
#include "TBufferFile.h"
typedef TBufferFile MyTBuffer;
#else
#include "TBuffer.h"
typedef TBuffer MyTBuffer;
#endif

#include <TClass.h>
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



class SumHistoCalibration : public edm::EDAnalyzer {
   public:
      explicit SumHistoCalibration(const edm::ParameterSet&);
      ~SumHistoCalibration();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      edm::ParameterSet config;

   TrackProbabilityCalibration * fromXml(edm::FileInPath xmlCalibration);

  
  std::vector<std::string>  m_xmlilelist2d;
  std::vector<std::string>  m_xmlilelist3d;
  bool m_sum2D;
  bool m_sum3D;
  unsigned int minLoop, maxLoop;
  TrackProbabilityCalibration * m_calibration[2];
 

  
};

SumHistoCalibration::SumHistoCalibration(const edm::ParameterSet& iConfig):config(iConfig)
{
  m_xmlilelist2d   = iConfig.getParameter<std::vector<std::string> >("xmlfiles2d");
  m_xmlilelist3d   = iConfig.getParameter<std::vector<std::string> >("xmlfiles3d");
  m_sum2D          = iConfig.getParameter<bool>("sum2D");   
  m_sum3D          = iConfig.getParameter<bool>("sum3D");   
}


SumHistoCalibration::~SumHistoCalibration()
{
}


void
SumHistoCalibration::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;
  using namespace std;

}










// ------------ method called once each job just before starting event loop  ------------
void 
SumHistoCalibration::beginJob()
{
  if(m_sum2D && m_sum3D){minLoop = 0; maxLoop =1;}
  if(m_sum2D && !m_sum3D){minLoop = 0; maxLoop =0;}
  if(!m_sum2D && m_sum3D){minLoop = 1; maxLoop =1;}
  using namespace edm;
  m_calibration[0] =   new TrackProbabilityCalibration();
  m_calibration[1] =   new TrackProbabilityCalibration();
  
  const TrackProbabilityCalibration * ca[2];
  edm::FileInPath fip(m_xmlilelist3d[0]);
  edm::FileInPath fip2(m_xmlilelist2d[0]);
  ca[0]  = fromXml(fip);
  ca[1]  = fromXml(fip2);
  
  for(unsigned int i=minLoop;i <=maxLoop ;i++)
   for(unsigned int j=0;j<ca[i]->data.size() ; j++)
   {
    TrackProbabilityCalibration::Entry e;
    e.category=ca[i]->data[j].category;
    e.histogram=ca[i]->data[j].histogram;
    m_calibration[i]->data.push_back(e);
   }

   delete ca[0];
   delete ca[1];

  
  
  
  
  
  
  
  
  
  
  
  
}

TrackProbabilityCalibration * SumHistoCalibration::fromXml(edm::FileInPath xmlCalibration)   
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
    throw cms::Exception("SumHistoCalibration")
      << "Unknown error parsing XML serialization"
      << std::endl;
  
  if (std::strcmp(classType->GetName(),
		  "TrackProbabilityCalibration")) {
    classType->Destructor(ptr);
    throw cms::Exception("SumHistoCalibration")
      << "Serialized object has wrong C++ type."
                        << std::endl;
  }
  
  return static_cast<TrackProbabilityCalibration*>(ptr);
}




// ------------ method called once each job just after ending the event loop  ------------
void 
SumHistoCalibration::endJob() {
  
  
  using namespace edm;
  using namespace reco;
  using namespace std;

  
  
  
  if(m_sum3D){
    for(unsigned int itFile =1; itFile< m_xmlilelist3d.size(); itFile++){
      edm::FileInPath fip(m_xmlilelist3d[itFile]);
      const TrackProbabilityCalibration *ca  = fromXml(fip);
      for(unsigned int j=0;j<ca->data.size() ; j++)
	{
	  for(int k = 0; k< m_calibration[0]->data[j].histogram.numberOfBins(); k++){
	    m_calibration[0]->data[j].histogram.setBinContent(k, ca->data[j].histogram.binContent(k)
							      + m_calibration[0]->data[j].histogram.binContent(k));
	  }
	}
      delete ca;
    } 
  }
  if(m_sum2D){
    for(unsigned int itFile =1; itFile< m_xmlilelist2d.size(); itFile++){
      edm::FileInPath fip(m_xmlilelist2d[itFile]);
      TrackProbabilityCalibration * ca  = fromXml(fip);
      for(unsigned int j=0;j<ca->data.size() ; j++)
	{
	  for(int k = 0; k< m_calibration[1]->data[j].histogram.numberOfBins(); k++){
	    m_calibration[1]->data[j].histogram.setBinContent(k,ca->data[j].histogram.binContent(k)
							      + m_calibration[1]->data[j].histogram.binContent(k)); 
	  }
	}
      delete ca; 
    }
  }
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  if(config.getParameter<bool>("writeToDB"))
    {
      edm::Service<cond::service::PoolDBOutputService> mydbservice;
      if( !mydbservice.isAvailable() ) return;
      //mydbservice->createNewIOV<TrackProbabilityCalibration>(m_calibration[0],  mydbservice->endOfTime(),"BTagTrackProbability3DRcd");
      //mydbservice->createNewIOV<TrackProbabilityCalibration>(m_calibration[1],  mydbservice->endOfTime(),"BTagTrackProbability2DRcd"); 
      mydbservice->createNewIOV<TrackProbabilityCalibration>(m_calibration[0],  mydbservice->beginOfTime(), mydbservice->endOfTime(),"BTagTrackProbability3DRcd");
      mydbservice->createNewIOV<TrackProbabilityCalibration>(m_calibration[1],  mydbservice->beginOfTime(), mydbservice->endOfTime(),"BTagTrackProbability2DRcd");
      
    } 
  
  
  if(config.getParameter<bool>("writeToRootXML"))
    {
      std::ofstream of2("2d.xml");
      TBufferXML b2(TBuffer::kWrite);
      of2 << b2.ConvertToXML(static_cast<void*>(m_calibration[1]),
			     TClass::GetClass("TrackProbabilityCalibration"),
			     kTRUE, kFALSE);
      of2.close();
      std::ofstream of3("3d.xml");
      TBufferXML b3(TBuffer::kWrite);
      of3 << b3.ConvertToXML(static_cast<void*>(m_calibration[0]),
			     TClass::GetClass("TrackProbabilityCalibration"),
			     kTRUE, kFALSE);
      of3.close();
    }
  
  
  if(config.getParameter<bool>("writeToBinary"))
    {
      std::ofstream ofile("2d.dat");
      MyTBuffer buffer(TBuffer::kWrite);
      buffer.StreamObject(const_cast<void*>(static_cast<const void*>(m_calibration[1])),
			  TClass::GetClass("TrackProbabilityCalibration"));
      Int_t size = buffer.Length();
      ofile.write(buffer.Buffer(),size);
      ofile.close();
      
      std::ofstream ofile3("3d.dat");
      MyTBuffer buffer3(TBuffer::kWrite);
      buffer3.StreamObject(const_cast<void*>(static_cast<const void*>(m_calibration[0])),
			   TClass::GetClass("TrackProbabilityCalibration"));
      Int_t size3 = buffer3.Length();
      ofile3.write(buffer3.Buffer(),size3);
      ofile3.close();
    }
  
  
  
  
  
}

//define this as a plug-in
DEFINE_FWK_MODULE(SumHistoCalibration);
