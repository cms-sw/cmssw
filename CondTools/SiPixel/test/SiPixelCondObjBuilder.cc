// -*- C++ -*-
//
// Package:    SiPixelCondObjBuilder
// Class:      SiPixelCondObjBuilder
// 
/**\class SiPixelCondObjBuilder SiPixelCondObjBuilder.cc SiPixel/test/SiPixelCondObjBuilder.cc

 Description: Test analyzer for writing pixel calibration in the DB

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo CHIOCHIA
//         Created:  Tue Oct 17 17:40:56 CEST 2006
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h"

#include "CLHEP/Random/RandGauss.h"
//
// class decleration
//

class SiPixelCondObjBuilder : public edm::EDAnalyzer {
public:
  explicit SiPixelCondObjBuilder(const edm::ParameterSet&);
  ~SiPixelCondObjBuilder();
  
  
private:
  
  SiPixelGainCalibration* SiPixelGainCalibration_;
  
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------
};
//
// constructors and destructor
//
SiPixelCondObjBuilder::SiPixelCondObjBuilder(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

}


SiPixelCondObjBuilder::~SiPixelCondObjBuilder()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

// ------------ method called to for each event  ------------
void
SiPixelCondObjBuilder::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   unsigned int run=iEvent.id().run();
   unsigned int nmodules = 0;
   uint32_t nchannels = 0;
   int mycol = 415;
   int myrow = 159;


   edm::LogInfo("SiPixelCondObjBuilder") << "... creating dummy SiPixelGainCalibration Data for Run " << run << "\n " << std::endl;
   SiPixelGainCalibration* SiPixelGainCalibration_ = new SiPixelGainCalibration();

   edm::ESHandle<TrackerGeometry> pDD;
   iSetup.get<TrackerDigiGeometryRecord>().get( pDD );     
   edm::LogInfo("SiPixelCondObjBuilder") <<" There are "<<pDD->dets().size() <<" detectors"<<std::endl;
   
   for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
     if( dynamic_cast<PixelGeomDetUnit*>((*it))!=0){
       uint32_t detid=((*it)->geographicalId()).rawId();
       nmodules++;
       //if(nmodules>10) break;

       const PixelGeomDetUnit * pixDet  = dynamic_cast<const PixelGeomDetUnit*>((*it));
       const PixelTopology & topol = pixDet->specificTopology();       
       // Get the module sizes.
       int nrows = topol.nrows();      // rows in x
       int ncols = topol.ncolumns();   // cols in y
       std::cout << " ---> PIXEL DETID " << detid << " Cols " << ncols << " Rows " << nrows << std::endl;

       std::vector<char> theSiPixelGainCalibration;

       // Loop over columns and rows of this DetID
       for(int i=0; i<ncols; i++) {
	 for(int j=0; j<nrows; j++) {
	   nchannels++;
	   float MeanPed  = 100.0;
	   float RmsPed   =   5.0;
	   float MeanGain =  25.0;
	   float RmsGain  =   5.0;
	   
	   float ped  = RandGauss::shoot(MeanPed ,RmsPed);
	   float gain = RandGauss::shoot(MeanGain,RmsGain);

	   if(i==mycol && j==myrow) {
	     //std::cout << "       Col "<<i<<" Row "<<j<<" Ped "<<ped<<" Gain "<<gain<<std::endl;
	   }
	   // Insert data in the container
	   SiPixelGainCalibration_->setData( ped, gain, theSiPixelGainCalibration);
	 }
       }
       SiPixelGainCalibration::Range range(theSiPixelGainCalibration.begin(),theSiPixelGainCalibration.end());
       if( !SiPixelGainCalibration_->put(detid,range,ncols) )
	 edm::LogError("SiPixelCondObjBuilder")<<"[SiPixelCondObjBuilder::analyze] detid already exists"<<std::endl;
     }
   }
   std::cout << " ---> PIXEL Modules  " << nmodules  << std::endl;
   std::cout << " ---> PIXEL Channels " << nchannels << std::endl;
   // Try to read object
   int mynmodules =0;
   for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
     if( dynamic_cast<PixelGeomDetUnit*>((*it))!=0){
       uint32_t mydetid=((*it)->geographicalId()).rawId();
       mynmodules++;
       if(mynmodules>10) break;   
       SiPixelGainCalibration::Range myrange = SiPixelGainCalibration_->getRange(mydetid);
       float mypedestal = SiPixelGainCalibration_->getPed (mycol,myrow,myrange,416);
       float mygain     = SiPixelGainCalibration_->getGain(mycol,myrow,myrange,416);
       //std::cout<<" PEDESTAL "<< mypedestal<<" GAIN "<<mygain<<std::endl; 
     }
   }

   // Write into DB
   edm::Service<cond::service::PoolDBOutputService> mydbservice;

   if( mydbservice.isAvailable() ){
     try{
       size_t callbackToken = mydbservice->callbackToken("SiPixelGainCalibration");
       edm::LogInfo("SiPixelCondObjBuilder")<<"CallbackToken SiPixelGainCalibration"<<callbackToken<<std::endl;
       mydbservice->newValidityForNewPayload<SiPixelGainCalibration>(SiPixelGainCalibration_, mydbservice->currentTime(), callbackToken);
     }catch(const cond::Exception& er){
       edm::LogError("SiPixelCondObjBuilder")<<er.what()<<std::endl;
     }catch(const std::exception& er){
       edm::LogError("SiPixelCondObjBuilder")<<"caught std::exception "<<er.what()<<std::endl;
     }catch(...){
       edm::LogError("SiPixelCondObjBuilder")<<"Funny error"<<std::endl;
     }
    }else{
      edm::LogError("SiPixelCondObjBuilder")<<"Service is unavailable"<<std::endl;
    }    
   

}


// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelCondObjBuilder::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelCondObjBuilder::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelCondObjBuilder)
