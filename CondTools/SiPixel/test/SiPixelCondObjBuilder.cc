#include <memory>
#include <string>
#include <iostream>
#include "CondTools/SiPixel/test/SiPixelCondObjBuilder.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "SimTracker/SiPixelDigitizer/interface/PixelIndices.h"
#include "CLHEP/Random/RandGauss.h"

namespace cms{
SiPixelCondObjBuilder::SiPixelCondObjBuilder(const edm::ParameterSet& iConfig) :
      conf_(iConfig),
      appendMode_(conf_.getUntrackedParameter<bool>("appendMode",true)),
      meanPed_(conf_.getParameter<double>("meanPed")),
      rmsPed_(conf_.getParameter<double>("rmsPed")),
      meanGain_(conf_.getParameter<double>("meanGain")),
      rmsGain_(conf_.getParameter<double>("rmsGain")),
      numberOfModules_(conf_.getParameter<int>("numberOfModules")),
      fromFile_(conf_.getParameter<bool>("fromFile")),
      fileName_(conf_.getParameter<std::string>("fileName"))
{
}

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
       
       // Stop if module limit reached
       nmodules++;
       if( nmodules > numberOfModules_ ) break;

       const PixelGeomDetUnit * pixDet  = dynamic_cast<const PixelGeomDetUnit*>((*it));
       const PixelTopology & topol = pixDet->specificTopology();       
       // Get the module sizes.
       int nrows = topol.nrows();      // rows in x
       int ncols = topol.ncolumns();   // cols in y
       std::cout << " ---> PIXEL DETID " << detid << " Cols " << ncols << " Rows " << nrows << std::endl;

       pIndexConverter_ = new PixelIndices( ncols , nrows );

       std::vector<char> theSiPixelGainCalibration;

       // Loop over columns and rows of this DetID
       for(int i=0; i<ncols; i++) {
	 for(int j=0; j<nrows; j++) {
	   nchannels++;
	   
	   float ped = 0.0, gain = 0.0;

	   if( fromFile_ ) {

	     // Use calibration from a file
	     int chipIndex=0, colROC=0, rowROC=0;
	     
	     pIndexConverter_->transformToROC( i , j ,chipIndex,colROC,rowROC);
	     int chanROC = PixelIndices::pixelToChannelROC(rowROC,colROC); // use ROC coordinates
	     float pp0=0, pp1=0;
	     std::map<int,CalParameters,std::less<int> >::const_iterator it=calmap_.find(chanROC);
	     CalParameters theCalParameters  = (*it).second;
	     ped  = theCalParameters.p0;
	     gain = theCalParameters.p1;

	   } else {
	     ped  = RandGauss::shoot( meanPed_  , rmsPed_  );
	     gain = RandGauss::shoot( meanGain_ , rmsGain_ );
	   }

// 	   if(i==mycol && j==myrow) {
	   //	     std::cout << "       Col "<<i<<" Row "<<j<<" Ped "<<ped<<" Gain "<<gain<<std::endl;
// 	   }

	   // Encode to use the full 8-bit range
	   float theEncodedPed  = ped*4.25;
	   float theEncodedGain = (gain-0.3)*2560.;

	   // Insert data in the container
	   SiPixelGainCalibration_->setData( theEncodedPed, theEncodedGain, theSiPixelGainCalibration);
	 }
       }

       delete pIndexConverter_;

       SiPixelGainCalibration::Range range(theSiPixelGainCalibration.begin(),theSiPixelGainCalibration.end());
       if( !SiPixelGainCalibration_->put(detid,range,ncols) )
	 edm::LogError("SiPixelCondObjBuilder")<<"[SiPixelCondObjBuilder::analyze] detid already exists"<<std::endl;
     }
   }
   std::cout << " ---> PIXEL Modules  " << nmodules  << std::endl;
   std::cout << " ---> PIXEL Channels " << nchannels << std::endl;

 //   // Try to read object
//    int mynmodules =0;
//    for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
//      if( dynamic_cast<PixelGeomDetUnit*>((*it))!=0){
//        uint32_t mydetid=((*it)->geographicalId()).rawId();
//        mynmodules++;
//        if( mynmodules > numberOfModules_) break;
//        SiPixelGainCalibration::Range myrange = SiPixelGainCalibration_->getRange(mydetid);
//        float mypedestal = SiPixelGainCalibration_->getPed (mycol,myrow,myrange,416);
//        float mygain     = SiPixelGainCalibration_->getGain(mycol,myrow,myrange,416);
//        //std::cout<<" PEDESTAL "<< mypedestal<<" GAIN "<<mygain<<std::endl; 
//      }
//    }

   // Write into DB
   edm::Service<cond::service::PoolDBOutputService> mydbservice;

   if( mydbservice.isAvailable() ){
     try{
       size_t callbackToken = mydbservice->callbackToken("SiPixelGainCalibration");
       edm::LogInfo("SiPixelCondObjBuilder")<<"CallbackToken SiPixelGainCalibration "<<callbackToken<<std::endl;

       unsigned long long tillTime;

       if ( appendMode_)
	 tillTime = mydbservice->currentTime();
       else
	 tillTime = mydbservice->endOfTime();
       
       edm::LogInfo("SiPixelCondObjBuilder")<<"[SiPixelCondObjBuilder::analyze] tillTime = "<<tillTime<<std::endl;

       mydbservice->newValidityForNewPayload<SiPixelGainCalibration>(SiPixelGainCalibration_, tillTime , callbackToken);
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
SiPixelCondObjBuilder::beginJob(const edm::EventSetup&) {
  if(fromFile_) {
    if( loadFromFile() ) {
      edm::LogInfo("SiPixelCondObjBuilder")<<" Calibration loaded: Map size " 
					   <<calmap_.size() <<" max "
					   <<calmap_.max_size() << " "
					   <<calmap_.empty()<<std::endl;
    }
  }

}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelCondObjBuilder::endJob() {
}

bool SiPixelCondObjBuilder::loadFromFile() {
  
  float par0,par1,par2,par3;
  int rocid,colid,rowid;
  std::string name;
  
  std::ifstream in_file;  // data file pointer
  in_file.open( fileName_.c_str() , std::ios::in ); // in C++
  if (in_file.bad()) {
    edm::LogError("SiPixelCondObjBuilder")<<"Input file not found"<<std::endl;
  }
  if ( in_file.eof() != 0 ) {
    edm::LogError("SiPixelCondObjBuilder")<< in_file.eof() << " " << in_file.gcount() << " "
	   << in_file.fail() << " " << in_file.good() << " end of file "
					  << std::endl;
    return false;
  }
  //Load file header
  char line[500];
  for (int i = 0; i < 3; i++) {
    in_file.getline(line, 500,'\n');
    edm::LogInfo("SiPixelCondObjBuilder")<<line<<std::endl;
  }
  //Loading calibration constants from file, loop on pixels
  for(int i=0;i<(52*80);i++)  {
    in_file >> par0 >> par1 >> name >> colid >> rowid;

    std::cout<<" Col "<<colid
	     <<" Row "<<rowid
	     <<" P0 "<<par0
	     <<" P1 "<<par1
	     <<std::endl; 
    
    CalParameters onePix;
    onePix.p0=par0;
    onePix.p1=par1;
    
    // Convert ROC pixel index to channel 
    int chan = PixelIndices::pixelToChannelROC(rowid,colid);
    calmap_.insert(std::pair<int,CalParameters>(chan,onePix));
  }
  
  bool flag;
  if(calmap_.size()==4160) {
    flag = true; 
  } else {
    flag = false;
  }
  return flag;
  
}

}
