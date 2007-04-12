// -*- C++ -*-
//
// Package:    SiPixelGainCalibration
// Class:      SiPixelGainCalibration
// 
/**\class SiPixelGainCalibration SiPixelGainCalibration.cc SiPixelGainCalibration/SiPixelGainCalibration/src/SiPixelGainCalibration.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Anders Ryd
//         Created:  Thu Feb 22 23:12:36 EST 2007
// $Id$
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibTracker/SiPixelGainCalibration/interface/PixelCalib.h"
#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalib.h"


//
// class declaration
//

class SiPixelGainCalibration : public edm::EDAnalyzer {
   public:
      explicit SiPixelGainCalibration(const edm::ParameterSet&);
      ~SiPixelGainCalibration();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;


      // ----------member data ---------------------------

      PixelCalib* calib_;
      //for now assume only on fed_id!
      PixelROCGainCalib rocgain_[36][24];

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SiPixelGainCalibration::SiPixelGainCalibration(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  calib_=0;
}


SiPixelGainCalibration::~SiPixelGainCalibration()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
SiPixelGainCalibration::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   unsigned int vcal=100;

   Handle<FEDRawDataCollection> fedRawDataCollection;
   iEvent.getByType(fedRawDataCollection);

   //I know that we are the pixels, fedid=0-39, but this
   //should be defined in some headerfile.

   for (unsigned int fed_id=0;fed_id<40;fed_id++){
     
     const FEDRawData& fedRawData = fedRawDataCollection->FEDData( fed_id );

     //edm::LogVerbatim("") << "fedid="<<fed_id<<" size="<<fedRawData.size()<<std::endl;

     unsigned int datasize=fedRawData.size();

     const unsigned char* dataptr=fedRawData.data();

     for( unsigned int i=0;i<datasize/8;i++ ){

       unsigned long long data=((const unsigned long long*)dataptr)[i];
       //edm::LogVerbatim("") << "i="<<i<<" data="<<std::hex<<data<<std::dec<<std::endl;

       if ((data >> 60) == 0x5) continue;
       if ((data >> 60) == 0xa) continue;

       unsigned int roc_id_old=0; 
       unsigned int pix_id_old=0; 
       unsigned int dcol_id_old=0; 

       for (unsigned ihit=0;ihit<2;ihit++){
	 unsigned int hitdata=(data>>(ihit*32))&(0xffffffff);
	 if (hitdata==0xffffffff) continue;
	 if (hitdata==1) continue;
	 unsigned int fed_channel=(hitdata>>26)&0x3f;
	 unsigned int roc_id=(hitdata>>21)&0x1f;
	 unsigned int dcol_id=(hitdata>>16)&0x1f;
	 unsigned int pix_id=(hitdata>>8)&0xff;
	 unsigned int adc=hitdata&0xff;
         unsigned int row=80-pix_id/2;
         unsigned int col=dcol_id*2+pix_id%2;

	 /*	 
         edm::LogVerbatim("") << " fed_channel="<<fed_channel
			      << " roc_id="<<roc_id
			      << " dcol_id="<<dcol_id
			      << " pix_id="<<pix_id
			      << " col="<<col
			      << " row="<<row;

	 if (roc_id_old==roc_id){
	   if (dcol_id<dcol_id_old) {
	     //throw cms::Exception("DcolOrderAssertion")<<"Invalid pixel hit ";
	     edm::LogVerbatim("") << " OLD:<< "
				  << " roc_id="<<roc_id_old
				  << " dcol_id="<<dcol_id_old
				  << " row="<<pix_id_old;
	     edm::LogVerbatim("") << " NEW:<< "
				  << " roc_id="<<roc_id
				  << " dcol_id="<<dcol_id
				  << " row="<<row;

	   }
	   if (dcol_id==dcol_id_old){
	     if (row<=pix_id_old) {
	       //throw cms::Exception("PixOrderAssertion")<<"Invalid pixel hit ";
	       edm::LogVerbatim("") << " OLD:<< "
				    << " roc_id="<<roc_id_old
				    << " dcol_id="<<dcol_id_old
				    << " row="<<pix_id_old;
	       edm::LogVerbatim("") << " NEW:<< "
				    << " roc_id="<<roc_id
				    << " dcol_id="<<dcol_id
				    << " row="<<row;

	     }
	   }
	 }
	 roc_id_old=roc_id;
	 dcol_id_old=dcol_id;
         pix_id_old=row;


	 */

	 if( row>=80 || col >=52 || roc_id > 24 || fed_channel > 36 ) {
	   edm::LogVerbatim log("");
	   
	   log << " hitdata="<<std::hex<<hitdata<<std::dec
	       << " fed_channel="<<fed_channel
	       << " roc_id="<<roc_id
	       << " pix_id="<<pix_id
	       << " dcol_id="<<dcol_id;
	   log << " row="<<row
	       << " col="<<col
	       << " adc="<<adc
	       << std::endl;
	   throw cms::Exception("AssertFailure")<<"Invalid pixel hit ";
	 }
	 
	 rocgain_[fed_channel-1][roc_id-1].fill(row,col,vcal,adc);

       }       
       
     }

   }

}


// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelGainCalibration::beginJob(const edm::EventSetup&)
{

  edm::LogVerbatim("") << "In SiPixelGainCalibration::beginJob" << std::dec;

  calib_=new PixelCalib("/afs/cern.ch/cms/Tracker/Pixel/forward/ryd/calib_070106d.dat");

  int nvcal=calib_->nVcal();

  for(unsigned int linkid=0;linkid<36;linkid++){
    for(unsigned int rocid=0;rocid<24;rocid++){
      rocgain_[linkid][rocid].init(linkid,rocid,nvcal);
    }
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelGainCalibration::endJob() {


  edm::LogVerbatim("") << "In SiPixelGainCalibration::endJob" << std::dec;

    

  if (calib_!=0) delete calib_; calib_=0;

}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelGainCalibration)
