// -*- C++ -*-
//
// Package:    SiPixelCalibConfigurationReadDb
// Class:      SiPixelCalibConfigurationReadDb
// 
/**\class SiPixelCalibConfigurationReadDb SiPixelCalibConfigurationReadDb.cc CalibTracker/SiPixelTools/test/SiPixelCalibConfigurationReadDb.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Freya Blekman
//         Created:  Thu Sep 20 12:13:20 CEST 2007
// $Id: SiPixelCalibConfigurationReadDb.cc,v 1.3 2010/01/07 19:07:31 heyburn Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/SiPixelCalibConfigurationRcd.h"
#include <iostream>
//
// class decleration
//

class SiPixelCalibConfigurationReadDb : public edm::EDAnalyzer {
   public:
      explicit SiPixelCalibConfigurationReadDb(const edm::ParameterSet&);
      ~SiPixelCalibConfigurationReadDb();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
  bool verbose_;
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
SiPixelCalibConfigurationReadDb::SiPixelCalibConfigurationReadDb(const edm::ParameterSet& iConfig):
  verbose_(iConfig.getParameter<bool>("verbosity"))

{
   //now do what ever initialization is needed

}


SiPixelCalibConfigurationReadDb::~SiPixelCalibConfigurationReadDb()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
SiPixelCalibConfigurationReadDb::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   LogInfo("") << " examining SiPixelCalibConfiguration database object..." << std::endl;

   ESHandle<SiPixelCalibConfiguration> calib;
   iSetup.get<SiPixelCalibConfigurationRcd>().get(calib);
   std::cout << "calibration type: " << calib->getCalibrationMode() << std::endl;
   std::cout << "number of triggers: " << calib->getNTriggers() << std::endl;
   std::vector<short> vcalvalues= calib->getVCalValues();
   std::cout << "number of VCAL: " << vcalvalues.size() << std::endl;
   int ngoodcols=0;
   int ngoodrows=0;
   for(uint32_t i=0; i<vcalvalues.size(); ++i){
     if(verbose_){
       std::cout << "Vcal values " << i << "," << i+1 << " : " << vcalvalues[i] << "," ;
     }
     ++i;
     if(verbose_){
       if(i<vcalvalues.size())
	 std::cout << vcalvalues[i];
       std::cout << std::endl;
     }
   }
   if(verbose_)
     std::cout << "column patterns:" << std::endl;
   for(uint32_t i=0; i<calib->getColumnPattern().size(); ++i){
     if(calib->getColumnPattern()[i]!=-1){
       if(verbose_)
	 std::cout << calib->getColumnPattern()[i] ;
       ngoodcols++;
     }
     if(verbose_){
       if(i!=0)
	 std::cout << " ";
       if(calib->getColumnPattern()[i]==-1)
	 std::cout << "- " ;
     } 
   }
   if(verbose_){
     std::cout << std::endl;
     std::cout << "row patterns:" << std::endl;
   }
   for(uint32_t i=0; i<calib->getRowPattern().size(); ++i){
     if(calib->getRowPattern()[i]!=-1){
       if(verbose_)
	 std::cout << calib->getRowPattern()[i] ;
       ngoodrows++;
     }
     if(verbose_){
       if(i!=0)
	 std::cout << " ";
       if(calib->getRowPattern()[i]==-1)
	 std::cout << "- ";
     }
   }
   if(verbose_){
     std::cout << std::endl;
     std::cout << "number of row patterns: " << ngoodrows << std::endl;
     std::cout << "number of column patterns: " << ngoodcols << std::endl;
   }
   std::cout << "this payload is designed to run on " << vcalvalues.size()*ngoodcols*ngoodrows*calib->getNTriggers() << " events." << std::endl;
}


// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelCalibConfigurationReadDb::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelCalibConfigurationReadDb::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelCalibConfigurationReadDb);
