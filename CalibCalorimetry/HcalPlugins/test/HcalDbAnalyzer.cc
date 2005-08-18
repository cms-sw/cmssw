// -*- C++ -*-
//
// Package:    WhatsItAnalyzer
// Class:      WhatsItAnalyzer
// 
/**\class WhatsItAnalyzer WhatsItAnalyzer.cc test/WhatsItAnalyzer/src/WhatsItAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 24 19:13:25 EDT 2005
// $Id: WhatsItAnalyzer.cc,v 1.2 2005/07/14 22:20:57 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CalibFormats/HcalObjects/interface/HcalDbServiceHandle.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"


class HcalDbAnalyzer : public edm::EDAnalyzer {
   public:
      explicit HcalDbAnalyzer( const edm::ParameterSet& );
      ~HcalDbAnalyzer ();


      virtual void analyze( const edm::Event&, const edm::EventSetup& );
   private:
      // ----------member data ---------------------------
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
HcalDbAnalyzer::HcalDbAnalyzer( const edm::ParameterSet& iConfig )
{
   //now do what ever initialization is needed

}


HcalDbAnalyzer::~HcalDbAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
HcalDbAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   edm::eventsetup::ESHandle<HcalDbServiceHandle> pSetup;
   iSetup.get<HcalDbRecord>().get( pSetup );
   std::cout << "HcalDbAnalyzer::analyze-> got HcalDbRecord: " << pSetup->service()->name() << std::endl;
   std::cout << "HcalDbAnalyzer::analyze-> getting information for HB channel eta=1, phi=1, depth=1..." << std::endl;
   cms::HcalDetId cell (cms::HcalBarrel, 1, 1, 1);
   const HcalCalibrations* calibrations = pSetup->getHcalCalibrations (cell);
   const HcalCalibrationWidths* widths = pSetup->getHcalCalibrationWidths (cell);
   const HcalChannelCoder* coder = pSetup->getChannelCoder (cell);
   const QieShape* shape = pSetup->getBasicShape ();
   
   std::cout << "Values-> pedestals: " 
	     << calibrations->pedestal (0) << '/'
	     << calibrations->pedestal (1) << '/'
	     << calibrations->pedestal (2) << '/'
	     << calibrations->pedestal (3)
	     << ",  gains: "
	     << calibrations->gain (0) << '/'
	     << calibrations->gain (1) << '/'
	     << calibrations->gain (2) << '/'
	     << calibrations->gain (3)
	     << std::endl;
   std::cout << "Widths-> pedestals: " 
	     << widths->pedestal (0) << '/'
	     << widths->pedestal (1) << '/'
	     << widths->pedestal (2) << '/'
	     << widths->pedestal (3)
	     << ",  gains: "
	     << widths->gain (0) << '/'
	     << widths->gain (1) << '/'
	     << widths->gain (2) << '/'
	     << widths->gain (3) 
	     << std::endl;

   std::cout << "QIE shape:" << std::endl;
   for (int i = 0; i < 128; i++) {
     double q0 = coder->charge (*shape, i, 0);
     double q1 = coder->charge (*shape, i, 1);
     double q2 = coder->charge (*shape, i, 2);
     double q3 = coder->charge (*shape, i, 3);
     std::cout << ' ' << i << ':' << q0 << '/' << q1 << '/' << q2 << '/' << q3;
   }

}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalDbAnalyzer)
