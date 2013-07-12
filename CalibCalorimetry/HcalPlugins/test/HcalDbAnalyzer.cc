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
// $Id: HcalDbAnalyzer.cc,v 1.22 2008/11/10 10:13:20 rofierzy Exp $
//
//


// system include files
#include <memory>
#include <time.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

#include "CalibFormats/HcalObjects/interface/HcalText2DetIdConverter.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"

namespace {
  std::ostream& operator<<(std::ostream& fOut, const DetId& id) {
    if (id.null ()) fOut << "NULL";
    else if (id.det() == DetId:: Calo && id.subdetId()==HcalZDCDetId::SubdetectorId) {
      fOut << HcalZDCDetId (id);
    }
    else if (id.det () != DetId::Hcal)  fOut << "Not HCAL";
    else {
      HcalSubdetector sub = HcalSubdetector (id.subdetId());
      if (sub == HcalBarrel || sub == HcalEndcap || sub == HcalOuter || sub == HcalForward)
	fOut << HcalDetId (id);
      else if (sub == HcalEmpty) fOut << "EMPTY";
      else if (sub == HcalTriggerTower) fOut << HcalTrigTowerDetId (id);
      else if (sub == HcalOther) {
	HcalOtherDetId osub (id);
	HcalOtherSubdetector odetid = osub.subdet();
	if (odetid == HcalCalibration)  fOut << HcalCalibDetId (id);
	else fOut << "Unknown subtype";
      }
      else fOut << "Unknown type";
    }
    return fOut;
  }
}

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
  std::cout << "HcalDbAnalyzer::HcalDbAnalyzer->..." << std::endl;
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
  std::cout << "HcalDbAnalyzer::analyze->..." << std::endl;
  edm::ESHandle<HcalDbService> pSetup;
  iSetup.get<HcalDbRecord>().get( pSetup );
  std::cout << "HcalDbAnalyzer::analyze-> got HcalDbRecord: " << std::endl;
  std::cout << "HcalDbAnalyzer::analyze-> getting information for HB channel eta=1, phi=1, depth=1..." << std::endl;
  HcalDetId cell (HcalBarrel, 1, 1, 1);
  
  const HcalCalibrations& calibrations=pSetup->getHcalCalibrations(cell);
  const HcalCalibrationWidths widths = pSetup->getHcalCalibrationWidths(cell);
  const HcalQIECoder* coder = pSetup->getHcalCoder (cell);
  const HcalQIEShape* shape = pSetup->getHcalShape (cell);
  
  std::cout << "Values-> pedestals: " 
	    << calibrations.pedestal (0) << '/'
	    << calibrations.pedestal (1) << '/'
	    << calibrations.pedestal (2) << '/'
	    << calibrations.pedestal (3)
	    << ",  gains: "
	    << calibrations.rawgain (0) << '/'
	    << calibrations.rawgain (1) << '/'
	    << calibrations.rawgain (2) << '/'
	    << calibrations.rawgain (3)
	    << std::endl;
  std::cout << "Widths. pedestals: " 
	    << widths.pedestal (0) << '/'
	    << widths.pedestal (1) << '/'
	    << widths.pedestal (2) << '/'
	    << widths.pedestal (3)
	    << ",  gains: "
	    << widths.gain (0) << '/'
	    << widths.gain (1) << '/'
	    << widths.gain (2) << '/'
	    << widths.gain (3) 
	    << std::endl;
  
  std::cout << "QIE shape:" << std::endl;
  for (int i = 0; i < 128; i++) {
    double q0 = coder->charge (*shape, i, 0);
    double q1 = coder->charge (*shape, i, 1);
    double q2 = coder->charge (*shape, i, 2);
    double q3 = coder->charge (*shape, i, 3);

    unsigned adc0 = coder->adc (*shape, q0, 0);
    unsigned adc1 = coder->adc (*shape, q1, 0);
    unsigned adc2 = coder->adc (*shape, q2, 0);
    unsigned adc3 = coder->adc (*shape, q3, 0);

    std::cout << " ADC: " << i << " q1:" << q0 << " q2:" << q1 << " q3:" << q2 << " q4:" << q3 << std::endl;
    std::cout << " reverse ADC: " << i << " q1:" << adc0 << " q2:" << adc1 << " q3:" << adc2 << " q4:" << adc3 << std::endl;
  }

  std::cout << "Electronics map: " << std::endl;
  const HcalElectronicsMap* emap=pSetup->getHcalMapping();
  std::vector<HcalElectronicsId> eids=emap->allElectronicsId();
  for (std::vector<HcalElectronicsId>::const_iterator j=eids.begin(); j!=eids.end(); ++j) {
    DetId did=emap->lookup(*j);
    std::cout << *j << " " << did << std::endl;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalDbAnalyzer);
