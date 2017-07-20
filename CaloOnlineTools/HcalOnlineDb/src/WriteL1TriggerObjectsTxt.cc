// -*- C++ -*-
//
// Package:    WriteL1TriggerObjectsTxt
// Class:      WriteL1TriggerObjectsTxt
// 
/**\class WriteL1TriggerObjectsTxt WriteL1TriggerObjectsTxt.cc HcalTools/WriteL1TriggerObjectsTxt/src/WriteL1TriggerObjectsTxt.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ka Vang TSANG
//         Created:  Wed Aug  5 15:23:29 CEST 2009
// $Id: WriteL1TriggerObjectsTxt.cc,v 1.1 2010/07/15 10:40:22 kvtsang Exp $
//
//


// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalL1TriggerObjects.h"
#include "CondFormats/HcalObjects/interface/HcalL1TriggerObject.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

//
// class decleration
//

class WriteL1TriggerObjectsTxt : public edm::EDAnalyzer {
   public:
      explicit WriteL1TriggerObjectsTxt(const edm::ParameterSet&);
      ~WriteL1TriggerObjectsTxt();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      std::string tagName_;
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
WriteL1TriggerObjectsTxt::WriteL1TriggerObjectsTxt(const edm::ParameterSet& iConfig) :
   tagName_(iConfig.getParameter<std::string>("TagName"))
{
   //now do what ever initialization is needed

}


WriteL1TriggerObjectsTxt::~WriteL1TriggerObjectsTxt()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
WriteL1TriggerObjectsTxt::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;

    edm::ESHandle<HcalDbService> conditions;
    iSetup.get<HcalDbRecord>().get(conditions);

    const HcalLutMetadata *metadata = conditions->getHcalLutMetadata();
    const HcalTopology *topo = metadata->topo();

    std::auto_ptr<HcalL1TriggerObjects> HcalL1TrigObjCol(new HcalL1TriggerObjects);

    for (const auto& id: metadata->getAllChannels()) {
	if (not (id.det() == DetId::Hcal and topo->valid(id))) continue;
     
	HcalDetId cell(id);
	HcalSubdetector subdet = cell.subdet();
	if (subdet != HcalBarrel and subdet != HcalEndcap and subdet != HcalForward) continue;

           HcalCalibrations calibrations = conditions->getHcalCalibrations(cell);

           float gain = 0.0;
           float ped = 0.0;
           for (int i=0; i<4; ++i) {
             gain += calibrations.LUTrespcorrgain(i);
             ped += calibrations.pedestal(i);
           }

           gain /= 4.;
           ped /= 4.;

           const HcalChannelStatus* channelStatus = conditions->getHcalChannelStatus(cell);
           uint32_t status = channelStatus->getValue();
           HcalL1TriggerObject l1object(cell, ped, gain, status);
	   HcalL1TrigObjCol->setTopo(topo);
           HcalL1TrigObjCol->addValues(l1object);

   } 
   HcalL1TrigObjCol->setTagString(tagName_);
   HcalL1TrigObjCol->setAlgoString("A 2-TS Peak Finder");
   std::string outfilename = "Dump_L1TriggerObjects_";
   outfilename += tagName_;
   outfilename += ".txt";
   std::ofstream of(outfilename.c_str());
   HcalDbASCIIIO::dumpObject(of, *HcalL1TrigObjCol);
}


// ------------ method called once each job just before starting event loop  ------------
void 
WriteL1TriggerObjectsTxt::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
WriteL1TriggerObjectsTxt::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(WriteL1TriggerObjectsTxt);
