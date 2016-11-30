// -*- C++ -*-
//
// Package:    WriteL1TriggerObjetsXml
// Class:      WriteL1TriggerObjetsXml
// 
/**\class WriteL1TriggerObjetsXml WriteL1TriggerObjetsXml.cc Test/WriteL1TriggerObjetsXml/src/WriteL1TriggerObjetsXml.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ka Vang TSANG
//         Created:  Fri Jul 31 15:18:53 CEST 2009
//
//


// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/HcalL1TriggerObjectsXml.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
XERCES_CPP_NAMESPACE_USE 
//
// class decleration
//

class WriteL1TriggerObjetsXml : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit WriteL1TriggerObjetsXml(const edm::ParameterSet&);
  ~WriteL1TriggerObjetsXml();


private:
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override {}

  // ----------member data ---------------------------
  std::string tagname_;
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
WriteL1TriggerObjetsXml::WriteL1TriggerObjetsXml(const edm::ParameterSet& iConfig) : tagname_(iConfig.getParameter<std::string>("TagName"))
{
  //now do what ever initialization is needed
  
}


WriteL1TriggerObjetsXml::~WriteL1TriggerObjetsXml()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
WriteL1TriggerObjetsXml::analyze(edm::Event const& iEvent, 
				 edm::EventSetup const& iSetup) {
  edm::ESHandle<HcalTopology> htopo;
  iSetup.get<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo = htopo.product();
  edm::ESHandle<HcalDbService> conditions;
  iSetup.get<HcalDbRecord>().get(conditions);

  HcalSubdetector subDet[3] = {HcalBarrel, HcalEndcap, HcalForward};
  std::string subDetName[3] = {"HB", "HE", "HF"};
  int maxEta(0), maxPhi(72), maxDepth(0);
  const HcalDDDRecConstants* hcons = topo->dddConstants();
  for (int isub=0; isub<3; ++isub) {
    if (hcons->getMaxDepth(isub) > maxDepth) maxDepth = hcons->getMaxDepth(isub);
    if (hcons->getEtaRange(isub).second > maxEta) 
      maxEta  = hcons->getEtaRange(isub).second;
    if (hcons->getNPhi(isub) > maxPhi)       maxPhi  = hcons->getNPhi(isub);
  }

  HcalL1TriggerObjectsXml xml(tagname_);
  for (int isub = 0; isub < 3; ++isub){
    for (int ieta = -maxEta; ieta <= maxEta; ++ieta){
      for (int iphi = 1; iphi <=maxPhi; ++iphi){
	for (int depth = 1; depth <= maxDepth; ++depth){
	  HcalDetId id(subDet[isub], ieta, iphi, depth);

	  if (!topo->valid(id)) continue;
	  HcalCalibrations calibrations = conditions->getHcalCalibrations(id);
	  const HcalChannelStatus* channelStatus = conditions->getHcalChannelStatus(id);
	  uint32_t status = channelStatus->getValue();

	  double gain = 0.0;
	  double ped = 0.0;
	       
	  for (int i=0; i<4; ++i) {
	    gain += calibrations.LUTrespcorrgain(i);
	    ped += calibrations.pedestal(i);
	  }
	  gain /= 4.;
	  ped /= 4.;

	  xml.add_hcal_channel_dataset(ieta, iphi, depth, subDetName[isub], ped, gain, status);
	}// for depth
      }// for iphi
    }// for ieta
  }// for subdet

  std::string xmlOutputFileName(tagname_);
  xmlOutputFileName += ".xml";
  xml.write(xmlOutputFileName);
}

//define this as a plug-in
DEFINE_FWK_MODULE(WriteL1TriggerObjetsXml);
