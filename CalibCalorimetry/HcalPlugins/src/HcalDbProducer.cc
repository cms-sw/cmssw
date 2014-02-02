// -*- C++ -*-
//
// Package:    HcalDbProducer
// Class:      HcalDbProducer
// 
/**\class HcalDbProducer HcalDbProducer.h CalibFormats/HcalDbProducer/interface/HcalDbProducer.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Fedor Ratnikov
//         Created:  Tue Aug  9 19:10:10 CDT 2005
// $Id: HcalDbProducer.cc,v 1.28 2012/11/12 20:42:39 dlange Exp $
//
//


// system include files
#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include "CondFormats/HcalObjects/interface/AllObjects.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include "HcalDbProducer.h"

HcalDbProducer::HcalDbProducer( const edm::ParameterSet& fConfig)
  : ESProducer(),
    mService (new HcalDbService (fConfig)),
    mDumpRequest (),
    mDumpStream(0)
{
  //the following line is needed to tell the framework what data is being produced
  // comments of dependsOn:
  // 1) There are two ways one can use 'dependsOn' the first is passing it up to three arguments.  
  //    However, one can also extend the dependencies by first calling 'dependsOn() and then using '&' to add additional dependencies.  So
  //      dependsOn(&FooProd::func1, &FooProd::func2, &FooProd::func3)
  //    gives the same result as
  //      dependsOn(&FooProd::func1) & (&FooProd::func2) & (&FooProd::func3)
  // 2) Upon IOV change, all callbacks are called, in the inverse order of their specification below (tested).
  setWhatProduced (this, (dependsOn (&HcalDbProducer::pedestalsCallback) &
			  &HcalDbProducer::pedestalWidthsCallback &
			  &HcalDbProducer::respCorrsCallback &
			  &HcalDbProducer::gainsCallback &
			  &HcalDbProducer::LUTCorrsCallback &
			  &HcalDbProducer::PFCorrsCallback &
			  &HcalDbProducer::timeCorrsCallback &
			  &HcalDbProducer::QIEDataCallback &
			  &HcalDbProducer::gainWidthsCallback &
			  &HcalDbProducer::channelQualityCallback &
			  &HcalDbProducer::zsThresholdsCallback &
			  &HcalDbProducer::L1triggerObjectsCallback &
			  &HcalDbProducer::electronicsMapCallback &
			  &HcalDbProducer::lutMetadataCallback 
			  )
		   );
  
  //now do what ever other initialization is needed

  mDumpRequest = fConfig.getUntrackedParameter <std::vector <std::string> > ("dump", std::vector<std::string>());
  if (!mDumpRequest.empty()) {
    std::string otputFile = fConfig.getUntrackedParameter <std::string> ("file", "");
    mDumpStream = otputFile.empty () ? &std::cout : new std::ofstream (otputFile.c_str());
  }
}


HcalDbProducer::~HcalDbProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  if (mDumpStream != &std::cout) delete mDumpStream;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
boost::shared_ptr<HcalDbService> HcalDbProducer::produce( const HcalDbRecord&)
{
  return mService;
}

void HcalDbProducer::pedestalsCallback (const HcalPedestalsRcd& fRecord) {
  edm::ESHandle <HcalPedestals> item;
  fRecord.get (item);

  if (item->topo()==0) {
    edm::ESHandle<HcalTopology> htopo;
    fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
    const HcalTopology* topo=&(*htopo);
    item->setTopo(topo);
  }

  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Pedestals")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL Pedestals set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void HcalDbProducer::pedestalWidthsCallback (const HcalPedestalWidthsRcd& fRecord) {
  edm::ESHandle <HcalPedestalWidths> item;
  fRecord.get (item);

  if (item->topo()==0) {
    edm::ESHandle<HcalTopology> htopo;
    fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
    const HcalTopology* topo=&(*htopo);
    item->setTopo(topo);
  }

  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("PedestalWidths")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL PedestalWidths set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}


void HcalDbProducer::gainsCallback (const HcalGainsRcd& fRecord) {
  edm::ESHandle <HcalGains> item;
  fRecord.get (item);

  if (item->topo()==0) {
    edm::ESHandle<HcalTopology> htopo;
    fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
    const HcalTopology* topo=&(*htopo);
    item->setTopo(topo);
  }

  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Gains")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL Gains set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}


void HcalDbProducer::gainWidthsCallback (const HcalGainWidthsRcd& fRecord) {
  edm::ESHandle <HcalGainWidths> item;
  fRecord.get (item);

  if (item->topo()==0) {
    edm::ESHandle<HcalTopology> htopo;
    fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
    const HcalTopology* topo=&(*htopo);
    item->setTopo(topo);
  }

  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("GainWidths")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL GainWidths set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void HcalDbProducer::QIEDataCallback (const HcalQIEDataRcd& fRecord) {
  edm::ESHandle <HcalQIEData> item;
  fRecord.get (item);

  if (item->topo()==0) {
    edm::ESHandle<HcalTopology> htopo;
    fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
    const HcalTopology* topo=&(*htopo);
    item->setTopo(topo);
  }

  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("QIEData")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL QIEData set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void HcalDbProducer::channelQualityCallback (const HcalChannelQualityRcd& fRecord) {
  edm::ESHandle <HcalChannelQuality> item;
  fRecord.get (item);

  if (item->topo()==0) {
    edm::ESHandle<HcalTopology> htopo;
    fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
    const HcalTopology* topo=&(*htopo);
    item->setTopo(topo);
  }

  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ChannelQuality")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL ChannelQuality set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void HcalDbProducer::respCorrsCallback (const HcalRespCorrsRcd& fRecord) {
  edm::ESHandle <HcalRespCorrs> item;
  fRecord.get (item);

  if (item->topo()==0) {
    edm::ESHandle<HcalTopology> htopo;
    fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
    const HcalTopology* topo=&(*htopo);
    item->setTopo(topo);
  }

  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("RespCorrs")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL RespCorrs set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void HcalDbProducer::LUTCorrsCallback (const HcalLUTCorrsRcd& fRecord) {
  edm::ESHandle <HcalLUTCorrs> item;
  fRecord.get (item);

  if (item->topo()==0) {
    edm::ESHandle<HcalTopology> htopo;
    fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
    const HcalTopology* topo=&(*htopo);
    item->setTopo(topo);
  }

  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("LUTCorrs")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL LUTCorrs set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void HcalDbProducer::PFCorrsCallback (const HcalPFCorrsRcd& fRecord) {
  edm::ESHandle <HcalPFCorrs> item;
  fRecord.get (item);

  if (item->topo()==0) {
    edm::ESHandle<HcalTopology> htopo;
    fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
    const HcalTopology* topo=&(*htopo);
    item->setTopo(topo);
  }

  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("PFCorrs")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL PFCorrs set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void HcalDbProducer::timeCorrsCallback (const HcalTimeCorrsRcd& fRecord) {
  edm::ESHandle <HcalTimeCorrs> item;
  fRecord.get (item);

  if (item->topo()==0) {
    edm::ESHandle<HcalTopology> htopo;
    fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
    const HcalTopology* topo=&(*htopo);
    item->setTopo(topo);
  }

  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("TimeCorrs")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL TimeCorrs set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void HcalDbProducer::zsThresholdsCallback (const HcalZSThresholdsRcd& fRecord) {
  edm::ESHandle <HcalZSThresholds> item;
  fRecord.get (item);

  if (item->topo()==0) {
    edm::ESHandle<HcalTopology> htopo;
    fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
    const HcalTopology* topo=&(*htopo);
    item->setTopo(topo);
  }

  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ZSThresholds")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL ZSThresholds set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void HcalDbProducer::L1triggerObjectsCallback (const HcalL1TriggerObjectsRcd& fRecord) {
  edm::ESHandle <HcalL1TriggerObjects> item;
  fRecord.get (item);

  if (item->topo()==0) {
    edm::ESHandle<HcalTopology> htopo;
    fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
    const HcalTopology* topo=&(*htopo);
    item->setTopo(topo);
  }

  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("L1TriggerObjects")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL L1TriggerObjects set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void HcalDbProducer::electronicsMapCallback (const HcalElectronicsMapRcd& fRecord) {
  edm::ESHandle <HcalElectronicsMap> item;
  fRecord.get (item);
  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ElectronicsMap")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL Electronics Map set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}

void HcalDbProducer::lutMetadataCallback (const HcalLutMetadataRcd& fRecord) {
  edm::ESHandle <HcalLutMetadata> item;
  fRecord.get (item);

  if (item->topo()==0) {
    edm::ESHandle<HcalTopology> htopo;
    fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
    const HcalTopology* topo=&(*htopo);
    item->setTopo(topo);
  }

  mService->setData (item.product ());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("LutMetadata")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL LUT Metadata set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item.product ()));
  }
}



