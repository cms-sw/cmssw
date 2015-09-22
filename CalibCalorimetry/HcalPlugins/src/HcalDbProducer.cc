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
//
//


// system include files
#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"


#include "CondFormats/HcalObjects/interface/AllObjects.h"

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

  setWhatProduced(this, &HcalDbProducer::produceChannelQualityWithTopo, edm::es::Label("withTopo"));
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
  edm::ESTransientHandle <HcalPedestals> item;
  fRecord.get (item);

  mPedestals.reset( new HcalPedestals(*item) );
  
  edm::ESHandle<HcalTopology> htopo;
  fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  mPedestals->setTopo(topo);
  

  mService->setData (mPedestals.get());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Pedestals")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL Pedestals set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(mPedestals));
  }
}

boost::shared_ptr<HcalChannelQuality> HcalDbProducer::produceChannelQualityWithTopo(const HcalChannelQualityRcd& fRecord)
{
  edm::ESHandle <HcalChannelQuality> item;
  fRecord.get (item);

  boost::shared_ptr<HcalChannelQuality> channelQuality( new HcalChannelQuality(*item) );

  edm::ESHandle<HcalTopology> htopo;
  fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  channelQuality->setTopo(topo);

  return channelQuality;
}

void HcalDbProducer::pedestalWidthsCallback (const HcalPedestalWidthsRcd& fRecord) {
  edm::ESTransientHandle <HcalPedestalWidths> item;
  fRecord.get (item);

  mPedestalWidths.reset( new HcalPedestalWidths(*item));

  edm::ESHandle<HcalTopology> htopo;
  fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  mPedestalWidths->setTopo(topo);

  mService->setData (mPedestalWidths.get());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("PedestalWidths")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL PedestalWidths set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(mPedestalWidths));
  }
}


void HcalDbProducer::gainsCallback (const HcalGainsRcd& fRecord) {
  edm::ESTransientHandle <HcalGains> item;
  fRecord.get (item);

  mGains.reset( new HcalGains(*item) );
  edm::ESHandle<HcalTopology> htopo;
  fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  mGains->setTopo(topo);


  mService->setData (mGains.get());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("Gains")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL Gains set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(mGains));
  }
}


void HcalDbProducer::gainWidthsCallback (const HcalGainWidthsRcd& fRecord) {
  edm::ESTransientHandle <HcalGainWidths> item;
  fRecord.get (item);

  mGainWidths.reset( new HcalGainWidths(*item));
  edm::ESHandle<HcalTopology> htopo;
  fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  mGainWidths->setTopo(topo);


  mService->setData (mGainWidths.get());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("GainWidths")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL GainWidths set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(mGainWidths));
  }
}

void HcalDbProducer::QIEDataCallback (const HcalQIEDataRcd& fRecord) {
  edm::ESTransientHandle <HcalQIEData> item;
  fRecord.get (item);

  mQIEData.reset( new HcalQIEData(*item));


  edm::ESHandle<HcalTopology> htopo;
  fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  mQIEData->setTopo(topo);

  mService->setData (mQIEData.get());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("QIEData")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL QIEData set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(mQIEData));
  }
}

void HcalDbProducer::channelQualityCallback (const HcalChannelQualityRcd& fRecord) {
  edm::ESHandle <HcalChannelQuality> item;
  fRecord.get ("withTopo", item );

  mService->setData (item.product());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ChannelQuality")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL ChannelQuality set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(item));
  }
}

void HcalDbProducer::respCorrsCallback (const HcalRespCorrsRcd& fRecord) {
  edm::ESTransientHandle <HcalRespCorrs> item;
  fRecord.get (item);

  mRespCorrs.reset( new HcalRespCorrs(*item) );

  edm::ESHandle<HcalTopology> htopo;
  fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  mRespCorrs->setTopo(topo);
  
  mService->setData (mRespCorrs.get());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("RespCorrs")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL RespCorrs set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(mRespCorrs));
  }
}

void HcalDbProducer::LUTCorrsCallback (const HcalLUTCorrsRcd& fRecord) {
  edm::ESTransientHandle <HcalLUTCorrs> item;
  fRecord.get (item);

  mLUTCorrs.reset( new HcalLUTCorrs(*item) );

  edm::ESHandle<HcalTopology> htopo;
  fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  mLUTCorrs->setTopo(topo);

  mService->setData (mLUTCorrs.get());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("LUTCorrs")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL LUTCorrs set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(mLUTCorrs));
  }
}

void HcalDbProducer::PFCorrsCallback (const HcalPFCorrsRcd& fRecord) {
  edm::ESTransientHandle <HcalPFCorrs> item;
  fRecord.get (item);

  mPFCorrs.reset( new HcalPFCorrs(*item) );

  edm::ESHandle<HcalTopology> htopo;
  fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  mPFCorrs->setTopo(topo);

  mService->setData (mPFCorrs.get());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("PFCorrs")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL PFCorrs set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(mPFCorrs));
  }
}

void HcalDbProducer::timeCorrsCallback (const HcalTimeCorrsRcd& fRecord) {
  edm::ESTransientHandle <HcalTimeCorrs> item;
  fRecord.get (item);

  mTimeCorrs.reset( new HcalTimeCorrs(*item) );

  edm::ESHandle<HcalTopology> htopo;
  fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  mTimeCorrs->setTopo(topo);

  mService->setData (mTimeCorrs.get());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("TimeCorrs")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL TimeCorrs set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(mTimeCorrs));
  }
}

void HcalDbProducer::zsThresholdsCallback (const HcalZSThresholdsRcd& fRecord) {
  edm::ESTransientHandle <HcalZSThresholds> item;
  fRecord.get (item);

  mZSThresholds.reset( new HcalZSThresholds(*item) );

  edm::ESHandle<HcalTopology> htopo;
  fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  mZSThresholds->setTopo(topo);

  mService->setData (mZSThresholds.get());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("ZSThresholds")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL ZSThresholds set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(mZSThresholds));
  }
}

void HcalDbProducer::L1triggerObjectsCallback (const HcalL1TriggerObjectsRcd& fRecord) {
  edm::ESTransientHandle <HcalL1TriggerObjects> item;
  fRecord.get (item);

  mL1TriggerObjects.reset( new HcalL1TriggerObjects(*item) );

  edm::ESHandle<HcalTopology> htopo;
  fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  mL1TriggerObjects->setTopo(topo);

  mService->setData (mL1TriggerObjects.get());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("L1TriggerObjects")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL L1TriggerObjects set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(mL1TriggerObjects));
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
  edm::ESTransientHandle <HcalLutMetadata> item;
  fRecord.get (item);

  mLutMetadata.reset( new HcalLutMetadata(*item) );

  edm::ESHandle<HcalTopology> htopo;
  fRecord.getRecord<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology* topo=&(*htopo);
  mLutMetadata->setTopo(topo);

  mService->setData (mLutMetadata.get());
  if (std::find (mDumpRequest.begin(), mDumpRequest.end(), std::string ("LutMetadata")) != mDumpRequest.end()) {
    *mDumpStream << "New HCAL LUT Metadata set" << std::endl;
    HcalDbASCIIIO::dumpObject (*mDumpStream, *(mLutMetadata));
  }
}



