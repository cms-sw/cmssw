// -*- C++ -*-
//
// Package:    EventFilter/L1TXRawToDigi
// Class:      L1TCaloLayer1RawToDigi
// 
/**\class L1TCaloLayer1RawToDigi L1TCaloLayer1RawToDigi.cc EventFilter/L1TCaloLayer1RawToDigi/plugins/L1TCaloLayer1RawToDigi.cc

   Description: Raw data unpacker for Layer-1 of Level-1 Calorimeter Trigger

   Implementation: This is meant to be simple and straight-forward.  
       Potential complications due to any changes in firmware are not foreseen.
       It is assumed that if there are major changes, which is unlikely, there 
       will be a new unpacker written to deal with that.

*/
//
// Original Author:  Sridhara Rao Dasu
//         Created:  Sun, 13 Sep 2015 00:31:25 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "FWCore/Framework/interface/Event.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

// Raw data collection headers
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h"

#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveSample.h"

#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

#include "UCTDAQRawData.h"
#include "UCTAMCRawData.h"
#include "UCTCTP7RawData.h"

using namespace edm;

//
// class declaration
//

class L1TCaloLayer1RawToDigi : public stream::EDProducer<> {
public:
  explicit L1TCaloLayer1RawToDigi(const ParameterSet&);
  ~L1TCaloLayer1RawToDigi();

  static void fillDescriptions(ConfigurationDescriptions& descriptions);

private:
  virtual void beginStream(StreamID) override;
  virtual void produce(Event&, const EventSetup&) override;
  virtual void endStream() override;

  void makeECalTPGs(uint32_t lPhi, UCTCTP7RawData& ctp7Data, std::auto_ptr<EcalTrigPrimDigiCollection>& ecalTPGs);

  void makeHCalTPGs(uint32_t lPhi, UCTCTP7RawData& ctp7Data, std::auto_ptr<HcalTrigPrimDigiCollection>& hcalTPGs);

  void makeHFTPGs(uint32_t lPhi, UCTCTP7RawData& ctp7Data, std::auto_ptr<HcalTrigPrimDigiCollection>& hcalTPGs);

  void makeRegions(uint32_t lPhi, UCTCTP7RawData& ctp7Data, std::auto_ptr<L1CaloRegionCollection>& regions);

  //virtual void beginRun(Run const&, EventSetup const&) override;
  //virtual void endRun(Run const&, EventSetup const&) override;
  //virtual void beginLuminosityBlock(LuminosityBlock const&, EventSetup const&) override;
  //virtual void endLuminosityBlock(LuminosityBlock const&, EventSetup const&) override;

  // ----------member data ---------------------------

  InputTag fedRawDataLabel;
  std::vector<int> fedIDs;

  uint32_t event;

  bool verbose;

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
L1TCaloLayer1RawToDigi::L1TCaloLayer1RawToDigi(const ParameterSet& iConfig) :
  fedRawDataLabel(iConfig.getParameter<InputTag>("fedRawDataLabel")),
  fedIDs(iConfig.getParameter<std::vector<int> >("FEDIDs")),
  event(0),
  verbose(iConfig.getParameter<bool>("verbose"))
{

  produces<EcalTrigPrimDigiCollection>();
  produces<HcalTrigPrimDigiCollection>();
  produces<L1CaloRegionCollection>();

  consumes<FEDRawDataCollection>(fedRawDataLabel);

}


L1TCaloLayer1RawToDigi::~L1TCaloLayer1RawToDigi()
{
 
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1TCaloLayer1RawToDigi::produce(Event& iEvent, const EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;
  
  Handle<FEDRawDataCollection> fedRawDataCollection;
  iEvent.getByLabel(fedRawDataLabel, fedRawDataCollection);

  std::auto_ptr<EcalTrigPrimDigiCollection> ecalTPGs(new EcalTrigPrimDigiCollection);
  std::auto_ptr<HcalTrigPrimDigiCollection> hcalTPGs(new HcalTrigPrimDigiCollection);
  std::auto_ptr<L1CaloRegionCollection> regions (new L1CaloRegionCollection);

  // if raw data collection is present, check the headers and do the unpacking
  if (fedRawDataCollection.isValid()) {
    
    for(uint32_t i = 0; i < fedIDs.size(); i++) {

      uint32_t fed = fedIDs[i];

      const FEDRawData& fedRawData = fedRawDataCollection->FEDData(fed);
      
      //Check FED size
      if(verbose) LogDebug("L1TCaloLayer1RawToDigi") << "Upacking FEDRawData for fed " << std::dec << fed << " of size " << fedRawData.size();
      
      const uint64_t *fedRawDataArray = (const uint64_t *) fedRawData.data();

      if ( fedRawData.size() == 0 || fedRawDataArray == nullptr ) {
        LogError("L1TCaloLayer1RawToDigi") << "Could not load FED data for " << fed << ", putting empty collections!";
        continue;
      }
      
      UCTDAQRawData daqData(fedRawDataArray);
      if(verbose && event < 5) daqData.print();
      for(uint32_t i = 0; i < daqData.nAMCs(); i++) {
	UCTAMCRawData amcData(daqData.amcPayload(i));
	if(verbose && event < 5) {	
	  LogDebug("L1TCaloLayer1") << endl;
	  amcData.print();
	  LogDebug("L1TCaloLayer1") << endl;
	}
	uint32_t lPhi = amcData.layer1Phi();
	UCTCTP7RawData ctp7Data(amcData.payload());
	if(verbose && event < 5) ctp7Data.print();
	if(verbose && event < 5) LogDebug("L1TCaloLayer1") << endl;
	makeECalTPGs(lPhi, ctp7Data, ecalTPGs);
	makeHCalTPGs(lPhi, ctp7Data, hcalTPGs);
	// Note: HF TPGs are added at the tail of other TPGs
	makeHFTPGs(lPhi, ctp7Data, hcalTPGs);
	makeRegions(lPhi, ctp7Data, regions);
      }

    }

  }
  else{

    LogError("L1T") << "Cannot unpack: no collection found";

    return; 
  }

  iEvent.put(ecalTPGs);
  iEvent.put(hcalTPGs);
  iEvent.put(regions);

  event++;
  if(verbose && event == 5) LogDebug("L1TCaloLayer1") << "L1TCaloLayer1RawToDigi: Goodbye! Tired of printing junk" << endl;

}

void L1TCaloLayer1RawToDigi::makeECalTPGs(uint32_t lPhi, UCTCTP7RawData& ctp7Data, std::auto_ptr<EcalTrigPrimDigiCollection>& ecalTPGs) {
  UCTCTP7RawData::CaloType cType = UCTCTP7RawData::EBEE;
  for(uint32_t iPhi = 0; iPhi < 4; iPhi++) { // Loop over all four phi divisions on card
    int cPhi = - 1 + lPhi * 4 + iPhi; // Calorimeter phi index
    if(cPhi == 0) cPhi = 72;
    else if(cPhi == -1) cPhi = 71;
    else if(cPhi < -1) {
      LogError("L1TCaloLayer1RawToDigi") << "L1TCaloLayer1RawToDigi: Major error in makeECalTPGs" << std::endl;
      return;
    }
    for(int cEta =  -28; cEta <= 28; cEta++) { // Calorimeter Eta indices (HB/HE for now)
      if(cEta != 0) { // Calorimeter eta = 0 is invalid
	bool negativeEta = false;
	if(cEta < 0) negativeEta = true;
	uint32_t iEta = abs(cEta);
	// This code is fragile! Note that towerDatum is packed as is done in EcalTriggerPrimitiveSample
	// Bottom 8-bits are ET
	// Then finegrain feature bit
	// Then three bits have ttBits, which I have no clue about (not available on ECAL links so not set)
	// Then there is a spare FG Veto bit, which is used for L1 spike detection (not available on ECAL links so not set)
	// Top three bits seem to be unused. So, we steal those to set the tower masking, link masking and link status information 
	// To decode these custom three bits use ((EcalTriggerPrimitiveSample::raw() >> 13) & 0x7)
	uint32_t towerDatum = ctp7Data.getET(cType, negativeEta, iEta, iPhi);
	if(ctp7Data.getFB(cType, negativeEta, iEta, iPhi)) towerDatum |= 0x0100;
	if(ctp7Data.isTowerMasked(cType, negativeEta, iEta, iPhi)) towerDatum |= 0x2000;
	if(ctp7Data.isLinkMasked(cType, negativeEta, iEta, iPhi)) towerDatum |= 0x4000;
	if(ctp7Data.isLinkMisaligned(cType, negativeEta, iEta, iPhi) ||
	   ctp7Data.isLinkInError(cType, negativeEta, iEta, iPhi) ||
	   ctp7Data.isLinkDown(cType, negativeEta, iEta, iPhi)) towerDatum |= 0x8000;
	EcalTriggerPrimitiveSample sample(towerDatum); 
	int zSide = cEta / ((int) iEta);
	EcalSubdetector ecalTriggerTower = EcalSubdetector::EcalTriggerTower;
	EcalTrigTowerDetId id(zSide, ecalTriggerTower, iEta, cPhi);
	EcalTriggerPrimitiveDigi tpg(id);
	tpg.setSize(1);
	tpg.setSample(0, sample);
	ecalTPGs->push_back(tpg);
      }
    }
  }

}

void L1TCaloLayer1RawToDigi::makeHCalTPGs(uint32_t lPhi, UCTCTP7RawData& ctp7Data, std::auto_ptr<HcalTrigPrimDigiCollection>& hcalTPGs) {
  UCTCTP7RawData::CaloType cType = UCTCTP7RawData::HBHE;
  for(uint32_t iPhi = 0; iPhi < 4; iPhi++) { // Loop over all four phi divisions on card
    int cPhi = - 1 + lPhi * 4 + iPhi; // Calorimeter phi index
    if(cPhi == 0) cPhi = 72;
    else if(cPhi == -1) cPhi = 71;
    else if(cPhi < -1) {
      LogError("L1TCaloLayer1RawToDigi") << "L1TCaloLayer1RawToDigi: Major error in makeHCalTPGs" << std::endl;
      return;
    }
    for(int cEta =  -28; cEta <= 28; cEta++) { // Calorimeter Eta indices (HB/HE for now)
      if(cEta != 0) { // Calorimeter eta = 0 is invalid
	bool negativeEta = false;
	if(cEta < 0) negativeEta = true;
	uint32_t iEta = abs(cEta);
	// This code is fragile! Note that towerDatum is packed as is done in HcalTriggerPrimitiveSample
	// Bottom 8-bits are ET
	// Then feature bit
	// The remaining bits are undefined presently
	// We use next three bits for link details, which we did not have room in EcalTriggerPrimitiveSample case
	// We use next three bits to set the tower masking, link masking and link status information as done for Ecal
	// To decode these custom six bits use ((EcalTriggerPrimitiveSample::raw() >> 9) & 0x77)
	uint32_t towerDatum = ctp7Data.getET(cType, negativeEta, iEta, iPhi);
	if(ctp7Data.getFB(cType, negativeEta, iEta, iPhi)) towerDatum |= 0x0100;
	if(ctp7Data.isLinkMisaligned(cType, negativeEta, iEta, iPhi)) towerDatum |= 0x0200;
	if(ctp7Data.isLinkInError(cType, negativeEta, iEta, iPhi)) towerDatum |= 0x0400;
	if(ctp7Data.isLinkDown(cType, negativeEta, iEta, iPhi)) towerDatum |= 0x0800;
	if(ctp7Data.isTowerMasked(cType, negativeEta, iEta, iPhi)) towerDatum |= 0x2000;
	if(ctp7Data.isLinkMasked(cType, negativeEta, iEta, iPhi)) towerDatum |= 0x4000;
	if(ctp7Data.isLinkMisaligned(cType, negativeEta, iEta, iPhi) ||
	   ctp7Data.isLinkInError(cType, negativeEta, iEta, iPhi) ||
	   ctp7Data.isLinkDown(cType, negativeEta, iEta, iPhi)) towerDatum |= 0x8000;
	HcalTriggerPrimitiveSample sample(towerDatum); 
	HcalTrigTowerDetId id(cEta, cPhi);
	HcalTriggerPrimitiveDigi tpg(id);
	tpg.setSize(1);
	tpg.setSample(0, sample);
	hcalTPGs->push_back(tpg);
      }
    }
  }

}

void L1TCaloLayer1RawToDigi::makeHFTPGs(uint32_t lPhi, UCTCTP7RawData& ctp7Data, std::auto_ptr<HcalTrigPrimDigiCollection>& hcalTPGs) {
  UCTCTP7RawData::CaloType cType = UCTCTP7RawData::HF;
  for(uint32_t side = 0; side <= 1; side++) {
    bool negativeEta = false;
    if(side == 0) negativeEta = true;
    for(uint32_t iEta = 30; iEta <= 40; iEta++) {
      for(uint32_t iPhi = 0; iPhi < 2; iPhi++) {
	if(iPhi == 1 && iEta == 40) iEta = 41;
	int cPhi = 1 + lPhi * 4 + iPhi * 2; // Calorimeter phi index: 1, 3, 5, ... 71
	int cEta = iEta;
	if(negativeEta) cEta = -iEta;
	// This code is fragile! Note that towerDatum is packed as is done in HcalTriggerPrimitiveSample
	// Bottom 8-bits are ET
	// Then feature bit
	// The remaining bits are undefined presently
	// We use next three bits for link details, which we did not have room in EcalTriggerPrimitiveSample case
	// We use next three bits to set the tower masking, link masking and link status information as done for Ecal
	// To decode these custom six bits use ((EcalTriggerPrimitiveSample::raw() >> 9) & 0x77)
	uint32_t towerDatum = ctp7Data.getET(cType, negativeEta, iEta, iPhi);
	if(ctp7Data.getFB(cType, negativeEta, iEta, iPhi)) towerDatum |= 0x0100;
	if(ctp7Data.isLinkMisaligned(cType, negativeEta, iEta, iPhi)) towerDatum |= 0x0200;
	if(ctp7Data.isLinkInError(cType, negativeEta, iEta, iPhi)) towerDatum |= 0x0400;
	if(ctp7Data.isLinkDown(cType, negativeEta, iEta, iPhi)) towerDatum |= 0x0800;
	if(ctp7Data.isTowerMasked(cType, negativeEta, iEta, iPhi)) towerDatum |= 0x2000;
	if(ctp7Data.isLinkMasked(cType, negativeEta, iEta, iPhi)) towerDatum |= 0x4000;
	if(ctp7Data.isLinkMisaligned(cType, negativeEta, iEta, iPhi) ||
	   ctp7Data.isLinkInError(cType, negativeEta, iEta, iPhi) ||
	   ctp7Data.isLinkDown(cType, negativeEta, iEta, iPhi)) towerDatum |= 0x8000;
	HcalTriggerPrimitiveSample sample(towerDatum); 
	HcalTrigTowerDetId id(cEta, cPhi);
	id.setVersion(1); // To not process these 1x1 HF TPGs with RCT
	HcalTriggerPrimitiveDigi tpg(id);
	tpg.setSize(1);
	tpg.setSample(0, sample);
	hcalTPGs->push_back(tpg);
      }
    }
  }
}

void 
L1TCaloLayer1RawToDigi::makeRegions(uint32_t lPhi, UCTCTP7RawData& ctp7Data, std::auto_ptr<L1CaloRegionCollection>& regions) {
  for(uint32_t side = 0; side <= 1; side++) {
    bool negativeEta = false;
    if(side == 0) negativeEta = true;
    for(uint32_t region = 0; region <= 6; region++) {
      uint32_t regionData = ctp7Data.getRegionSummary(negativeEta, region);
      uint32_t lEta = region + 4; // GCT eta goes 0-21, 0-3 -HF, 4-10 -B/E, 11-17 +B/E, 18-21 +HF
      if(!negativeEta) lEta += 7;
      regions->push_back(L1CaloRegion((uint16_t) regionData, (unsigned) lEta, (unsigned) lPhi, (int16_t) 0));
    }
  }
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
L1TCaloLayer1RawToDigi::beginStream(StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
L1TCaloLayer1RawToDigi::endStream() {
}

// ------------ method called when starting to processes a run  ------------
/*
  void
  L1TCaloLayer1RawToDigi::beginRun(Run const&, EventSetup const&)
  {
  }
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
  void
  L1TCaloLayer1RawToDigi::endRun(Run const&, EventSetup const&)
  {
  }
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
  void
  L1TCaloLayer1RawToDigi::beginLuminosityBlock(LuminosityBlock const&, EventSetup const&)
  {
  }
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
  void
  L1TCaloLayer1RawToDigi::endLuminosityBlock(LuminosityBlock const&, EventSetup const&)
  {
  }
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TCaloLayer1RawToDigi::fillDescriptions(ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TCaloLayer1RawToDigi);
