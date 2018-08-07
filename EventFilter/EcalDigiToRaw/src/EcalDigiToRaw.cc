// -*- C++ -*-
//
// Package:    EcalDigiToRaw
// Class:      EcalDigiToRaw
// 
/**\class EcalDigiToRaw EcalDigiToRaw.cc EventFilter/EcalDigiToRaw/src/EcalDigiToRaw.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Emmanuelle Perez
//         Created:  Sat Nov 25 13:59:51 CET 2006
//
//


// system include files
#include <memory>
#include <iostream>
#include <string>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/EcalDigiToRaw/interface/TowerBlockFormatter.h"
#include "EventFilter/EcalDigiToRaw/interface/TCCBlockFormatter.h"
#include "EventFilter/EcalDigiToRaw/interface/BlockFormatter.h"
#include "EventFilter/EcalDigiToRaw/interface/SRBlockFormatter.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalSrFlag.h"

#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"


#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

class EcalDigiToRaw : public edm::global::EDProducer<> {
   public:
      EcalDigiToRaw(const edm::ParameterSet& pset);

      void produce(edm::StreamID, edm::Event& e, const edm::EventSetup& c) const override;

      typedef long long Word64;
      typedef unsigned int Word32;

      static const int BXMAX = 2808;


   private:


      // ----------member data ---------------------------

       edm::EDGetTokenT<EcalTrigPrimDigiCollection> labelTT_ ;
       edm::EDGetTokenT<EBSrFlagCollection> labelEBSR_ ;
       edm::EDGetTokenT<EESrFlagCollection> labelEESR_ ;
       edm::EDGetTokenT<EBDigiCollection> EBDigiToken_ ;
       edm::EDGetTokenT<EEDigiCollection> EEDigiToken_;
       edm::EDPutTokenT<FEDRawDataCollection> putToken_;
  
       const std::vector<int32_t> listDCCId_;
    

       const BlockFormatter::Config config_;

};

using namespace edm;
using namespace std;

EcalDigiToRaw::EcalDigiToRaw(const edm::ParameterSet& iConfig):
  listDCCId_{iConfig.getUntrackedParameter< std::vector<int32_t> >("listDCCId")},
  config_{
    &listDCCId_,
      iConfig.getUntrackedParameter<bool>("debug"),
      iConfig.getUntrackedParameter<bool>("DoBarrel"),
      iConfig.getUntrackedParameter<bool>("DoEndCap"),
      iConfig.getUntrackedParameter<bool>("WriteTCCBlock"),
      iConfig.getUntrackedParameter<bool>("WriteSRFlags"),
      iConfig.getUntrackedParameter<bool>("WriteTowerBlock") }
{
   auto label= iConfig.getParameter<string>("Label");
   auto instanceNameEB = iConfig.getParameter<string>("InstanceEB");
   auto instanceNameEE = iConfig.getParameter<string>("InstanceEE");

   edm::InputTag EBlabel = edm::InputTag(label,instanceNameEB);
   edm::InputTag EElabel = edm::InputTag(label,instanceNameEE);

   EBDigiToken_ = consumes<EBDigiCollection>(EBlabel);
   EEDigiToken_ = consumes<EEDigiCollection>(EElabel);

   labelTT_ = consumes<EcalTrigPrimDigiCollection>(iConfig.getParameter<edm::InputTag>("labelTT"));

   labelEBSR_ = consumes<EBSrFlagCollection>(iConfig.getParameter<edm::InputTag>("labelEBSRFlags"));
   labelEESR_ = consumes<EESrFlagCollection>(iConfig.getParameter<edm::InputTag>("labelEESRFlags"));

   putToken_ = produces<FEDRawDataCollection>();
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
EcalDigiToRaw::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{

  if (config_.debug_) cout << "Enter in EcalDigiToRaw::produce ... " << endl;

  ESHandle< EcalElectronicsMapping > ecalmapping;
  iSetup.get< EcalMappingRcd >().get(ecalmapping);
  const EcalElectronicsMapping* TheMapping = ecalmapping.product();
  
  FEDRawDataCollection productRawData;
  
  BlockFormatter::Params params;
  int counter = iEvent.id().event();
  params.counter_ = counter;
  params.orbit_number_ = iEvent.orbitNumber();
  params.bx_ = iEvent.bunchCrossing();
  params.lv1_ = counter % (0x1<<24);
  params.runnumber_ = iEvent.id().run();
  
  BlockFormatter Headerblockformatter(config_,params);
  TCCBlockFormatter TCCblockformatter(config_,params);
  TowerBlockFormatter Towerblockformatter(config_,params);
  SRBlockFormatter SRblockformatter(config_,params);

  
  Headerblockformatter.DigiToRaw(&productRawData);


// ---------   Now the Trigger Block part

  Handle<EcalTrigPrimDigiCollection> ecalTrigPrim;
  
  Handle<EBSrFlagCollection> ebSrFlags;
  Handle<EESrFlagCollection> eeSrFlags;


  if (config_.doTCC_) {

     if (config_.debug_) cout << "Creation of the TCC block  " << endl;
     // iEvent.getByType(ecalTrigPrim);
	iEvent.getByToken(labelTT_, ecalTrigPrim);

     // loop on TP's and add one by one to the block
     for (EcalTrigPrimDigiCollection::const_iterator it = ecalTrigPrim -> begin();
			   it != ecalTrigPrim -> end(); it++) {

	   const EcalTriggerPrimitiveDigi& trigprim = *it;
	   const EcalTrigTowerDetId& detid = it -> id();

	   if ( (detid.subDet() == EcalBarrel) && (! config_.doBarrel_) ) continue;
           if ( (detid.subDet() == EcalEndcap) && (! config_.doEndCap_) ) continue;

	   int iDCC = TheMapping -> DCCid(detid);
           int FEDid = FEDNumbering::MINECALFEDID + iDCC;

           FEDRawData& rawdata = productRawData.FEDData(FEDid);
	   
	   // adding the primitive to the block
	   TCCblockformatter.DigiToRaw(trigprim, rawdata, TheMapping);

     }   // end loop on ecalTrigPrim

   }  // endif doTCC


   if (config_.doSR_) {	
	if (config_.debug_) cout << " Process the SR flags " << endl;

	if (config_.doBarrel_) {

        // iEvent.getByType(ebSrFlags);
	   iEvent.getByToken(labelEBSR_, ebSrFlags);
                                                                                                                                                
           for (EBSrFlagCollection::const_iterator it = ebSrFlags -> begin();
                        it != ebSrFlags -> end(); it++) {
                const EcalSrFlag& srflag = *it;
		int flag = srflag.value();

		EcalTrigTowerDetId id = srflag.id();
		int Dccid = TheMapping -> DCCid(id);
		int DCC_Channel = TheMapping -> iTT(id);
		int FEDid = FEDNumbering::MINECALFEDID + Dccid;
		// if (Dccid == 10) cout << "Dcc " << Dccid << " DCC_Channel " << DCC_Channel << " flag " << flag << endl;
		if (config_.debug_) cout << "will process SRblockformatter_ for FEDid " << dec << FEDid << endl;
		FEDRawData& rawdata = productRawData.FEDData(FEDid);
		if (config_.debug_) Headerblockformatter.print(rawdata);
		SRblockformatter.DigiToRaw(Dccid,DCC_Channel,flag, rawdata);

           }
	}  // end DoBarrel


	if (config_.doEndCap_) {
	// iEvent.getByType(eeSrFlags);
	iEvent.getByToken(labelEESR_, eeSrFlags);

           for (EESrFlagCollection::const_iterator it = eeSrFlags -> begin();
                        it != eeSrFlags -> end(); it++) {
                const EcalSrFlag& srflag = *it;
                int flag = srflag.value();
		EcalScDetId id = srflag.id();
		pair<int, int> ind = TheMapping -> getDCCandSC(id);
		int Dccid = ind.first;
		int DCC_Channel = ind.second;

                int FEDid = FEDNumbering::MINECALFEDID + Dccid;
                FEDRawData& rawdata = productRawData.FEDData(FEDid);
                SRblockformatter.DigiToRaw(Dccid,DCC_Channel,flag, rawdata);
           }
	}  // end doEndCap

   }   // endif doSR


// ---------  Now the Tower Block part

  Handle<EBDigiCollection> ebDigis;
  Handle<EEDigiCollection> eeDigis;

  if (config_.doTower_) {

	if (config_.doBarrel_) {
   	if (config_.debug_) cout << "Creation of the TowerBlock ... Barrel case " << endl;
        iEvent.getByToken(EBDigiToken_,ebDigis);
        for (EBDigiCollection::const_iterator it=ebDigis -> begin();
                                it != ebDigis->end(); it++) {
                const EBDataFrame& dataframe = *it;
                const EBDetId& ebdetid = it -> id();
		int DCCid = TheMapping -> DCCid(ebdetid);
        	int FEDid = FEDNumbering::MINECALFEDID + DCCid ;
                FEDRawData& rawdata = productRawData.FEDData(FEDid);
                Towerblockformatter.DigiToRaw(dataframe, rawdata, TheMapping);
        }

	}

	if (config_.doEndCap_) {
	if (config_.debug_) cout << "Creation of the TowerBlock ... EndCap case " << endl;
        iEvent.getByToken(EEDigiToken_,eeDigis);
        for (EEDigiCollection::const_iterator it=eeDigis -> begin();
                                it != eeDigis->end(); it++) {
                const EEDataFrame& dataframe = *it;
                const EEDetId& eedetid = it -> id();
		EcalElectronicsId elid = TheMapping -> getElectronicsId(eedetid);
                int DCCid = elid.dccId() ;   
                int FEDid = FEDNumbering::MINECALFEDID + DCCid;
                FEDRawData& rawdata = productRawData.FEDData(FEDid);
                Towerblockformatter.DigiToRaw(dataframe, rawdata, TheMapping);
        }
	}

  }  // endif config_.doTower_



// -------- Clean up things ...

  map<int, map<int,int> >& FEDorder = Towerblockformatter.GetFEDorder();

  Headerblockformatter.CleanUp(&productRawData, &FEDorder);


/*
   cout << "For FED 633 " << endl;
         FEDRawData& rawdata = productRawData -> FEDData(633);
         Headerblockformatter_ -> print(rawdata);
*/

 // Headerblockformatter_ -> PrintSizes(productRawData.get());



 Towerblockformatter.EndEvent(&productRawData);

 iEvent.emplace(putToken_, std::move(productRawData));


 return;

}


DEFINE_FWK_MODULE(EcalDigiToRaw);
