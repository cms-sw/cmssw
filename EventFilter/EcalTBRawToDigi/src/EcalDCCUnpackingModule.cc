/* \file EcalDCCUnpackingModule.h
 *
 *  \author N. Marinelli
 *  \author G. Della Ricca
 *  \author G. Franzoni
 *  \author A. Ghezzi
 */

#include <EventFilter/EcalTBRawToDigi/interface/EcalDCCUnpackingModule.h>
#include <EventFilter/EcalTBRawToDigi/src/EcalTBDaqFormatter.h>
#include <EventFilter/EcalTBRawToDigi/src/EcalSupervisorDataFormatter.h>
#include <EventFilter/EcalTBRawToDigi/src/CamacTBDataFormatter.h>
#include <EventFilter/EcalTBRawToDigi/src/TableDataFormatter.h>
#include <EventFilter/EcalTBRawToDigi/src/MatacqDataFormatter.h>
#include <EventFilter/EcalTBRawToDigi/src/ECALParserException.h>
#include <EventFilter/EcalTBRawToDigi/src/ECALParserBlockException.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <TBDataFormats/EcalTBObjects/interface/EcalTBCollections.h>
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>

#include <iostream>
#include <iomanip>

// in full CMS this range cannot be used (allocated to pixel, see DataFormats/ FEDRawData/ src/ FEDNumbering.cc)
#define BEG_DCC_FED_ID 0
#define END_DCC_FED_ID 35
#define BEG_DCC_FED_ID_GLOBAL 600
#define END_DCC_FED_ID_GLOBAL 670

#define ECAL_SUPERVISOR_FED_ID 40
#define TBCAMAC_FED_ID 41
#define TABLE_FED_ID 42
#define MATACQ_FED_ID 43

EcalDCCTBUnpackingModule::EcalDCCTBUnpackingModule(const edm::ParameterSet& pset)
    : fedRawDataCollectionTag_(pset.getParameter<edm::InputTag>("fedRawDataCollectionTag")) {
  formatter_ = new EcalTBDaqFormatter();
  ecalSupervisorFormatter_ = new EcalSupervisorTBDataFormatter();
  camacTBformatter_ = new CamacTBDataFormatter();
  tableFormatter_ = new TableDataFormatter();
  matacqFormatter_ = new MatacqTBDataFormatter();

  // digis
  produces<EBDigiCollection>("ebDigis");
  produces<EcalMatacqDigiCollection>();
  produces<EcalPnDiodeDigiCollection>();
  produces<EcalRawDataCollection>();
  produces<EcalTrigPrimDigiCollection>("EBTT");

  //TB specifics data
  produces<EcalTBHodoscopeRawInfo>();
  produces<EcalTBTDCRawInfo>();
  produces<EcalTBEventHeader>();

  // crystals' integrity
  produces<EBDetIdCollection>("EcalIntegrityDCCSizeErrors");
  produces<EcalElectronicsIdCollection>("EcalIntegrityTTIdErrors");
  produces<EcalElectronicsIdCollection>("EcalIntegrityBlockSizeErrors");
  produces<EBDetIdCollection>("EcalIntegrityChIdErrors");
  produces<EBDetIdCollection>("EcalIntegrityGainErrors");
  produces<EBDetIdCollection>("EcalIntegrityGainSwitchErrors");

  // mem channels' integrity
  produces<EcalElectronicsIdCollection>("EcalIntegrityMemTtIdErrors");
  produces<EcalElectronicsIdCollection>("EcalIntegrityMemBlockSize");
  produces<EcalElectronicsIdCollection>("EcalIntegrityMemChIdErrors");
  produces<EcalElectronicsIdCollection>("EcalIntegrityMemGainErrors");
}

EcalDCCTBUnpackingModule::~EcalDCCTBUnpackingModule() { delete formatter_; }

void EcalDCCTBUnpackingModule::produce(edm::Event& e, const edm::EventSetup& c) {
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByLabel(fedRawDataCollectionTag_, rawdata);

  // create the collection of Ecal Digis
  auto productEb = std::make_unique<EBDigiCollection>();

  // create the collection of Matacq Digi
  auto productMatacq = std::make_unique<EcalMatacqDigiCollection>();

  // create the collection of Ecal PN's
  auto productPN = std::make_unique<EcalPnDiodeDigiCollection>();

  //create the collection of Ecal DCC Header
  auto productDCCHeader = std::make_unique<EcalRawDataCollection>();

  // create the collection with trigger primitives, bits and flags
  auto productTriggerPrimitives = std::make_unique<EcalTrigPrimDigiCollection>();

  // create the collection of Ecal Integrity DCC Size
  auto productDCCSize = std::make_unique<EBDetIdCollection>();

  // create the collection of Ecal Integrity TT Id
  auto productTTId = std::make_unique<EcalElectronicsIdCollection>();

  // create the collection of Ecal Integrity TT Block Size
  auto productBlockSize = std::make_unique<EcalElectronicsIdCollection>();

  // create the collection of Ecal Integrity Ch Id
  auto productChId = std::make_unique<EBDetIdCollection>();

  // create the collection of Ecal Integrity Gain
  auto productGain = std::make_unique<EBDetIdCollection>();

  // create the collection of Ecal Integrity Gain Switch
  auto productGainSwitch = std::make_unique<EBDetIdCollection>();

  // create the collection of Ecal Integrity Mem towerBlock_id errors
  auto productMemTtId = std::make_unique<EcalElectronicsIdCollection>();

  // create the collection of Ecal Integrity Mem gain errors
  auto productMemBlockSize = std::make_unique<EcalElectronicsIdCollection>();

  // create the collection of Ecal Integrity Mem gain errors
  auto productMemGain = std::make_unique<EcalElectronicsIdCollection>();

  // create the collection of Ecal Integrity Mem ch_id errors
  auto productMemChIdErrors = std::make_unique<EcalElectronicsIdCollection>();

  // create the collection of TB specifics data
  auto productHodo = std::make_unique<EcalTBHodoscopeRawInfo>();
  auto productTdc = std::make_unique<EcalTBTDCRawInfo>();
  auto productHeader = std::make_unique<EcalTBEventHeader>();

  try {
    for (int id = 0; id <= FEDNumbering::MAXFEDID; ++id) {
      //    edm::LogInfo("EcalDCCTBUnpackingModule") << "EcalDCCTBUnpackingModule::Got FED ID "<< id <<" ";
      const FEDRawData& data = rawdata->FEDData(id);
      //    edm::LogInfo("EcalDCCTBUnpackingModule") << " Fed data size " << data.size() ;

      //std::cout <<"1 Fed id: "<<dec<<id<< " Fed data size: " <<data.size() << std::endl;
      //    const unsigned char * pData = data.data();
      //    int length = data.size();
      //    if(length >0 ){
      //      if(length >= 40){length = 40;}
      //    std::cout<<"##############################################################"<<std::endl;
      //    for( int i=0; i<length; i++ ) {
      //      std::cout << std::hex << std::setw(8) << int(pData[i]) << " ";
      //      if( (i+1)%8 == 0 ) std::cout << std::endl;
      //     }
      //    std::cout<<"##############################################################"<<std::endl;
      //    }
      if (data.size() > 16) {
        if ((id >= BEG_DCC_FED_ID && id <= END_DCC_FED_ID) ||
            (id >= BEG_DCC_FED_ID_GLOBAL &&
             id <= END_DCC_FED_ID_GLOBAL)) {  // do the DCC data unpacking and fill the collections

          (*productHeader).setSmInBeam(id);
          formatter_->interpretRawData(data,
                                       *productEb,
                                       *productPN,
                                       *productDCCHeader,
                                       *productDCCSize,
                                       *productTTId,
                                       *productBlockSize,
                                       *productChId,
                                       *productGain,
                                       *productGainSwitch,
                                       *productMemTtId,
                                       *productMemBlockSize,
                                       *productMemGain,
                                       *productMemChIdErrors,
                                       *productTriggerPrimitives);
          int runType = (*productDCCHeader)[0].getRunType();
          if (runType == EcalDCCHeaderBlock::COSMIC || runType == EcalDCCHeaderBlock::BEAMH4)
            (*productHeader).setTriggerMask(0x1);
          else if (runType == 4 || runType == 5 || runType == 6)  //laser runs
            (*productHeader).setTriggerMask(0x2000);
          else if (runType == 9 || runType == 10 || runType == 11)  //pedestal runs
            (*productHeader).setTriggerMask(0x800);
          LogDebug("EcalDCCTBUnpackingModule")
              << "Event type is " << (*productHeader).eventType() << " dbEventType " << (*productHeader).dbEventType();
        } else if (id == ECAL_SUPERVISOR_FED_ID)
          ecalSupervisorFormatter_->interpretRawData(data, *productHeader);
        else if (id == TBCAMAC_FED_ID)
          camacTBformatter_->interpretRawData(data, *productHeader, *productHodo, *productTdc);
        else if (id == TABLE_FED_ID)
          tableFormatter_->interpretRawData(data, *productHeader);
        else if (id == MATACQ_FED_ID)
          matacqFormatter_->interpretRawData(data, *productMatacq);
      }  // endif
    }    //endfor

    // commit to the event
    e.put(std::move(productPN));
    e.put(std::move(productEb), "ebDigis");
    e.put(std::move(productMatacq));
    e.put(std::move(productDCCHeader));
    e.put(std::move(productTriggerPrimitives), "EBTT");

    e.put(std::move(productDCCSize), "EcalIntegrityDCCSizeErrors");
    e.put(std::move(productTTId), "EcalIntegrityTTIdErrors");
    e.put(std::move(productBlockSize), "EcalIntegrityBlockSizeErrors");
    e.put(std::move(productChId), "EcalIntegrityChIdErrors");
    e.put(std::move(productGain), "EcalIntegrityGainErrors");
    e.put(std::move(productGainSwitch), "EcalIntegrityGainSwitchErrors");

    e.put(std::move(productMemTtId), "EcalIntegrityMemTtIdErrors");
    e.put(std::move(productMemBlockSize), "EcalIntegrityMemBlockSize");
    e.put(std::move(productMemChIdErrors), "EcalIntegrityMemChIdErrors");
    e.put(std::move(productMemGain), "EcalIntegrityMemGainErrors");

    e.put(std::move(productHodo));
    e.put(std::move(productTdc));
    e.put(std::move(productHeader));

  } catch (ECALTBParserException& e) {
    std::cout << "[EcalDCCTBUnpackingModule] " << e.what() << std::endl;
  } catch (ECALTBParserBlockException& e) {
    std::cout << "[EcalDCCTBUnpackingModule] " << e.what() << std::endl;
  } catch (cms::Exception& e) {
    std::cout << "[EcalDCCTBUnpackingModule] " << e.what() << std::endl;
  } catch (...) {
    std::cout << "[EcalDCCTBUnpackingModule] Unknown exception ..." << std::endl;
  }
}
