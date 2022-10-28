/* \file EcalDCCTB07UnpackingModule.h
 *
 *  \author Y. Maravin
 *  \author G. Franzoni
 *  \author G. Della Ricca
 */

#include <EventFilter/EcalTBRawToDigi/interface/EcalDCC07UnpackingModule.h>
#include <EventFilter/EcalTBRawToDigi/src/EcalTB07DaqFormatter.h>
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
#include <FWCore/ParameterSet/interface/FileInPath.h>

#include <iostream>
#include <iomanip>

// in full CMS this range cannot be used (allocated to pixel, see DataFormats/ FEDRawData/ src/ FEDNumbering.cc)
#define BEG_DCC_FED_ID 0
#define END_DCC_FED_ID 0
#define BEG_DCC_FED_ID_GLOBAL 0
#define END_DCC_FED_ID_GLOBAL 0

#define ECAL_SUPERVISOR_FED_ID 40
#define TBCAMAC_FED_ID 41
#define TABLE_FED_ID 42
#define MATACQ_FED_ID 43

EcalDCCTB07UnpackingModule::EcalDCCTB07UnpackingModule(const edm::ParameterSet& pset)
    : fedRawDataCollectionTag_(pset.getParameter<edm::InputTag>("fedRawDataCollectionTag")) {
  std::string tbName = pset.getUntrackedParameter<std::string>("tbName", std::string("h2"));

  ProduceEEDigis_ = pset.getUntrackedParameter<bool>("produceEEdigi", true);
  ProduceEBDigis_ = pset.getUntrackedParameter<bool>("produceEBdigi", false);

  // index of crystal <-> tower ID (DQM plots) position <-> stripIDs <-> channelIDs for the test beam (2007)
  std::vector<int> ics = pset.getUntrackedParameter<std::vector<int> >("ics", std::vector<int>());
  std::vector<int> towerIDs = pset.getUntrackedParameter<std::vector<int> >("towerIDs", std::vector<int>());
  std::vector<int> stripIDs = pset.getUntrackedParameter<std::vector<int> >("stripIDs", std::vector<int>());
  std::vector<int> channelIDs = pset.getUntrackedParameter<std::vector<int> >("channelIDs", std::vector<int>());

  // status id <-> tower CCU ID <-> DQM plots position mapping for the test beam (2007)
  std::vector<int> statusIDs = pset.getUntrackedParameter<std::vector<int> >("statusIDs", std::vector<int>());
  std::vector<int> ccuIDs = pset.getUntrackedParameter<std::vector<int> >("ccuIDs", std::vector<int>());
  std::vector<int> positionIDs = pset.getUntrackedParameter<std::vector<int> >("positionIDs", std::vector<int>());

  // check if vectors are filled
  if (ics.empty() || towerIDs.empty() || stripIDs.empty() || channelIDs.empty()) {
    edm::LogError("EcalDCCTB07UnpackingModule")
        << "Some of the mapping info is missing! Check config files! "
        << " Size of IC vector is " << ics.size() << " Size of Tower ID vector is " << towerIDs.size()
        << " Size of Strip ID vector is " << stripIDs.size() << " Size of Channel ID vector is " << channelIDs.size();
  }
  if (statusIDs.empty() || ccuIDs.empty() || positionIDs.empty()) {
    edm::LogError("EcalDCCTB07UnpackingModule")
        << "Some of the mapping info is missing! Check config files! "
        << " Size of status ID vector is " << statusIDs.size() << " Size of ccu ID vector is " << ccuIDs.size()
        << " positionIDs size is " << positionIDs.size();
  }

  // check if vectors have the same size
  if (ics.size() != towerIDs.size() || ics.size() != stripIDs.size() || ics.size() != channelIDs.size() ||
      towerIDs.size() != stripIDs.size() || towerIDs.size() != channelIDs.size() ||
      stripIDs.size() != channelIDs.size())
    edm::LogError("EcalDCCTB07UnpackingModule")
        << "Mapping information is corrupted. "
        << "Tower/DQM position/strip/channel vectors are of different size! Check cfi files! \n"
        << " Size of IC vector is " << ics.size() << " Size of Tower ID vector is " << towerIDs.size()
        << " Size of Strip ID vector is " << stripIDs.size() << " Size of Channel ID vector is " << channelIDs.size();

  if (statusIDs.size() != ccuIDs.size() || statusIDs.size() != positionIDs.size() ||
      ccuIDs.size() != positionIDs.size())
    edm::LogError("EcalDCCTB07UnpackingModule")
        << "Mapping information is corrupted. "
        << "Status/CCU ID/DQM position vectors are of different size! Check cfi files! \n"
        << " Size of status ID vector is " << statusIDs.size() << " Size of ccu ID vector is " << ccuIDs.size()
        << " positionIDs size is " << positionIDs.size();

  int cryIcMap[68][5][5];
  int tbStatusToLocation[71];
  int tbTowerIDToLocation[201];
  for (unsigned it = 1; it <= 68; ++it)
    for (unsigned is = 1; is <= 5; ++is)
      for (unsigned ic = 1; ic <= 5; ++ic)
        cryIcMap[it - 1][is - 1][ic - 1] = 1700;

  for (unsigned it = 1; it <= 71; ++it)
    tbStatusToLocation[it - 1] = it - 1;

  for (unsigned it = 1; it <= 201; ++it)
    tbTowerIDToLocation[it - 1] = it - 1;

  // Fill the cry IC map
  for (unsigned int i = 0; i < ics.size(); ++i) {
    int tower = towerIDs[i];
    int strip = stripIDs[i];
    int channel = channelIDs[i];
    int ic = ics[i];
    cryIcMap[tower - 1][strip - 1][channel - 1] = ic;
  }
  for (unsigned int i = 0; i < statusIDs.size(); ++i) {
    int is = statusIDs[i];
    int it = ccuIDs[i];
    int itEB = positionIDs[i];

    tbStatusToLocation[is] = itEB;
    tbTowerIDToLocation[it] = itEB;
  }

  formatter_ = new EcalTB07DaqFormatter(tbName, cryIcMap, tbStatusToLocation, tbTowerIDToLocation);
  ecalSupervisorFormatter_ = new EcalSupervisorTBDataFormatter();
  camacTBformatter_ = new CamacTBDataFormatter();
  tableFormatter_ = new TableDataFormatter();
  matacqFormatter_ = new MatacqTBDataFormatter();

  // digis
  produces<EBDigiCollection>("ebDigis");
  produces<EEDigiCollection>("eeDigis");
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

EcalDCCTB07UnpackingModule::~EcalDCCTB07UnpackingModule() { delete formatter_; }

void EcalDCCTB07UnpackingModule::produce(edm::Event& e, const edm::EventSetup& c) {
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByLabel(fedRawDataCollectionTag_, rawdata);

  // create the collection of Ecal Digis
  auto productEb = std::make_unique<EBDigiCollection>();

  // YM create the collection of Ecal Endcap Digis
  auto productEe = std::make_unique<EEDigiCollection>();

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
      //    edm::LogInfo("EcalDCCTB07UnpackingModule") << "EcalDCCTB07UnpackingModule::Got FED ID "<< id <<" ";
      const FEDRawData& data = rawdata->FEDData(id);
      //    edm::LogInfo("EcalDCCTB07UnpackingModule") << " Fed data size " << data.size() ;

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
            (BEG_DCC_FED_ID_GLOBAL <= id &&
             id <= END_DCC_FED_ID_GLOBAL)) {  // do the DCC data unpacking and fill the collections

          (*productHeader).setSmInBeam(id);
          // YM add productEe to the list of arguments of the formatter
          formatter_->interpretRawData(data,
                                       *productEb,
                                       *productEe,
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
          LogDebug("EcalDCCTB07UnpackingModule")
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
    if (ProduceEBDigis_)
      e.put(std::move(productEb), "ebDigis");
    if (ProduceEEDigis_)
      e.put(std::move(productEe), "eeDigis");
    e.put(std::move(productMatacq));
    e.put(std::move(productDCCHeader));
    e.put(std::move(productTriggerPrimitives), "EBTT");

    if (ProduceEBDigis_)
      e.put(std::move(productDCCSize), "EcalIntegrityDCCSizeErrors");
    if (ProduceEBDigis_)
      e.put(std::move(productTTId), "EcalIntegrityTTIdErrors");
    if (ProduceEBDigis_)
      e.put(std::move(productBlockSize), "EcalIntegrityBlockSizeErrors");
    if (ProduceEBDigis_)
      e.put(std::move(productChId), "EcalIntegrityChIdErrors");
    if (ProduceEBDigis_)
      e.put(std::move(productGain), "EcalIntegrityGainErrors");
    if (ProduceEBDigis_)
      e.put(std::move(productGainSwitch), "EcalIntegrityGainSwitchErrors");

    if (ProduceEBDigis_)
      e.put(std::move(productMemTtId), "EcalIntegrityMemTtIdErrors");
    if (ProduceEBDigis_)
      e.put(std::move(productMemBlockSize), "EcalIntegrityMemBlockSize");
    if (ProduceEBDigis_)
      e.put(std::move(productMemChIdErrors), "EcalIntegrityMemChIdErrors");
    if (ProduceEBDigis_)
      e.put(std::move(productMemGain), "EcalIntegrityMemGainErrors");

    e.put(std::move(productHodo));
    e.put(std::move(productTdc));
    e.put(std::move(productHeader));

  } catch (ECALTBParserException& e) {
    std::cout << "[EcalDCCTB07UnpackingModule] " << e.what() << std::endl;
  } catch (ECALTBParserBlockException& e) {
    std::cout << "[EcalDCCTB07UnpackingModule] " << e.what() << std::endl;
  } catch (cms::Exception& e) {
    std::cout << "[EcalDCCTB07UnpackingModule] " << e.what() << std::endl;
  } catch (...) {
    std::cout << "[EcalDCCTB07UnpackingModule] Unknown exception ..." << std::endl;
  }
}
