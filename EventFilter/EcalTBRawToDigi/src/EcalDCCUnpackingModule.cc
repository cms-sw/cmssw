/* \file EcalDCCUnpackingModule.h
 *
 *  $Date: 2006/07/27 23:44:29 $
 *  $Revision: 1.27 $
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
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>


using namespace edm;
using namespace std;


#include <iostream>
#include <iomanip>

#define BEG_DCC_FED_ID 0
#define END_DCC_FED_ID 35
#define ECAL_SUPERVISOR_FED_ID 40 
#define TBCAMAC_FED_ID 41
#define TABLE_FED_ID 42
#define MATACQ_FED_ID 43

EcalDCCUnpackingModule::EcalDCCUnpackingModule(const edm::ParameterSet& pset){

  formatter_ = new EcalTBDaqFormatter();
  ecalSupervisorFormatter_ = new EcalSupervisorDataFormatter();
  camacTBformatter_ = new CamacTBDataFormatter();
  tableFormatter_ = new TableDataFormatter();
  matacqFormatter_ = new MatacqDataFormatter();

  // digis
  produces<EBDigiCollection>();
  produces<EcalMatacqDigiCollection>();
  produces<EcalPnDiodeDigiCollection>();
  produces<EcalRawDataCollection>();

  //TB specifics data
  produces<EcalTBHodoscopeRawInfo>();
  produces<EcalTBTDCRawInfo>();
  produces<EcalTBEventHeader>();

  // crystals' integrity
  produces<EBDetIdCollection>("EcalIntegrityDCCSizeErrors");
  produces<EcalTrigTowerDetIdCollection>("EcalIntegrityTTIdErrors");
  produces<EcalTrigTowerDetIdCollection>("EcalIntegrityBlockSizeErrors");
  produces<EBDetIdCollection>("EcalIntegrityChIdErrors");
  produces<EBDetIdCollection>("EcalIntegrityGainErrors");
  produces<EBDetIdCollection>("EcalIntegrityGainSwitchErrors");
  produces<EBDetIdCollection>("EcalIntegrityGainSwitchStayErrors");
  produces<EBDetIdCollection>("EcalIntegrityGainSwitchStayErrors");

  // mem channels' integrity
  produces<EcalElectronicsIdCollection>("EcalIntegrityMemTtIdErrors");
  produces<EcalElectronicsIdCollection>("EcalIntegrityMemBlockSize");
  produces<EcalElectronicsIdCollection>("EcalIntegrityMemChIdErrors");
  produces<EcalElectronicsIdCollection>("EcalIntegrityMemGainErrors");
}


EcalDCCUnpackingModule::~EcalDCCUnpackingModule(){

  delete formatter_;

}

void EcalDCCUnpackingModule::beginJob(const edm::EventSetup& c){

}

void EcalDCCUnpackingModule::endJob(){

}

void EcalDCCUnpackingModule::produce(edm::Event & e, const edm::EventSetup& c){

  Handle<FEDRawDataCollection> rawdata;
  e.getByType(rawdata);
  

  // create the collection of Ecal Digis
  auto_ptr<EBDigiCollection> productEb(new EBDigiCollection);

  // create the collection of Matacq Digi
  auto_ptr<EcalMatacqDigiCollection> productMatacq(new EcalMatacqDigiCollection());

  // create the collection of Ecal PN's
  auto_ptr<EcalPnDiodeDigiCollection> productPN(new EcalPnDiodeDigiCollection);
  
  //create the collection of Ecal DCC Header
  auto_ptr<EcalRawDataCollection> productDCCHeader(new EcalRawDataCollection);


  // create the collection of Ecal Integrity DCC Size
  auto_ptr<EBDetIdCollection> productDCCSize(new EBDetIdCollection);

  // create the collection of Ecal Integrity TT Id
  auto_ptr<EcalTrigTowerDetIdCollection> productTTId(new EcalTrigTowerDetIdCollection);

  // create the collection of Ecal Integrity TT Block Size
  auto_ptr<EcalTrigTowerDetIdCollection> productBlockSize(new EcalTrigTowerDetIdCollection);

  // create the collection of Ecal Integrity Ch Id
  auto_ptr<EBDetIdCollection> productChId(new EBDetIdCollection);

  // create the collection of Ecal Integrity Gain
  auto_ptr<EBDetIdCollection> productGain(new EBDetIdCollection);

  // create the collection of Ecal Integrity Gain Switch
  auto_ptr<EBDetIdCollection> productGainSwitch(new EBDetIdCollection);

  // create the collection of Ecal Integrity Gain Switch Stay
  auto_ptr<EBDetIdCollection> productGainSwitchStay(new EBDetIdCollection);

  // create the collection of Ecal Integrity Mem towerBlock_id errors
  auto_ptr<EcalElectronicsIdCollection> productMemTtId(new EcalElectronicsIdCollection);
  
  // create the collection of Ecal Integrity Mem gain errors
  auto_ptr< EcalElectronicsIdCollection> productMemBlockSize(new EcalElectronicsIdCollection);

  // create the collection of Ecal Integrity Mem gain errors
  auto_ptr< EcalElectronicsIdCollection> productMemGain(new EcalElectronicsIdCollection);

  // create the collection of Ecal Integrity Mem ch_id errors
  auto_ptr<EcalElectronicsIdCollection> productMemChIdErrors(new EcalElectronicsIdCollection);
  
  // create the collection of TB specifics data
  auto_ptr<EcalTBHodoscopeRawInfo> productHodo(new EcalTBHodoscopeRawInfo());         
  auto_ptr<EcalTBTDCRawInfo> productTdc(new EcalTBTDCRawInfo());                      
  auto_ptr<EcalTBEventHeader> productHeader(new EcalTBEventHeader());                      


  try {

  for (int id= 0; id<=FEDNumbering::lastFEDId(); ++id){ 

    //    LogInfo("EcalDCCUnpackingModule") << "EcalDCCUnpackingModule::Got FED ID "<< id <<" ";
    const FEDRawData& data = rawdata->FEDData(id);
    //    LogInfo("EcalDCCUnpackingModule") << " Fed data size " << data.size() ;
   
    //cout <<"1 Fed id: "<<dec<<id<< " Fed data size: " <<data.size() << endl;
//    const unsigned char * pData = data.data();
//    int length = data.size();
//    if(length >0 ){
//      if(length >= 40){length = 40;}
//    cout<<"##############################################################"<<endl;
//    for( int i=0; i<length; i++ ) {
//      std::cout << std::hex << std::setw(8) << int(pData[i]) << " ";
//      if( (i+1)%8 == 0 ) std::cout << std::endl;
//     }
//    cout<<"##############################################################"<<endl;
//    } 
    if (data.size()>16){

      if (id >= BEG_DCC_FED_ID && id <= END_DCC_FED_ID)
	{	// do the DCC data unpacking and fill the collections
	  
	  (*productHeader).setSmInBeam(id);
	  formatter_->interpretRawData(data,  *productEb, *productPN, 
				       *productDCCHeader, 
				       *productDCCSize, 
				       *productTTId, *productBlockSize, 
				       *productChId, *productGain, *productGainSwitch, *productGainSwitchStay, 
				       *productMemTtId,  *productMemBlockSize,
				       *productMemGain,  *productMemChIdErrors);
	  int runType = (*productDCCHeader)[0].getRunType();
	  if ( runType == EcalDCCHeaderBlock::COSMIC || runType == EcalDCCHeaderBlock::BEAMH4 ) 
	    (*productHeader).setTriggerMask(0x1);
	  else if ( runType == 4 || runType == 5 || runType == 6 ) //laser runs
	    (*productHeader).setTriggerMask(0x2000);
	  else if ( runType == 9 || runType == 10 || runType == 11 ) //pedestal runs
	    (*productHeader).setTriggerMask(0x800);
	  LogDebug("EcalDCCUnpackingModule") << "Event type is " << (*productHeader).eventType() << " dbEventType " << (*productHeader).dbEventType();
	} 
      else if ( id == ECAL_SUPERVISOR_FED_ID )
	ecalSupervisorFormatter_->interpretRawData(data, *productHeader);
      else if ( id == TBCAMAC_FED_ID )
	camacTBformatter_->interpretRawData(data, *productHeader,*productHodo, *productTdc );
      else if ( id == TABLE_FED_ID )
	tableFormatter_->interpretRawData(data, *productHeader);
      else if ( id == MATACQ_FED_ID )	  
	matacqFormatter_->interpretRawData(data, *productMatacq);
    }// endif 
  }//endfor
  

  // commit to the event  
  e.put(productPN);
  e.put(productEb);
  e.put(productMatacq);
  e.put(productDCCHeader);

  e.put(productDCCSize,"EcalIntegrityDCCSizeErrors");
  e.put(productTTId,"EcalIntegrityTTIdErrors");
  e.put(productBlockSize,"EcalIntegrityBlockSizeErrors");
  e.put(productChId,"EcalIntegrityChIdErrors");
  e.put(productGain,"EcalIntegrityGainErrors");
  e.put(productGainSwitch,"EcalIntegrityGainSwitchErrors");
  e.put(productGainSwitchStay,"EcalIntegrityGainSwitchStayErrors");

  e.put(productMemTtId,"EcalIntegrityMemTtIdErrors");
  e.put(productMemBlockSize,"EcalIntegrityMemBlockSize");
  e.put(productMemChIdErrors,"EcalIntegrityMemChIdErrors");
  e.put(productMemGain,"EcalIntegrityMemGainErrors");

  e.put(productHodo);
  e.put(productTdc);
  e.put(productHeader);

  } catch (ECALParserException &e) {
    cout << "[EcalDCCUnpackingModule] " << e.what() << endl;
  } catch (ECALParserBlockException &e) {
    cout << "[EcalDCCUnpackingModule] " << e.what() << endl;
  } catch (...) {
    cout << "[EcalDCCUnpackingModule] Unknown exception ..." << endl;
  }

}
