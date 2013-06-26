/* \file EcalDCCUnpackingModule.h
 *
 *  $Date: 2012/09/12 18:18:44 $
 *  $Revision: 1.45 $
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

EcalDCCTBUnpackingModule::EcalDCCTBUnpackingModule(const edm::ParameterSet& pset) :
  fedRawDataCollectionTag_(pset.getParameter<edm::InputTag>("fedRawDataCollectionTag")) {

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


EcalDCCTBUnpackingModule::~EcalDCCTBUnpackingModule(){

  delete formatter_;

}

void EcalDCCTBUnpackingModule::beginJob(){

}

void EcalDCCTBUnpackingModule::endJob(){

}

void EcalDCCTBUnpackingModule::produce(edm::Event & e, const edm::EventSetup& c){

  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByLabel(fedRawDataCollectionTag_, rawdata);
  

  // create the collection of Ecal Digis
  std::auto_ptr<EBDigiCollection> productEb(new EBDigiCollection);

  // create the collection of Matacq Digi
  std::auto_ptr<EcalMatacqDigiCollection> productMatacq(new EcalMatacqDigiCollection());

  // create the collection of Ecal PN's
  std::auto_ptr<EcalPnDiodeDigiCollection> productPN(new EcalPnDiodeDigiCollection);
  
  //create the collection of Ecal DCC Header
  std::auto_ptr<EcalRawDataCollection> productDCCHeader(new EcalRawDataCollection);

  // create the collection with trigger primitives, bits and flags
  std::auto_ptr<EcalTrigPrimDigiCollection> productTriggerPrimitives(new EcalTrigPrimDigiCollection);

  // create the collection of Ecal Integrity DCC Size
  std::auto_ptr<EBDetIdCollection> productDCCSize(new EBDetIdCollection);

  // create the collection of Ecal Integrity TT Id
  std::auto_ptr<EcalElectronicsIdCollection> productTTId(new EcalElectronicsIdCollection);

  // create the collection of Ecal Integrity TT Block Size
  std::auto_ptr<EcalElectronicsIdCollection> productBlockSize(new EcalElectronicsIdCollection);

  // create the collection of Ecal Integrity Ch Id
  std::auto_ptr<EBDetIdCollection> productChId(new EBDetIdCollection);

  // create the collection of Ecal Integrity Gain
  std::auto_ptr<EBDetIdCollection> productGain(new EBDetIdCollection);

  // create the collection of Ecal Integrity Gain Switch
  std::auto_ptr<EBDetIdCollection> productGainSwitch(new EBDetIdCollection);

  // create the collection of Ecal Integrity Mem towerBlock_id errors
  std::auto_ptr<EcalElectronicsIdCollection> productMemTtId(new EcalElectronicsIdCollection);
  
  // create the collection of Ecal Integrity Mem gain errors
  std::auto_ptr< EcalElectronicsIdCollection> productMemBlockSize(new EcalElectronicsIdCollection);

  // create the collection of Ecal Integrity Mem gain errors
  std::auto_ptr< EcalElectronicsIdCollection> productMemGain(new EcalElectronicsIdCollection);
  
  // create the collection of Ecal Integrity Mem ch_id errors
  std::auto_ptr<EcalElectronicsIdCollection> productMemChIdErrors(new EcalElectronicsIdCollection);
  
  // create the collection of TB specifics data
  std::auto_ptr<EcalTBHodoscopeRawInfo> productHodo(new EcalTBHodoscopeRawInfo());         
  std::auto_ptr<EcalTBTDCRawInfo> productTdc(new EcalTBTDCRawInfo());                      
  std::auto_ptr<EcalTBEventHeader> productHeader(new EcalTBEventHeader());                      


  try {

  for (int id= 0; id<=FEDNumbering::MAXFEDID; ++id){ 

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
    if (data.size()>16){

      if ( (id >= BEG_DCC_FED_ID && id <= END_DCC_FED_ID) ||
	   (id >= BEG_DCC_FED_ID_GLOBAL && id <= END_DCC_FED_ID_GLOBAL)
	 )
	{	// do the DCC data unpacking and fill the collections
	  
	  (*productHeader).setSmInBeam(id);
	  formatter_->interpretRawData(data,  *productEb, *productPN, 
				       *productDCCHeader, 
				       *productDCCSize, 
				       *productTTId, *productBlockSize, 
				       *productChId, *productGain, *productGainSwitch, 
				       *productMemTtId,  *productMemBlockSize,
				       *productMemGain,  *productMemChIdErrors,
				       *productTriggerPrimitives);
	  int runType = (*productDCCHeader)[0].getRunType();
	  if ( runType == EcalDCCHeaderBlock::COSMIC || runType == EcalDCCHeaderBlock::BEAMH4 ) 
	    (*productHeader).setTriggerMask(0x1);
	  else if ( runType == 4 || runType == 5 || runType == 6 ) //laser runs
	    (*productHeader).setTriggerMask(0x2000);
	  else if ( runType == 9 || runType == 10 || runType == 11 ) //pedestal runs
	    (*productHeader).setTriggerMask(0x800);
	  LogDebug("EcalDCCTBUnpackingModule") << "Event type is " << (*productHeader).eventType() << " dbEventType " << (*productHeader).dbEventType();
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
  e.put(productEb,"ebDigis");
  e.put(productMatacq);
  e.put(productDCCHeader);
  e.put(productTriggerPrimitives,"EBTT");

  e.put(productDCCSize,"EcalIntegrityDCCSizeErrors");
  e.put(productTTId,"EcalIntegrityTTIdErrors");
  e.put(productBlockSize,"EcalIntegrityBlockSizeErrors");
  e.put(productChId,"EcalIntegrityChIdErrors");
  e.put(productGain,"EcalIntegrityGainErrors");
  e.put(productGainSwitch,"EcalIntegrityGainSwitchErrors");

  e.put(productMemTtId,"EcalIntegrityMemTtIdErrors");
  e.put(productMemBlockSize,"EcalIntegrityMemBlockSize");
  e.put(productMemChIdErrors,"EcalIntegrityMemChIdErrors");
  e.put(productMemGain,"EcalIntegrityMemGainErrors");

  e.put(productHodo);
  e.put(productTdc);
  e.put(productHeader);

  } catch (ECALTBParserException &e) {
    std::cout << "[EcalDCCTBUnpackingModule] " << e.what() << std::endl;
  } catch (ECALTBParserBlockException &e) {
    std::cout << "[EcalDCCTBUnpackingModule] " << e.what() << std::endl;
  } catch (cms::Exception &e) {
    std::cout << "[EcalDCCTBUnpackingModule] " << e.what() << std::endl;
  } catch (...) {
    std::cout << "[EcalDCCTBUnpackingModule] Unknown exception ..." << std::endl;
  }

}
