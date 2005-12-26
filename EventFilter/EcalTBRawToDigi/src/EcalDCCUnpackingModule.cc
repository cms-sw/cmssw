/* \file EcalDCCUnpackingModule.h
 *
 *  $Date: 2005/12/21 16:00:52 $
 *  $Revision: 1.19 $
 *  \author N. Marinelli
 *  \author G. Della Ricca
 *  \author G. Franzoni
 */

#include <EventFilter/EcalTBRawToDigi/interface/EcalDCCUnpackingModule.h>
#include <EventFilter/EcalTBRawToDigi/src/EcalTBDaqFormatter.h>
#include <EventFilter/EcalTBRawToDigi/src/ECALParserException.h>
#include <EventFilter/EcalTBRawToDigi/src/ECALParserBlockException.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>

#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>


using namespace edm;
using namespace std;


#include <iostream>

EcalDCCUnpackingModule::EcalDCCUnpackingModule(const edm::ParameterSet& pset){

  formatter = new EcalTBDaqFormatter();

  produces<EBDigiCollection>();
  produces<EcalPnDiodeDigiCollection>();

  produces<EBDetIdCollection>("EcalIntegrityDCCSizeErrors");
  produces<EcalTrigTowerDetIdCollection>("EcalIntegrityTTIdErrors");
  produces<EcalTrigTowerDetIdCollection>("EcalIntegrityBlockSizeErrors");
  produces<EBDetIdCollection>("EcalIntegrityChIdErrors");
  produces<EBDetIdCollection>("EcalIntegrityGainErrors");
  produces<EBDetIdCollection>("EcalIntegrityGainSwitchErrors");
  produces<EBDetIdCollection>("EcalIntegrityGainSwitchStayErrors");

}


EcalDCCUnpackingModule::~EcalDCCUnpackingModule(){

  delete formatter;

}

void EcalDCCUnpackingModule::beginJob(const edm::EventSetup& c){

}

void EcalDCCUnpackingModule::endJob(){

}

void EcalDCCUnpackingModule::produce(edm::Event & e, const edm::EventSetup& c){

  Handle<FEDRawDataCollection> rawdata;
  e.getByLabel("EcalDaqRawData", rawdata);
  
  // create the collection of Ecal Digis
  auto_ptr<EBDigiCollection> productEb(new EBDigiCollection);

  // create the collection of Ecal PN's
  auto_ptr<EcalPnDiodeDigiCollection> productPN(new EcalPnDiodeDigiCollection);

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

  try {

  for (unsigned int id= 0; id<=FEDNumbering::lastFEDId(); ++id){ 

    //     cout << "EcalDCCUnpackingModule::Got FED ID "<< id <<" ";
    const FEDRawData& data = rawdata->FEDData(id);
    // cout << " Fed data size " << data.size() << endl;
    
    if (data.size()){
      
      // do the data unpacking and fill the collections
      formatter->interpretRawData(data,  *productEb, *productPN, *productDCCSize, *productTTId, *productBlockSize, *productChId, *productGain, *productGainSwitch, *productGainSwitchStay);
 
    }// endif 
  }//endfor
  

  // commit to the event  
  e.put(productPN);
  e.put(productEb);

  e.put(productDCCSize,"EcalIntegrityDCCSizeErrors");
  e.put(productTTId,"EcalIntegrityTTIdErrors");
  e.put(productBlockSize,"EcalIntegrityBlockSizeErrors");
  e.put(productChId,"EcalIntegrityChIdErrors");
  e.put(productGain,"EcalIntegrityGainErrors");
  e.put(productGainSwitch,"EcalIntegrityGainSwitchErrors");
  e.put(productGainSwitchStay,"EcalIntegrityGainSwitchStayErrors");

  } catch (ECALParserException &e) {
    cout << e.what() << endl;
  } catch (ECALParserBlockException &e) {
    cout << e.what() << endl;
  } catch (...) {
    cout << "Unknown exception ..." << endl;
  }

}
