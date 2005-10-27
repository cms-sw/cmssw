/* \file EcalDCCUnpackingModule.h
 *
 *  $Date: 2005/10/18 09:06:15 $
 *  $Revision: 1.12 $
 *  \author N. Marinelli 
 *  \author G. Della Ricca
 */

#include <EventFilter/EcalTBRawToDigi/interface/EcalDCCUnpackingModule.h>
#include <EventFilter/EcalTBRawToDigi/src/EcalTBDaqFormatter.h>
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

  outputFile = pset.getUntrackedParameter<string>("outputFile", "");

  if ( outputFile.size() != 0 ) {
    cout << "Ecal Integrity histograms will be saved to " << outputFile.c_str() << endl;
  }

  dbe = 0;
  if ( pset.getUntrackedParameter<bool>("DBEinterface", false) ) {
    dbe = edm::Service<DaqMonitorBEInterface>().operator->();

    dbe->setVerbose(0);

    edm::Service<MonitorDaemon> daemon;
    daemon.operator->();
  }

  formatter = new EcalTBDaqFormatter(dbe);

  produces<EBDigiCollection>();

}


EcalDCCUnpackingModule::~EcalDCCUnpackingModule(){

  delete formatter;

}

void EcalDCCUnpackingModule::beginJob(const edm::EventSetup& c){

}

void EcalDCCUnpackingModule::endJob(){

  if ( outputFile.size() != 0 && dbe ) dbe->save(outputFile);

}

void EcalDCCUnpackingModule::produce(edm::Event & e, const edm::EventSetup& c){

  Handle<FEDRawDataCollection> rawdata;
  e.getByLabel("EcalDaqRawData", rawdata);
  
  // create the collection of Ecal Digis
  auto_ptr<EBDigiCollection> product(new EBDigiCollection);

  for (unsigned int id= 0; id<=FEDNumbering::lastFEDId(); ++id){ 

    //     cout << "EcalDCCUnpackingModule::Got FED ID "<< id <<" ";
    const FEDRawData& data = rawdata->FEDData(id);
    // cout << " Fed data size " << data.size() << endl;
    
    if (data.size()){
      
      // do the conversion and fill the container
      formatter->interpretRawData(data,  *product );
    }// endif 
  }//endfor
  

  // commit to the event  

  e.put(product);



}
