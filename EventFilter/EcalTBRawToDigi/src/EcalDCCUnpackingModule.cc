/* \file EcalDCCUnpackingModule.h
 *
 *  $Date: 2005/10/11 06:56:58 $
 *  $Revision: 1.10 $
 *  \author N. Marinelli 
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

  string outputFile = pset.getUntrackedParameter<string>("outputFile", "");

  rootFile = 0;
  if ( outputFile.size() != 0 ) {
    cout << "Ecal Integrity histograms will be saved to " << outputFile.c_str() << endl;
    rootFile = new TFile(outputFile.c_str(), "recreate");
    rootFile->cd();
  }

  formatter = new EcalTBDaqFormatter(rootFile);

  produces<EBDigiCollection>();

}


EcalDCCUnpackingModule::~EcalDCCUnpackingModule(){

  if ( rootFile ) {
    rootFile->Write();
    rootFile->Close();
    delete rootFile;
  }

  delete formatter;
}


void EcalDCCUnpackingModule::produce(Event & e, const EventSetup& c){

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
