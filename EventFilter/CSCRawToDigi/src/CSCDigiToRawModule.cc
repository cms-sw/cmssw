/** \file
 *
 *  $Date: 2007/02/14 20:20:04 $
 *  $Revision: 1.4.2.1 $
 *  \author A. Tumanov - Rice
 */

#include <EventFilter/CSCRawToDigi/src/CSCDigiToRawModule.h>
#include <EventFilter/CSCRawToDigi/src/CSCDigiToRaw.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/CSCDigi/interface/CSCStripDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include "CondFormats/CSCObjects/interface/CSCReadoutMappingFromFile.h"

using namespace edm;
using namespace std;

CSCDigiToRawModule::CSCDigiToRawModule(const edm::ParameterSet & pset): 
  packer(new CSCDigiToRaw) {
  
  /*cout<<"here goes the mapping printout<<" <<endl;
  int dmb=0;
  int vme=60;
  for (int e=1;e<3;e++) {
    for (int s=1;s<5;s++) {
      for (int r=1;r<5;r++) {
	if ((r>2)&&(s>1)) continue;
	for (int c=1;c<37;c++) {
	  if ((s>1)&&(r==1)&&(c>18)) continue;
	  cout << e <<" "<< s <<" "<< r <<" "<< c <<" "<< (int)(vme++/10) <<" "<<(dmb++%10)<<" -1 1 1 1 0"<<endl;
	}
      }
    }
  }
  */
  theMapping  = CSCReadoutMappingFromFile(pset);
  digiCreator = pset.getUntrackedParameter<string>("DigiCreator", "cscunpacker");
  produces<FEDRawDataCollection>("CSCRawData"); 
}


CSCDigiToRawModule::~CSCDigiToRawModule(){
  delete packer;
}


void CSCDigiToRawModule::produce(Event & e, const EventSetup& c){


  auto_ptr<FEDRawDataCollection> fed_buffers(new FEDRawDataCollection);
  // Take digis from the event
  Handle<CSCStripDigiCollection> stripDigis;
  e.getByLabel(digiCreator,"MuonCSCStripDigi", stripDigis);
  Handle<CSCWireDigiCollection> wireDigis;
  e.getByLabel(digiCreator,"MuonCSCWireDigi", wireDigis);


  // Create the packed data
  packer->createFedBuffers(*stripDigis, *wireDigis, *(fed_buffers.get()), theMapping);
  
  // put the raw data to the event
  e.put(fed_buffers, "CSCRawData");
}


