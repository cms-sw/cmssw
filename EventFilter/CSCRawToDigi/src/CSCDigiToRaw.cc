/** \file
 *
 *  $Date: 2006/11/20 22:59:34 $
 *  $Revision: 1.4 $
 *  \author A. Tumanov - Rice
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "EventFilter/CSCRawToDigi/src/CSCDigiToRaw.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "CondFormats/CSCObjects/interface/CSCReadoutMappingFromFile.h"
#include <boost/dynamic_bitset.hpp>


using namespace edm;
using namespace std;

CSCDigiToRaw::CSCDigiToRaw(){}


CSCDigiToRaw::~CSCDigiToRaw(){}

map<CSCDetId, CSCEventData> 
CSCDigiToRaw::fillChamberDataMap(const CSCStripDigiCollection & stripDigis, 
				 const CSCWireDigiCollection & wireDigis, 
				 CSCReadoutMappingFromFile & mapping) {
 
  map<CSCDetId, CSCEventData> chamberMap;
  ///iterate over chambers with strip digis in them
  for (CSCStripDigiCollection::DigiRangeIterator j=stripDigis.begin(); j!=stripDigis.end(); j++) {
    CSCDetId const cscDetId=(*j).first;
    CSCDetId chamberID =cscDetId.chamberId();
    /// find the entry into the map
    map<CSCDetId, CSCEventData>::iterator chamberMapItr = chamberMap.find(chamberID);
    if(chamberMapItr == chamberMap.end()) {
      /// make an entry, telling it the correct chamberType
      int istation = cscDetId.station();
      int iring = cscDetId.ring();
      int chamberType = 2 * istation + iring; // i=2S+R
      if ( istation == 1 ) {
	--chamberType;             // ring 1R -> i=1+R (2S+R-1=1+R for S=1)
	if ( chamberType > 4 ) {
	  chamberType = 1;       // But ring 1A (R=4) -> i=1
	}
      }   
      chamberMapItr = chamberMap.insert(pair<CSCDetId, CSCEventData>(chamberID, CSCEventData(chamberType))).first;
    }    
    CSCEventData & cscData = chamberMapItr->second;
    cscData.dmbHeader().setCrateAddress(mapping.crate(cscDetId), mapping.dmbId(cscDetId));
    ///add strip digis to that chamber
    std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
      cscData.add(*digiItr, cscDetId.layer() );    
    }
  }
  ///repeat the same for wire digis
  for (CSCWireDigiCollection::DigiRangeIterator j=wireDigis.begin(); j!=wireDigis.end(); j++) {
    CSCDetId const cscDetId=(*j).first;
    CSCDetId chamberID =cscDetId.chamberId();
    /// find the entry into the map
    map<CSCDetId, CSCEventData>::iterator chamberMapItr = chamberMap.find(chamberID);
    if(chamberMapItr == chamberMap.end()) {
      /// make an entry, telling it the correct chamberType
      int istation = cscDetId.station();
      int iring = cscDetId.ring();
      int chamberType = 2 * istation + iring; // i=2S+R
      if ( istation == 1 ) {
        --chamberType;             // ring 1R -> i=1+R (2S+R-1=1+R for S=1)
        if ( chamberType > 4 ) {
          chamberType = 1;       // But ring 1A (R=4) -> i=1
        }
      }
      chamberMapItr = chamberMap.insert(pair<CSCDetId, CSCEventData>(chamberID, CSCEventData(chamberType))).first;
    }
    CSCEventData & cscData = chamberMapItr->second;
    cscData.dmbHeader().setCrateAddress(mapping.crate(cscDetId), mapping.dmbId(cscDetId));
    ///add strip digis to that chamber
    std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
      cscData.add(*digiItr, cscDetId.layer() );
    }
  }

  return chamberMap;

}




void CSCDigiToRaw::createFedBuffers(const CSCStripDigiCollection& stripDigis,
				    const CSCWireDigiCollection& wireDigis,
				    FEDRawDataCollection& fed_buffers,
				    CSCReadoutMappingFromFile& mapping){

  ///bits of code from ORCA/Muon/METBFormatter - thanks, Rick:)!
 
  ///get fed object from fed_buffers
  /// make a map from the index of a chamber to the event data from it
  map<CSCDetId, CSCEventData> chamberDataMap 
    = fillChamberDataMap(stripDigis, wireDigis, mapping);
 

  int l1a=1; ///need to add increments or get it from lct digis 
  int bx = 0;///same as above
  int startingFED = FEDNumbering::getCSCFEDIds().first;

  for (int idcc=FEDNumbering::getCSCFEDIds().first-startingFED;
       idcc<=FEDNumbering::getCSCFEDIds().second-startingFED;idcc++) {
    ///idcc goes from 0 to 7
    /// @@ if ReadoutMapping changes, this'll have to change
    /// DCCs 1, 2,4,5have 5 DDUs.  Otherwise, 4
    ///int nDDUs = (idcc < 2) || (idcc ==4) || (idcc ==5)
    ///          ? 5 : 4; 
    ///@@ WARNING some DCCs only have 4 DDUs, but I'm giving them all 5, for now
    int nDDUs = 5;
    int indexDDU=0;
    int oldDDUNumber = mapping.dduId(chamberDataMap.begin()->first);///initialize to first DDU

    CSCDCCEventData dccEvent(idcc, nDDUs, bx, l1a);
    /// for every chamber with data, add to a DDU in this DCC Event
    for(map<CSCDetId, CSCEventData>::iterator chamberItr = chamberDataMap.begin();
	chamberItr != chamberDataMap.end(); ++chamberItr)  {
      int indexDCC = mapping.dccId(chamberItr->first);
      if (idcc==indexDCC) { ///fill the right dcc 
	dccEvent.dduData()[indexDDU].add(chamberItr->second);
	boost::dynamic_bitset<> dccbits=dccEvent.pack();	
	FEDRawData * rawData = new FEDRawData(dccbits.size());
	unsigned char * data = rawData->data();
	for (unsigned int i=0;i<dccbits.size();i++) {//fill char data words bit by bit
	  data[i/8] = (dccbits[i]<<7)+
	              (dccbits[i+1]<<6)+
	              (dccbits[i+2]<<5)+
	              (dccbits[i+3]<<4)+
	              (dccbits[i+4]<<3)+
	              (dccbits[i+5]<<2)+
	              (dccbits[i+6]<<2)+
	              dccbits[i+7];
	  i+=7; ///jump by 8
	}
	int dduId = mapping.dduId(chamberItr->first); ///get ddu id based on ChamberId form mapping
	if (oldDDUNumber!=dduId) { //if new ddu increment indexDDU counter
	  indexDDU++;
	  oldDDUNumber = dduId;
	}
	FEDRawData& fedRawData = fed_buffers.FEDData(startingFED+idcc); 
	fedRawData = *rawData;
      }
    }
  }
}



