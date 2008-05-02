/** \file
 *
 *  $Date: 2008/03/05 12:43:09 $
 *  $Revision: 1.21 $
 *  \author A. Tumanov - Rice
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "EventFilter/CSCRawToDigi/src/CSCDigiToRaw.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include <boost/dynamic_bitset.hpp>
#include "EventFilter/CSCRawToDigi/src/bitset_append.h"
#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include "EventFilter/Utilities/interface/Crc.h"
#include "CondFormats/CSCObjects/interface/CSCChamberMap.h"
#include <algorithm>



using namespace edm;
using namespace std;

CSCDigiToRaw::CSCDigiToRaw(){}


CSCDigiToRaw::~CSCDigiToRaw(){}

map<CSCDetId, CSCEventData> 
CSCDigiToRaw::fillChamberDataMap(const CSCStripDigiCollection & stripDigis, 
                                 const CSCWireDigiCollection & wireDigis, 
                                 const CSCChamberMap* mapping) 
{
  map<CSCDetId, CSCEventData> chamberMap;
  //iterate over chambers with strip digis in them
  for (CSCStripDigiCollection::DigiRangeIterator j=stripDigis.begin(); j!=stripDigis.end(); ++j)
    {
      CSCDetId const cscDetId=(*j).first;
      CSCDetId chamberID =cscDetId.chamberId();
     
      bool me1a = (CSCDetId::station(cscDetId)==1) && (CSCDetId::ring(cscDetId)==4);
      bool zplus = (CSCDetId::endcap(cscDetId) == 1);
      bool me1b = (CSCDetId::station(cscDetId)==1) && (CSCDetId::ring(cscDetId)==1);

      //std::cout<<"strip id"<<cscDetId<<std::endl;
      // find the entry into the map
      map<CSCDetId, CSCEventData>::iterator chamberMapItr = chamberMap.find(chamberID);
      if(chamberMapItr == chamberMap.end())
        {
          // make an entry, telling it the correct chamberType
          int chamberType = chamberID.iChamberType();
          chamberMapItr = chamberMap.insert(pair<CSCDetId, CSCEventData>(chamberID, CSCEventData(chamberType))).first;
        }    
      CSCEventData & cscData = chamberMapItr->second;
      //std::cout<<"printing CSCDetId just before dmb" <<cscDetId<<std::endl;
      //std::cout<<"crate " <<mapping->crate(cscDetId)<<std::endl;
      //std::cout<<"dmb " <<mapping->dmb(cscDetId)<<std::endl;
      cscData.dmbHeader()->setCrateAddress(mapping->crate(cscDetId), mapping->dmb(cscDetId));
      //add strip digis to that chamber
      std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
      std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
      for( ; digiItr != last; ++digiItr) 
        {
	  CSCStripDigi digi = *digiItr;
	  int strip = digi.getStrip();
	  if ( me1a && zplus ) { digi.setStrip(17-strip); } // 1-16 -> 16-1
	  if ( me1b && !zplus) { digi.setStrip(65-strip);} // 1-64 -> 64-1
	  if ( me1a ) { strip = digi.getStrip(); digi.setStrip(strip+64);} // reset back 1-16 to 65-80 digi
          cscData.add(digi, cscDetId.layer() );    
        }
    }
  //repeat the same for wire digis
  for (CSCWireDigiCollection::DigiRangeIterator j=wireDigis.begin(); j!=wireDigis.end(); ++j) 
    {
      CSCDetId const cscDetId=(*j).first;
      CSCDetId chamberID =cscDetId.chamberId();
      //std::cout<<"wire id"<<cscDetId<<std::endl;
      // find the entry into the map
      map<CSCDetId, CSCEventData>::iterator chamberMapItr = chamberMap.find(chamberID);
      if(chamberMapItr == chamberMap.end())
        {
          // make an entry, telling it the correct chamberType
          int chamberType = chamberID.iChamberType();
          chamberMapItr = chamberMap.insert(pair<CSCDetId, CSCEventData>(chamberID, CSCEventData(chamberType))).first;
        }
      CSCEventData & cscData = chamberMapItr->second;
      cscData.dmbHeader()->setCrateAddress(mapping->crate(cscDetId), mapping->dmb(cscDetId));
      //add strip digis to that chamber
      std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
      std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;
      for( ; digiItr != last; ++digiItr) 
        {
          cscData.add(*digiItr, cscDetId.layer() );
        }
    }

  //std::cout<<"finished iterating and about to return the map size "<<chamberMap.size()<<std::endl;
  return chamberMap;
}


void CSCDigiToRaw::createFedBuffers(const CSCStripDigiCollection& stripDigis,
                                    const CSCWireDigiCollection& wireDigis,
                                    FEDRawDataCollection& fed_buffers,
                                    const CSCChamberMap* mapping,
                                    Event & e)
{

  //bits of code from ORCA/Muon/METBFormatter - thanks, Rick:)!
  
  //get fed object from fed_buffers
  // make a map from the index of a chamber to the event data from it
  map<CSCDetId, CSCEventData> chamberDataMap 
    = fillChamberDataMap(stripDigis, wireDigis, mapping);
  
  int l1a=1; //need to add increments or get it from lct digis 
  int bx = 0;//same as above
  //int startingFED = FEDNumbering::getCSCFEDIds().first;

  for (int idcc=FEDNumbering::getCSCFEDIds().first;
       idcc<=FEDNumbering::getCSCFEDIds().second;++idcc) 
    {
      //idcc goes from startingFed to startingFED+7
      // @@ if ReadoutMapping changes, this'll have to change
      // DCCs 1, 2,4,5have 5 DDUs.  Otherwise, 4
      //int nDDUs = (idcc < 2) || (idcc ==4) || (idcc ==5)
      //          ? 5 : 4; 
      //@@ WARNING some DCCs only have 4 DDUs, but I'm giving them all 5, for now
      int nDDUs = 5;
      // this is a vector of which ddu IDs correspond to the ones in the DCC
      vector<int> dduIds;
      dduIds.reserve(nDDUs);

      CSCDCCEventData dccEvent(idcc, nDDUs, bx, l1a);
      // for every chamber with data, add to a DDU in this DCC Event
      for(map<CSCDetId, CSCEventData>::iterator chamberItr = chamberDataMap.begin();
            chamberItr != chamberDataMap.end(); ++chamberItr)
        {

           //std::cout<<"inside the pack loop" <<std::endl;
           int indexDCC = mapping->slink(chamberItr->first);

           //std::cout<<" indexDCC=" << indexDCC <<std::endl;
           //std::cout<<" idcc = " <<idcc<<std::endl;
           //fill the right dcc
           if (idcc==indexDCC) 
             {
                // get ddu id based on ChamberId from mapping
                int dduId = mapping->ddu(chamberItr->first);
                vector<int>::iterator dduIdItr = find(dduIds.begin(), dduIds.end(), dduId);
                // if it's not found, it'll point at end(), so the distance should work out right.
                int indexDDU = distance(dduIds.begin(), dduIdItr);
                if(dduIdItr == dduIds.end()) 
                {
                  dduIds.push_back(dduId);
                }
                // just to make sure we're using STL right
                assert(dduIds.at(indexDDU) == dduId);

                dccEvent.dduData()[indexDDU].add(chamberItr->second);
                boost::dynamic_bitset<> dccBits = dccEvent.pack();
                FEDRawData & fedRawData = fed_buffers.FEDData(idcc);
                fedRawData.resize(dccBits.size());
                //fill data with dccEvent
                bitset_utilities::bitsetToChar(dccBits, fedRawData.data());
                FEDHeader cscFEDHeader(fedRawData.data());
                cscFEDHeader.set(fedRawData.data(), 0, e.id().event(), 0, idcc);
                FEDTrailer cscFEDTrailer(fedRawData.data()+(fedRawData.size()-8));
                cscFEDTrailer.set(fedRawData.data()+(fedRawData.size()-8), 
                                  fedRawData.size()/8, 
                                  evf::compute_crc(fedRawData.data(),fedRawData.size()), 0, 0);
             }
         }
      
    }

  // FIXME: FEDRawData size set to 2*64 to add FED header and trailer
  for (int idcc=FEDNumbering::getCSCFEDIds().first;
       idcc<=FEDNumbering::getCSCFEDIds().second;++idcc) 
    {
      FEDRawData & fedRawData = fed_buffers.FEDData(idcc);
      if(fedRawData.size()==0){
        fedRawData.resize(16);
        FEDHeader cscFEDHeader(fedRawData.data());
        cscFEDHeader.set(fedRawData.data(), 0, e.id().event(), 0, idcc);
        FEDTrailer cscFEDTrailer(fedRawData.data()+(fedRawData.size()-8));
        cscFEDTrailer.set(fedRawData.data()+(fedRawData.size()-8), 
                          fedRawData.size()/8, 
                          evf::compute_crc(fedRawData.data(),fedRawData.size()), 0, 0);
      }
    }

}



