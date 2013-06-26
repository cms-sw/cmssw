/** \file
 *
 *  $Date: 2011/06/22 23:53:24 $
 *  $Revision: 1.43 $
 *  \author A. Tumanov - Rice
 *  But long, long ago...
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "EventFilter/CSCRawToDigi/src/CSCDigiToRaw.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "boost/dynamic_bitset.hpp"
#include "boost/foreach.hpp"
#include "EventFilter/CSCRawToDigi/src/bitset_append.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "FWCore/Utilities/interface/CRC16.h"
#include "CondFormats/CSCObjects/interface/CSCChamberMap.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include <algorithm>

using namespace edm;
using namespace std;

namespace cscd2r {
  /// takes layer ID, converts to chamber ID, switching ME1A to ME11
  CSCDetId chamberID(const CSCDetId & cscDetId) {
    CSCDetId chamberId = cscDetId.chamberId();
    if(chamberId.ring() ==4)
    {
      chamberId = CSCDetId(chamberId.endcap(), chamberId.station(), 1, chamberId.chamber(), 0);
    }
    return chamberId;
  }

  template<typename LCTCollection>
  bool accept(const CSCDetId & cscId, const LCTCollection & lcts,
         int bxMin, int bxMax)
  {
    if(bxMin == -999) return true;
    int nominalBX = 6;
    CSCDetId chamberId = chamberID(cscId);
    typename LCTCollection::Range lctRange = lcts.get(chamberId);
    bool result = false;
    for(typename LCTCollection::const_iterator lctItr = lctRange.first;
        lctItr != lctRange.second; ++lctItr)
    {
      int bx = lctItr->getBX() - nominalBX;
      if(bx >= bxMin && bx <= bxMax)
      {
        result = true;
        break;
      }
    }
    return result;
  }

  // need to specialize for pretriggers, since they don't have a getBX()
  template<>
  bool accept(const CSCDetId & cscId, const CSCCLCTPreTriggerCollection & lcts,
         int bxMin, int bxMax)
  {
    if(bxMin == -999) return true;
    int nominalBX = 6;
    CSCDetId chamberId = chamberID(cscId);
    CSCCLCTPreTriggerCollection::Range lctRange = lcts.get(chamberId);
    bool result = false;
    for(CSCCLCTPreTriggerCollection::const_iterator lctItr = lctRange.first;
        lctItr != lctRange.second; ++lctItr)
    {
      int bx = *lctItr - nominalBX;
      if(bx >= bxMin && bx <= bxMax)
      {
        result = true;
        break;
      }
    }
    return result;
  }

}


CSCDigiToRaw::CSCDigiToRaw(const edm::ParameterSet & pset)
: alctWindowMin_(pset.getParameter<int>("alctWindowMin")),
  alctWindowMax_(pset.getParameter<int>("alctWindowMax")),
  clctWindowMin_(pset.getParameter<int>("clctWindowMin")),
  clctWindowMax_(pset.getParameter<int>("clctWindowMax")),
  preTriggerWindowMin_(pset.getParameter<int>("preTriggerWindowMin")),
  preTriggerWindowMax_(pset.getParameter<int>("preTriggerWindowMax"))
{}

void CSCDigiToRaw::beginEvent(const CSCChamberMap* electronicsMap)
{
  theChamberDataMap.clear();
  theElectronicsMap = electronicsMap;
}


CSCEventData & CSCDigiToRaw::findEventData(const CSCDetId & cscDetId) 
{
  CSCDetId chamberId = cscd2r::chamberID(cscDetId);
  // find the entry into the map
  map<CSCDetId, CSCEventData>::iterator chamberMapItr = theChamberDataMap.find(chamberId);
  if(chamberMapItr == theChamberDataMap.end())
    {
      // make an entry, telling it the correct chamberType
      int chamberType = chamberId.iChamberType();
      chamberMapItr = theChamberDataMap.insert(pair<CSCDetId, CSCEventData>(chamberId, CSCEventData(chamberType))).first;
    }
  CSCEventData & cscData = chamberMapItr->second;
  cscData.dmbHeader()->setCrateAddress(theElectronicsMap->crate(cscDetId), theElectronicsMap->dmb(cscDetId));
  return cscData;
}


  
void CSCDigiToRaw::add(const CSCStripDigiCollection& stripDigis, 
                       const CSCCLCTPreTriggerCollection & preTriggers)
{  //iterate over chambers with strip digis in them
  for (CSCStripDigiCollection::DigiRangeIterator j=stripDigis.begin(); j!=stripDigis.end(); ++j)
    {
      CSCDetId cscDetId=(*j).first;
      // only digitize if there are pre-triggers
      if(cscd2r::accept(cscDetId, preTriggers, preTriggerWindowMin_, preTriggerWindowMax_))
      {
        bool me1a = (cscDetId.station()==1) && (cscDetId.ring()==4);
        bool zplus = (cscDetId.endcap() == 1);
        bool me1b = (cscDetId.station()==1) && (cscDetId.ring()==1);

        CSCEventData & cscData = findEventData(cscDetId);

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
    }
}


void CSCDigiToRaw::add(const CSCWireDigiCollection& wireDigis, 
                       const CSCALCTDigiCollection & alctDigis) 
{
  add(alctDigis);
  for (CSCWireDigiCollection::DigiRangeIterator j=wireDigis.begin(); j!=wireDigis.end(); ++j)
    {
      CSCDetId cscDetId=(*j).first;
      if(cscd2r::accept(cscDetId, alctDigis, alctWindowMin_, alctWindowMax_))
      {
        CSCEventData & cscData = findEventData(cscDetId);
        std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
        std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;
        for( ; digiItr != last; ++digiItr)
        {
          cscData.add(*digiItr, cscDetId.layer() );
        }
      }
    }

}

void CSCDigiToRaw::add(const CSCComparatorDigiCollection & comparatorDigis,
                       const CSCCLCTDigiCollection & clctDigis)
{
  add(clctDigis);
  for (CSCComparatorDigiCollection::DigiRangeIterator j=comparatorDigis.begin(); j!=comparatorDigis.end(); ++j)
    {
      CSCDetId cscDetId=(*j).first;
      CSCEventData & cscData = findEventData(cscDetId);
      if(cscd2r::accept(cscDetId, clctDigis, clctWindowMin_, clctWindowMax_))
      {
        bool me1a = (cscDetId.station()==1) && (cscDetId.ring()==4);

        BOOST_FOREACH(CSCComparatorDigi digi, (*j).second)
        {
	  // Move ME1/A comparators from CFEB=0 to CFEB=4 if this has not
	  // been done already.
	  if (me1a && digi.getStrip() <= 16) {
	    CSCComparatorDigi digi_corr(64+digi.getStrip(),
					digi.getComparator(),
					digi.getTimeBinWord());
	    cscData.add(digi_corr, cscDetId.layer());
	  }
	  else {
	    cscData.add(digi, cscDetId.layer());
	  }
        }
      }
    }
}

void CSCDigiToRaw::add(const CSCALCTDigiCollection & alctDigis)
{
  for (CSCALCTDigiCollection::DigiRangeIterator j=alctDigis.begin(); j!=alctDigis.end(); ++j)
    {
      CSCDetId cscDetId=(*j).first;
      CSCEventData & cscData = findEventData(cscDetId);

       cscData.add(std::vector<CSCALCTDigi>((*j).second.first, (*j).second.second));
    }
}

void CSCDigiToRaw::add(const CSCCLCTDigiCollection & clctDigis)
{
  for (CSCCLCTDigiCollection::DigiRangeIterator j=clctDigis.begin(); j!=clctDigis.end(); ++j)
    {
      CSCDetId cscDetId=(*j).first;
      CSCEventData & cscData = findEventData(cscDetId);

      cscData.add(std::vector<CSCCLCTDigi>((*j).second.first, (*j).second.second));
    }
}

void CSCDigiToRaw::add(const CSCCorrelatedLCTDigiCollection & corrLCTDigis)
{
  for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator j=corrLCTDigis.begin(); j!=corrLCTDigis.end(); ++j)
    {
      CSCDetId cscDetId=(*j).first;
      CSCEventData & cscData = findEventData(cscDetId);

      cscData.add(std::vector<CSCCorrelatedLCTDigi>((*j).second.first, (*j).second.second));
    }

}


void CSCDigiToRaw::createFedBuffers(const CSCStripDigiCollection& stripDigis,
                                    const CSCWireDigiCollection& wireDigis,
                                    const CSCComparatorDigiCollection& comparatorDigis,
                                    const CSCALCTDigiCollection& alctDigis,
                                    const CSCCLCTDigiCollection& clctDigis,
                                    const CSCCLCTPreTriggerCollection & preTriggers,
                                    const CSCCorrelatedLCTDigiCollection& correlatedLCTDigis,
                                    FEDRawDataCollection& fed_buffers,
                                    const CSCChamberMap* mapping,
                                    Event & e)
{

  //bits of code from ORCA/Muon/METBFormatter - thanks, Rick:)!
  
  //get fed object from fed_buffers
  // make a map from the index of a chamber to the event data from it
  beginEvent(mapping);
  add(stripDigis, preTriggers);
  add(wireDigis, alctDigis);
  add(comparatorDigis, clctDigis);
  add(correlatedLCTDigis);
  
  int l1a=e.id().event(); //need to add increments or get it from lct digis 
  int bx = 0;//same as above
  //int startingFED = FEDNumbering::MINCSCFEDID;

  std::map<int, CSCDCCEventData> dccMap;
  for (int idcc=FEDNumbering::MINCSCFEDID;
       idcc<=FEDNumbering::MAXCSCFEDID;++idcc) 
  {
    //idcc goes from startingFed to startingFED+7
    // @@ if ReadoutMapping changes, this'll have to change
    // DCCs 1,2,4,5 have 5 DDUs.  Otherwise, 4
    //int nDDUs = (idcc < 2) || (idcc ==4) || (idcc ==5)
    //          ? 5 : 4; 
    //@@ WARNING some DCCs only have 4 DDUs, but I'm giving them all 5, for now
    int nDDUs = 5;
    dccMap.insert(std::pair<int, CSCDCCEventData>(idcc, CSCDCCEventData(idcc, nDDUs, bx, l1a) ) );

    // for every chamber with data, add to a DDU in this DCC Event
    for(map<CSCDetId, CSCEventData>::iterator chamberItr = theChamberDataMap.begin();
        chamberItr != theChamberDataMap.end(); ++chamberItr)
    {
      //std::cout<<"inside the pack loop" <<std::endl;
      int indexDCC = mapping->slink(chamberItr->first);
      if(indexDCC == idcc)
      {
        //FIXME (What does this mean? Is something wrong?)
        std::map<int, CSCDCCEventData>::iterator dccMapItr = dccMap.find(indexDCC);
        if(dccMapItr == dccMap.end())
        {
          throw cms::Exception("CSCDigiToRaw") << "Bad DCC number:" << indexDCC;
        } 
        // get id's based on ChamberId from mapping

        int dduId    = mapping->ddu(chamberItr->first);
        int dduSlot  = mapping->dduSlot(chamberItr->first);
        int dduInput = mapping->dduInput(chamberItr->first);
        int dmbId    = mapping->dmb(chamberItr->first);
        dccMapItr->second.addChamber(chamberItr->second, dduId, dduSlot, dduInput, dmbId);
      }
    }
  }
      
  // FIXME: FEDRawData size set to 2*64 to add FED header and trailer
  for(std::map<int, CSCDCCEventData>::iterator dccMapItr = dccMap.begin(); 
      dccMapItr != dccMap.end(); ++dccMapItr)
  {
     boost::dynamic_bitset<> dccBits = dccMapItr->second.pack();
     FEDRawData & fedRawData = fed_buffers.FEDData(dccMapItr->first);
     fedRawData.resize(dccBits.size());
     //fill data with dccEvent
     bitset_utilities::bitsetToChar(dccBits, fedRawData.data());
     FEDTrailer cscFEDTrailer(fedRawData.data()+(fedRawData.size()-8));
     cscFEDTrailer.set(fedRawData.data()+(fedRawData.size()-8), 
                       fedRawData.size()/8, 
                       evf::compute_crc(fedRawData.data(),fedRawData.size()), 0, 0);
  }
}



