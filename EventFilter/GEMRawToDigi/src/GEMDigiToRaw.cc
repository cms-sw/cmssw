/** \file
 *  \based on CSCDigiToRaw
 *  \author J. Lee - UoS
 */

#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "EventFilter/GEMRawToDigi/src/GEMDigiToRaw.h"
//#include "EventFilter/GEMRawToDigi/interface/GEMDCCEventData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "boost/dynamic_bitset.hpp"
#include "boost/foreach.hpp"
#include "EventFilter/GEMRawToDigi/src/bitset_append.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "FWCore/Utilities/interface/CRC16.h"
#include "CondFormats/GEMObjects/interface/GEMEMap.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include <algorithm>

using namespace edm;
using namespace std;

void GEMDigiToRaw::beginEvent(const GEMChamberMap* electronicsMap)
{
  theChamberDataMap.clear();
  theElectronicsMap = electronicsMap;
}

GEMEventData & GEMDigiToRaw::findEventData(const GEMDetId & gemDetId)
{
  GEMDetId chamberId = gemd2r::chamberID(gemDetId);
  // find the entry into the map
  map<GEMDetId, GEMEventData>::iterator chamberMapItr = theChamberDataMap.find(chamberId);
  if (chamberMapItr == theChamberDataMap.end())
    {
      // make an entry, telling it the correct chamberType
      int chamberType = chamberId.iChamberType();
      chamberMapItr = theChamberDataMap.insert(pair<GEMDetId, GEMEventData>(chamberId, GEMEventData(chamberType, formatVersion_))).first;
    }
  GEMEventData & gemData = chamberMapItr->second;
  gemData.dmbHeader()->setCrateAddress(theElectronicsMap->crate(gemDetId), theElectronicsMap->dmb(gemDetId));

  if (formatVersion_ == 2013)
    {
      // Set DMB version field to distinguish between ME11s and other chambers (ME11 - 2, others - 1)
      bool me11 = ((chamberId.station()==1) && (chamberId.ring()==4)) || ((chamberId.station()==1) && (chamberId.ring()==1));
      if (me11)
        {
          gemData.dmbHeader()->setdmbVersion(2);
        }
      else
        {
          gemData.dmbHeader()->setdmbVersion(1);
        }

    }
  return gemData;
}

void GEMDigiToRaw::add(const GEMDigiCollection& digis)
{  //iterate over chambers with strip digis in them
  for (GEMDigiCollection::DigiRangeIterator cItr=digis->begin(); cItr!=digis->end(); cItr++) {
    GEMDetId gemDetId = (*cItr).first;
    GEMEventData & gemData = findEventData(gemDetId);

    GEMDigiCollection::const_iterator digiItr;
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr) {

      gemData.add(*digiItr, gemDetId);      
    }
  }  
}

void GEMDigiToRaw::createFedBuffers(const GEMDigiCollection& gemDigi,
				    const GEMPadDigiCollection& gemPadDigi,
				    const GEMPadDigiClusterCollection& gemPadDigiCluster,
				    const GEMCoPadDigiCollection& gemCoPadDigi,			
				    FEDRawDataCollection& fed_buffers,
				    const GEMChamberMap* theMapping, 
				    edm::Event & e)
{
  beginEvent(mapping);
  add(gemDigi);
  add(gemPadDigi);
  add(gemPadDigiCluster);
  add(gemCoPadDigi);

  
  int l1a=e.id().event(); //need to add increments or get it from lct digis
  int bx = l1a;//same as above
  //int startingFED = FEDNumbering::MINGEMFEDID;

  if (formatVersion_ == 2005) /// Handle pre-LS1 format data
    {
      std::map<int, GEMDCCEventData> dccMap;
      for (int idcc=FEDNumbering::MINGEMFEDID;
           idcc<=FEDNumbering::MAXGEMFEDID; ++idcc)
        {
          //idcc goes from startingFed to startingFED+7
          // @@ if ReadoutMapping changes, this'll have to change
          // DCCs 1,2,4,5 have 5 DDUs.  Otherwise, 4
          //int nDDUs = (idcc < 2) || (idcc ==4) || (idcc ==5)
          //          ? 5 : 4;
          //@@ WARNING some DCCs only have 4 DDUs, but I'm giving them all 5, for now
          int nDDUs = 5;
          dccMap.insert(std::pair<int, GEMDCCEventData>(idcc, GEMDCCEventData(idcc, nDDUs, bx, l1a) ) );
        }

      for (int idcc=FEDNumbering::MINGEMFEDID;
           idcc<=FEDNumbering::MAXGEMFEDID; ++idcc)
        {

          // for every chamber with data, add to a DDU in this DCC Event
          for (map<GEMDetId, GEMEventData>::iterator chamberItr = theChamberDataMap.begin();
               chamberItr != theChamberDataMap.end(); ++chamberItr)
            {
              int indexDCC = mapping->slink(chamberItr->first);
              if (indexDCC == idcc)
                {
                  //FIXME (What does this mean? Is something wrong?)
                  std::map<int, GEMDCCEventData>::iterator dccMapItr = dccMap.find(indexDCC);
                  if (dccMapItr == dccMap.end())
                    {
                      throw cms::Exception("GEMDigiToRaw") << "Bad DCC number:" << indexDCC;
                    }
                  // get id's based on ChamberId from mapping

                  int dduId    = mapping->ddu(chamberItr->first);
                  int dduSlot  = mapping->dduSlot(chamberItr->first);
                  int dduInput = mapping->dduInput(chamberItr->first);
                  int dmbId    = mapping->dmb(chamberItr->first);
                  dccMapItr->second.addChamber(chamberItr->second, dduId, dduSlot, dduInput, dmbId, formatVersion_);
                }
            }
        }

      // FIXME: FEDRawData size set to 2*64 to add FED header and trailer
      for (std::map<int, GEMDCCEventData>::iterator dccMapItr = dccMap.begin();
           dccMapItr != dccMap.end(); ++dccMapItr)
        {
          boost::dynamic_bitset<> dccBits = dccMapItr->second.pack();
          FEDRawData & fedRawData = fed_buffers.FEDData(dccMapItr->first);
          fedRawData.resize(dccBits.size());
          //fill data with dccEvent
          bitset_utilities::bitsetToChar(dccBits, fedRawData.data());
          FEDTrailer gemFEDTrailer(fedRawData.data()+(fedRawData.size()-8));
          gemFEDTrailer.set(fedRawData.data()+(fedRawData.size()-8),
                            fedRawData.size()/8,
                            evf::compute_crc(fedRawData.data(),fedRawData.size()), 0, 0);
        }

    }
  else if (formatVersion_ == 2013) /// Handle post-LS1 format data
    {

      std::map<int, GEMDDUEventData> dduMap;
      unsigned int ddu_fmt_version = 0x7; /// 2013 Format
      const unsigned postLS1_map [] = { 841, 842, 843, 844, 845, 846, 847, 848, 849,
					831, 832, 833, 834, 835, 836, 837, 838, 839,
					861, 862, 863, 864, 865, 866, 867, 868, 869,
					851, 852, 853, 854, 855, 856, 857, 858, 859 };


      /// Create dummy DDU buffers
      for (unsigned int i = 0; i < 36; i++)
        {
	  unsigned int iddu = postLS1_map[i];
          // make a new one
          GEMDDUHeader newDDUHeader(bx,l1a, iddu, ddu_fmt_version);

          dduMap.insert(std::pair<int, GEMDDUEventData>(iddu, GEMDDUEventData(newDDUHeader) ) );
        }


      /// Loop over post-LS1 DDU FEDs
      for (unsigned int i = 0; i < 36; i++)
        {
	  unsigned int iddu = postLS1_map[i];
          // for every chamber with data, add to a DDU in this DCC Event
          for (map<GEMDetId, GEMEventData>::iterator chamberItr = theChamberDataMap.begin();
               chamberItr != theChamberDataMap.end(); ++chamberItr)
            {
              unsigned int indexDDU = mapping->slink(chamberItr->first);

              /// Lets handle possible mapping issues
              if ( (indexDDU >= FEDNumbering::MINGEMFEDID) && (indexDDU <= FEDNumbering::MAXGEMFEDID) ) // Still preLS1 DCC FEDs mapping
                {

                  int dduID    = mapping->ddu(chamberItr->first); // try to switch to DDU ID mapping
                  if ((dduID >= FEDNumbering::MINGEMDDUFEDID) && (dduID <= FEDNumbering::MAXGEMDDUFEDID) ) // DDU ID is in expectedi post-LS1 FED ID range
                    {
                      indexDDU = dduID;
                    }
                  else // Messy
                    {
                      // Lets try to change pre-LS1 1-36 ID range to post-LS1 MINGEMDDUFEDID - MAXGEMDDUFEDID range
                      dduID &= 0xFF;
                      if ((dduID <= 36) && (dduID > 0)) indexDDU = postLS1_map[dduID -1]; // indexDDU = FEDNumbering::MINGEMDDUFEDID + dduID-1;
                    }

                }

              if (indexDDU == iddu)
                {
                  std::map<int, GEMDDUEventData>::iterator dduMapItr = dduMap.find(indexDDU);
                  if (dduMapItr == dduMap.end())
                    {
                      throw cms::Exception("GEMDigiToRaw") << "Bad DDU number:" << indexDDU;
                    }
                  // get id's based on ChamberId from mapping

                  int dduInput = mapping->dduInput(chamberItr->first);
                  int dmbId    = mapping->dmb(chamberItr->first);
		  // int crateId  = mapping->crate(chamberItr->first);
                  dduMapItr->second.add( chamberItr->second, dmbId, dduInput, formatVersion_ );
                }
            }
        }

      // FIXME: FEDRawData size set to 2*64 to add FED header and trailer
      for (std::map<int, GEMDDUEventData>::iterator dduMapItr = dduMap.begin();
           dduMapItr != dduMap.end(); ++dduMapItr)
        {
          boost::dynamic_bitset<> dduBits = dduMapItr->second.pack();
          FEDRawData & fedRawData = fed_buffers.FEDData(dduMapItr->first);
          fedRawData.resize(dduBits.size()/8);
          //fill data with dduEvent
          bitset_utilities::bitsetToChar(dduBits, fedRawData.data());
          FEDTrailer gemFEDTrailer(fedRawData.data()+(fedRawData.size()-8));
          gemFEDTrailer.set(fedRawData.data()+(fedRawData.size()-8),
                            fedRawData.size()/8,
                            evf::compute_crc(fedRawData.data(),fedRawData.size()), 0, 0);

        }
    }
}


