/**\testCSCDigis.cpp
 *
 * Test suit for CSCDigi.
 * Based on testDTDigis.cpp
 *
 * $Date: 2009/03/04 03:16:41 $
 * $Revision: 1.24 $
 *
 * \author N. Terentiev, CMU (for CSCWireDigi, CSCRPCDigi, 
 *                                CSCALCTDigi, CSCCLCTDigi)
 * \author M. Schmitt,    Northwestern (for CSCComparatorDigi, CSCStripDigi)
 * \author A. Tumanov, Rice U.
 */

static const char CVSId[] = "$Id: testCSCDigis.cpp,v 1.24 2009/03/04 03:16:41 barvic Exp $";

#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
                                                                                
#include "DataFormats/CSCDigi/interface/CSCComparatorDigi.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"

#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"

#include "DataFormats/CSCDigi/interface/CSCRPCDigi.h"
#include "DataFormats/CSCDigi/interface/CSCRPCDigiCollection.h"

#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"

#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

#include "DataFormats/CSCDigi/interface/CSCCFEBStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCFEBStatusDigiCollection.h"

#include "DataFormats/CSCDigi/interface/CSCTMBStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCTMBStatusDigiCollection.h"

#include "DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigiCollection.h"

//#include "DataFormats/CSCDigi/interface/CSCEventFormatStatusDigi.h"
//#include "DataFormats/CSCDigi/interface/CSCEventFormatStatusDigiCollection.h"

#include <stdio.h>
#include <iostream>

using namespace std;

CSCIdType makeCSCId(const uint16_t crateId, const uint16_t dmbId) 
        { return ( (CSCIdType(crateId&0xFF)<<4) | (dmbId&0xF) ); }

class testCSCDigis: public CppUnit::TestFixture {

  CPPUNIT_TEST_SUITE(testCSCDigis);
  CPPUNIT_TEST(testDigiCollectionPut);
  CPPUNIT_TEST_SUITE_END();

public:

  void setUp(){}
  void tearDown(){}  

  void fillCSCWireDigi(CSCWireDigiCollection &);
  void fillCSCComparatorDigi(CSCComparatorDigiCollection &);
  void fillCSCStripDigi(CSCStripDigiCollection &);
  void fillCSCRPCDigi(CSCRPCDigiCollection &);
  void fillCSCALCTDigi(CSCALCTDigiCollection &);
  void fillCSCCLCTDigi(CSCCLCTDigiCollection &);
  void fillCSCCorrLCTDigi(CSCCorrelatedLCTDigiCollection &);
  void fillCSCCFEBStatusDigi(CSCCFEBStatusDigiCollection &);
  void fillCSCTMBStatusDigi(CSCTMBStatusDigiCollection &);
  void fillCSCDCCFormatStatusDigi(CSCDCCFormatStatusDigiCollection &);
//  void fillCSCEventFormatStatusDigi(CSCEventFormatStatusDigiCollection &);

  void readCSCWireDigi(CSCWireDigiCollection &);
  void readCSCComparatorDigi(CSCComparatorDigiCollection &);
  void readCSCStripDigi(CSCStripDigiCollection &);
  void readCSCRPCDigi(CSCRPCDigiCollection &);
  void readCSCALCTDigi(CSCALCTDigiCollection &);
  void readCSCCLCTDigi(CSCCLCTDigiCollection &);
  void readCSCCorrLCTDigi(CSCCorrelatedLCTDigiCollection &);
  void readCSCCFEBStatusDigi(CSCCFEBStatusDigiCollection &);
  void readCSCTMBStatusDigi(CSCTMBStatusDigiCollection &);
  void readCSCDCCFormatStatusDigi(CSCDCCFormatStatusDigiCollection &);
//  void readCSCEventFormatStatusDigi(CSCEventFormatStatusDigiCollection &);

  void testDigiCollectionPut();
}; 

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testCSCDigis);


void testCSCDigis::fillCSCWireDigi(CSCWireDigiCollection & collection){

  for(int endcp=1; endcp<3; endcp++)
   for(int stn=1; stn<5; stn++)
    for(int rng=1; rng<4; rng++)
     for(int csc=1; csc<37; csc++)
      for(int pln=1; pln<7; pln++) {
  
       CSCDetId detid(endcp,stn,rng,csc,pln);
                                                                                
       std::vector<CSCWireDigi> digivec;
       for (int i=10; i<11; ++i){
           int wire=i;
//           int tbin=104; // Time bins 3,5,6 - Bits 3,5,6 ON i.e. 
                                  // 64+32+8 = 1101000 in binary
           unsigned int tbinb=0, tbit=1;
           for(int j=0;j<32;j++) {
              if(j==0 || j==3 || j==5 || j==6 || j==31) tbinb=tbinb+tbit;
              tbit=tbit<<1;
                          // Time bins 0,3,5,6,31 - Bits 0,3,5,6,31 ON i.e
                          // in binary - 0b10000000000000000000000001101001
           } 
           CSCWireDigi digi(wire,tbinb);
           digivec.push_back(digi);
        }
  
        collection.put(std::make_pair(digivec.begin(),digivec.end()),detid);
  
      } // end of for(int endcp=1 ...for(int pln=1 ...) 
}

void testCSCDigis::fillCSCComparatorDigi(CSCComparatorDigiCollection & collection){

  for(int endcp=1; endcp<3; endcp++)
   for(int stn=1; stn<5; stn++)
    for(int rng=1; rng<4; rng++)
     for(int csc=1; csc<37; csc++)
      for(int pln=1; pln<7; pln++) {
 
       CSCDetId detid(endcp,stn,rng,csc,pln);
 
       std::vector<CSCComparatorDigi> digivec;
       for (int i=10; i<11; ++i){
           int aStrip = i;
           int aComparator = 2;
	   int aTbin = 6; // time bin word 0...110 (16-bits) bits 1 and 2 ON
	   CSCComparatorDigi digi(aStrip, aComparator, aTbin);
           digivec.push_back(digi);
        }
 
        collection.put(std::make_pair(digivec.begin(), digivec.end()),detid);
 
      } // end of for(int endcp=1 ...for(int pln=1 ...)
}

void testCSCDigis::fillCSCStripDigi(CSCStripDigiCollection & collection){

  for(int endcp=1; endcp<3; endcp++)
   for(int stn=1; stn<5; stn++)
    for(int rng=1; rng<4; rng++)
     for(int csc=1; csc<37; csc++)
      for(int pln=1; pln<7; pln++) {
 
       CSCDetId detid(endcp,stn,rng,csc,pln);
 
       std::vector<CSCStripDigi> digivec;
       for (int i=10; i<11; ++i){
           int aStrip = i;
           std::vector<int> someADCCounts(8);
           someADCCounts[0] = 9;
           someADCCounts[1] = 11;
           someADCCounts[2] = 23;
           someADCCounts[3] = 51;
           someADCCounts[4] = 102;
           someADCCounts[5] = 88;
           someADCCounts[6] = 48;
           someADCCounts[7] = 19;

	   std::vector<uint16_t> someVec(8);

	   CSCStripDigi digi(aStrip, someADCCounts, someVec, someVec, someVec);
           digivec.push_back(digi);
        }
 
        collection.put(std::make_pair(digivec.begin(), digivec.end()),detid);
 
      } // end of for(int endcp=1 ...for(int pln=1 ...) 
}

void testCSCDigis::fillCSCRPCDigi(CSCRPCDigiCollection & collection){
  ///rpc digis need to be tested using 
  ///CSCDetIDs and CSC relevant RAT readout A.T.

  for(int endcp=1; endcp<3; endcp++)
    for(int stn=1; stn<5; stn++)
      for(int rng=1; rng<4; rng++)
	for(int csc=1; csc<37; csc++)
	  for(int pln=3; pln<4; pln++) {  // ALCT primitives are for layer 3 only
   
	    CSCDetId detid(endcp,stn,rng,csc,pln);
   
	    std::vector<CSCRPCDigi> digivec;
	    for (int i=5; i<6; ++i){
	      int rpc=i;
	      int bxn=1;
	      int pad=1;
	      int tbin=3;
	      CSCRPCDigi digi(rpc, pad, bxn, tbin);
	      digivec.push_back(digi);
	    }
 
	    collection.put(std::make_pair(digivec.begin(),digivec.end()),detid);
	  }

}

void testCSCDigis::fillCSCALCTDigi(CSCALCTDigiCollection & collection){
 
  for(int endcp=1; endcp<3; endcp++)
   for(int stn=1; stn<5; stn++)
    for(int rng=1; rng<4; rng++)
     for(int csc=1; csc<37; csc++)
      for(int pln=3; pln<4; pln++) {  // ALCT primitives are for layer 3 only
   
       CSCDetId detid(endcp,stn,rng,csc,pln);
                                                                                
       std::vector<CSCALCTDigi> digivec;
       for (int i=1; i<3; ++i){
           int valid=1;
           int quality=3;
           int accel=0;
           int patternb=1;
           int keywire=16;
           int bx=4;
           int trknmb=1;

           CSCALCTDigi digi(valid,quality,accel,patternb,keywire,bx,trknmb);
           digivec.push_back(digi);
        }
        
        collection.put(std::make_pair(digivec.begin(),digivec.end()),detid);

      } // end of for(int endcp=1 ...for(int pln=1 ...)
}

void testCSCDigis::fillCSCCLCTDigi(CSCCLCTDigiCollection & collection){
  
  for(int endcp=1; endcp<3; endcp++)
   for(int stn=1; stn<5; stn++)
    for(int rng=1; rng<4; rng++)
     for(int csc=1; csc<37; csc++)
      for(int pln=3; pln<4; pln++) {  // CLCT primitives are for layer 3 only
    
       CSCDetId detid(endcp,stn,rng,csc,pln);
                                                                                
       std::vector<CSCCLCTDigi> digivec;
       for (int i=1; i<3; ++i){
           int valid=1;
           int quality=3;
           int patshape=2;
           int striptype=1;
           int bend=0;
           int strip=16;
           int cfeb=5;
           int bx=3;
           int trknmb=1;
 
           CSCCLCTDigi digi(valid, quality, patshape, striptype, bend, strip, cfeb, bx, trknmb);
           digivec.push_back(digi);
        }
        
collection.put(std::make_pair(digivec.begin(),digivec.end()),detid);
 
      } // end of for(int endcp=1 ...for(int pln=1 ...)
}

void testCSCDigis::fillCSCCorrLCTDigi(CSCCorrelatedLCTDigiCollection & collection)
{
  for(int endcap = 1; endcap <=2; ++endcap)
    for(int station = 1; station <=4; ++station)
      for(int ring = 1; ring <= ((station == 1) ? 3 : 2); ++ring)
	for(int chamber = 1; chamber <= ((ring == 1 && station != 1) ? 18 : 36); ++chamber)
	  for(int layer = 3; layer <=3; ++layer)
	    {
	      CSCDetId detid(endcap,station,ring,chamber,layer);
	      std::vector<CSCCorrelatedLCTDigi> digivec;

	      for(int trknmb = 1; trknmb <=2 ; ++trknmb)
                {
		  int valid = 1;
		  int quality = 15;
		  int pattern = 2;
		  int strip = 16;
		  int keywire = 10;
		  int bend = 1;
		  int bx = 0;
		  int mpclink = 1;

		  CSCCorrelatedLCTDigi digi(trknmb,valid,quality,keywire,strip,pattern,bend,bx,mpclink);
		  digivec.push_back(digi);
		}
	      collection.put(std::make_pair(digivec.begin(),digivec.end()),detid);
	    }
}

void testCSCDigis::fillCSCCFEBStatusDigi(CSCCFEBStatusDigiCollection & collection){

  for(int endcp=1; endcp<3; endcp++)
   for(int stn=1; stn<5; stn++)
    for(int rng=1; rng<4; rng++)
      for(int csc=1; csc<37; csc++) {
 
       CSCDetId detid(endcp,stn,rng,csc,0);
 
       std::vector<CSCCFEBStatusDigi> digivec;
       for (int i=1; i<6; ++i) // loop over cfeb=1-5
       {
           int aCfeb = i;
           CSCCFEBStatusDigi digi(aCfeb);
           std::vector<uint16_t> crc(8,0); crc[0]=1;crc[7]=8;
           digi.setCRC(crc);
           std::vector<uint16_t> scac(8,0); scac[0]=11;scac[7]=18;
           scac[0]=scac[0]+256+2048+4096+8192+16384+32768;
           scac[7]=scac[7]+256+2048+4096+8192+16384+32768;
           digi.setSCAC(scac);
           digivec.push_back(digi);
       }
 
        collection.put(std::make_pair(digivec.begin(), digivec.end()),detid);
 
      } // end of for(int endcp=1 ...for(int csc=1 ...) 
}

void testCSCDigis::fillCSCTMBStatusDigi(CSCTMBStatusDigiCollection & collection) {
  for(int endcp=1; endcp<3; endcp++)
   for(int stn=1; stn<5; stn++)
    for(int rng=1; rng<4; rng++)
      for(int csc=1; csc<37; csc++) {
                                                                                
       CSCDetId detid(endcp,stn,rng,csc,0);
                                                                                
       std::vector<CSCTMBStatusDigi> digivec;
       CSCTMBStatusDigi digi;
       digivec.push_back(digi);
                                                                                
       collection.put(std::make_pair(digivec.begin(), digivec.end()),detid);
                                                                                
      } // end of for(int endcp=1 ...for(int csc=1 ...)
}
/*
void testCSCDigis::fillCSCDCCFormatStatusDigi(CSCDCCFormatStatusDigiCollection & collection){
  std::vector<CSCDCCFormatStatusDigi> digivec;
  CSCDetId detid(0,0,0,0,0);

  for(int dcc=1; dcc<3; dcc++)
    for(int ddu=1; ddu<5; ddu++) {
 
           CSCDCCFormatStatusDigi digi(dcc,ddu);
           digi.setCSCDCCExaminerInfo(1);
           digi.setUnpackerInfo(2);
           digi.setErInfo(3);
           std::vector<CSCDetId> cscid;
           for(int i=31;i<33;i++) {
             CSCDetId id(1,2,3,i,0);
             cscid.push_back(id);
           } 
           digi.setCSCId(cscid);

           digivec.push_back(digi);
    }
 
  collection.put(std::make_pair(digivec.begin(), digivec.end()),detid);
}
*/
void testCSCDigis::fillCSCDCCFormatStatusDigi(CSCDCCFormatStatusDigiCollection & collection)
{
  std::vector<CSCDCCFormatStatusDigi> digivec;
  CSCDetId detid(0,0,0,0,0);
  for(int dcc=FEDNumbering::MINCSCFEDID; dcc<FEDNumbering::MAXCSCFEDID; dcc++) {
            
           CSCDCCFormatStatusDigi digi(dcc);
#ifdef DEBUG
	   digi.setDCCId(dcc);
	   digi.setDCCMask(0x111);
	   digi.setCSCMask(0x01);
	   for (int i=1; i<5; i++)
	   	digi.setDDUErrors(i, i);
	   for (int i=40; i<50;i++) 
		for (int j=1; j<10; j++) {
			digi.setCSCPayload(makeCSCId(i,j),i*j);
			digi.setCSCErrors(makeCSCId(i,j),i*j);
		}
#endif
           digivec.push_back(digi);
    }
  collection.put(std::make_pair(digivec.begin(), digivec.end()),detid);
} 

void testCSCDigis::readCSCCorrLCTDigi(CSCCorrelatedLCTDigiCollection & collection)
{
  int count = 0;
  CSCCorrelatedLCTDigiCollection::DigiRangeIterator detUnitIt;

  for(detUnitIt = collection.begin() ; detUnitIt != collection.end();
      detUnitIt++)
    {
      const CSCDetId& id = (*detUnitIt).first;

      const CSCCorrelatedLCTDigiCollection::Range& range = (*detUnitIt).second;

      for(CSCCorrelatedLCTDigiCollection::const_iterator digiIt = range.first;
	  digiIt != range.second; digiIt++)
	{
	  ++count;
	  CPPUNIT_ASSERT(digiIt->isValid() == 1);
	  CPPUNIT_ASSERT(digiIt->getQuality() == 15);
	  CPPUNIT_ASSERT(digiIt->getCLCTPattern() == 2);
	  CPPUNIT_ASSERT(digiIt->getStrip() == 16);
	  CPPUNIT_ASSERT(digiIt->getKeyWG() == 10);
	  CPPUNIT_ASSERT(digiIt->getBend() == 1);
	  CPPUNIT_ASSERT(digiIt->getBX() == 0);
	  
	  std::cout << "CSC Correlated LCT - endcap station ring csc layer LCT# Quality: " << id.endcap() << " " << id.station()
		    << " " << id.ring() << " " << id.chamber() << " " << id.layer() << " " << digiIt->getTrknmb() << " " 
		    << digiIt->getQuality() << std::endl;
	}
    }
  std::cout << "CSC Correlated LCT Digi count - " << count << std::endl;
}

void testCSCDigis::readCSCWireDigi(CSCWireDigiCollection & collection){

        int count=0;
        CSCWireDigiCollection::DigiRangeIterator detUnitIt;
        for (detUnitIt=collection.begin(); detUnitIt!=collection.end(); ++detUnitIt) {
  
          const CSCDetId& id = (*detUnitIt).first;
          const CSCWireDigiCollection::Range& range =(*detUnitIt).second;
 
          for (CSCWireDigiCollection::const_iterator digiIt=range.first; digiIt!=range.second; ++digiIt) {
  
            count++;
            CPPUNIT_ASSERT((*digiIt).getWireGroup()==10);
            CPPUNIT_ASSERT((*digiIt).getTimeBin()==0);
            printf("CSC Wire - endcap station ring csc layer wire tbin: %3d %3d %3d %3d %3d %3d  %3d\n",
               id.endcap(),id.station(),id.ring(),id.chamber(),id.layer(),
               (*digiIt).getWireGroup(),(*digiIt).getTimeBin());
            std::cout << " CSC Wire Time Bins On ";
            std::vector<int> tbins=(*digiIt).getTimeBinsOn();
            for(unsigned int i=0; i<tbins.size();++i) std::cout<<tbins[i]<<" ";
            std::cout<<std::endl; 
            std::cout << " CSC Wire Word with Time Bins Bits On "
                      <<(*digiIt).getTimeBinWord()<<std::endl;
          }// for digis in layer
        }// end of for (detUnitIt=...

    printf("CSC Wire count:  %3d \n", count);
} 

void testCSCDigis::readCSCComparatorDigi(CSCComparatorDigiCollection & collection){

  int count=0;
  CSCComparatorDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt=collection.begin(); detUnitIt!=collection.end(); ++detUnitIt) {
 
    const CSCDetId& id = (*detUnitIt).first;
    const CSCComparatorDigiCollection::Range& range = (*detUnitIt).second;

    for (CSCComparatorDigiCollection::const_iterator digiIt=range.first; digiIt!=range.second; ++digiIt) {
      count++;
      CPPUNIT_ASSERT((*digiIt).getComparator()==2);
      CPPUNIT_ASSERT((*digiIt).getStrip()==10);
      CPPUNIT_ASSERT((*digiIt).getTimeBin()==1); // time bin word=6 means bit 1 should be first!
      printf("CSCComparatorDigi - endcap station ring csc layer strip comparator time: %3d %3d %3d %3d %3d %3d %3d %3d\n",
         id.endcap(),id.station(),id.ring(),id.chamber(),id.layer(),
	     (*digiIt).getStrip(), (*digiIt).getComparator(), (*digiIt).getTimeBin());
      std::cout << " CSCComparatorDigi - time bins ON: ";
      std::vector<int> tbins=(*digiIt).getTimeBinsOn();
      for(unsigned int i=0; i<tbins.size();++i) std::cout<<tbins[i]<<" ";
      std::cout<<std::endl; 

    }// for digis in layer
  }// end of for (detUnitIt=...
   
  printf("CSCComparatorDigi count:  %3d \n", count);
 
}

void testCSCDigis::readCSCStripDigi(CSCStripDigiCollection & collection){

  int count=0;

  CSCStripDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt=collection.begin();
       detUnitIt!=collection.end();
       ++detUnitIt){
 
    const CSCDetId& id = (*detUnitIt).first;
 
    const CSCStripDigiCollection::Range& range = (*detUnitIt).second;
    for (CSCStripDigiCollection::const_iterator digiIt =
           range.first; digiIt!=range.second; ++digiIt){
      count++;
      CPPUNIT_ASSERT((*digiIt).getStrip()==10);
      printf("CSC Strip - endcap station ring csc layer: %3d %3d %3d %3d %3d strip: %3d  ADC: %4d %4d %4d %4d %4d %4d %4d %4d\n",
             id.endcap(),id.station(),id.ring(),id.chamber(),id.layer(),
             (*digiIt).getStrip(),
             (*digiIt).getADCCounts()[0],
             (*digiIt).getADCCounts()[1],
             (*digiIt).getADCCounts()[2],
             (*digiIt).getADCCounts()[3],
             (*digiIt).getADCCounts()[4],
             (*digiIt).getADCCounts()[5],
             (*digiIt).getADCCounts()[6],
             (*digiIt).getADCCounts()[7]);
 
    }// for digis in layer
  }// end of for (detUnitIt=...
   
  printf("CSC Strip count:  %3d \n", count);
}

void testCSCDigis::readCSCRPCDigi(CSCRPCDigiCollection & collection){

        int count=0;
        CSCRPCDigiCollection::DigiRangeIterator detUnitIt;
        for (detUnitIt=collection.begin();
             detUnitIt!=collection.end();
           ++detUnitIt){
 
           const CSCDetId& id = (*detUnitIt).first;
 
           const CSCRPCDigiCollection::Range& range = (*detUnitIt).second;
           for (CSCRPCDigiCollection::const_iterator digiIt =
                range.first; digiIt!=range.second;
              ++digiIt){
 
              count++;
              CPPUNIT_ASSERT((*digiIt).getRpc()==5);
              CPPUNIT_ASSERT((*digiIt).getBXN()==1);

	      printf("RPC digi - endcap station ring csc layer: %3d %3d %3d %3d %3d %3d  %4d \n",id.endcap(),id.station(),id.ring(),id.chamber(),id.layer(),(*digiIt).getRpc(),(*digiIt).getBXN());
   
    }// for digis in layer
   }// end of for (detUnitIt=...
 
    printf("RPC count:  %3d \n", count); 
}

void testCSCDigis::readCSCALCTDigi(CSCALCTDigiCollection & collection){
 
        int count=0;
        CSCALCTDigiCollection::DigiRangeIterator detUnitIt;
        for (detUnitIt=collection.begin();
             detUnitIt!=collection.end();
           ++detUnitIt){
   
           const CSCDetId& id = (*detUnitIt).first;
                                                                                
           const CSCALCTDigiCollection::Range& range =(*detUnitIt).second;
           for (CSCALCTDigiCollection::const_iterator digiIt =
                range.first; digiIt!=range.second;
              ++digiIt){
   
              count++;
              CPPUNIT_ASSERT((*digiIt).isValid());
              CPPUNIT_ASSERT((*digiIt).getQuality()==3);
              CPPUNIT_ASSERT((*digiIt).getAccelerator()==0);
              CPPUNIT_ASSERT((*digiIt).getCollisionB()==1);
              CPPUNIT_ASSERT((*digiIt).getKeyWG()==16);
              CPPUNIT_ASSERT((*digiIt).getBX()==4);
              CPPUNIT_ASSERT((*digiIt).getTrknmb()==1);

 printf("CSC ALCT - endcap station ring csc layer valid quality accel pattern wire bx track: %3d %3d %3d %3d %3d %3d  %3d %3d %3d %3d %3d %3d\n",id.endcap(),id.station(),id.ring(),id.chamber(),id.layer(),(*digiIt).isValid(),(*digiIt).getQuality(),(*digiIt).getAccelerator(),(*digiIt).getCollisionB(),(*digiIt).getKeyWG(),(*digiIt).getBX(),(*digiIt).getTrknmb());
    }// for digis in layer
   }// end of for (detUnitIt=...
    printf("CSC ALCT count:  %3d \n", count);
}

void testCSCDigis::readCSCCLCTDigi(CSCCLCTDigiCollection & collection){
  
        int count=0;
        CSCCLCTDigiCollection::DigiRangeIterator detUnitIt;
        for (detUnitIt=collection.begin();
             detUnitIt!=collection.end();
           ++detUnitIt){
    
           const CSCDetId& id = (*detUnitIt).first;
                                                                                
           const CSCCLCTDigiCollection::Range& range =(*detUnitIt).second;
           for (CSCCLCTDigiCollection::const_iterator digiIt =
                range.first; digiIt!=range.second;
              ++digiIt){
    
              count++;
              CPPUNIT_ASSERT((*digiIt).isValid());
              CPPUNIT_ASSERT((*digiIt).getQuality()==3);
              CPPUNIT_ASSERT((*digiIt).getPattern()==2);
              CPPUNIT_ASSERT((*digiIt).getStripType()==1);
              CPPUNIT_ASSERT((*digiIt).getBend()==0);
              CPPUNIT_ASSERT((*digiIt).getStrip()==16);
              CPPUNIT_ASSERT((*digiIt).getCFEB()==5);              
              CPPUNIT_ASSERT((*digiIt).getBX()==3);
              CPPUNIT_ASSERT((*digiIt).getTrknmb()==1);

 printf("CSC CLCT - endcap station ring csc layer valid quality pattern striptype bend strip cfeb bx tracknmb %3d %3d %3d %3d %3d %3d  %3d %3d %3d %3d %3d %3d %3d %3d\n",id.endcap(),id.station(),id.ring(),id.chamber(),id.layer(),(*digiIt).isValid(),(*digiIt).getQuality(),(*digiIt).getPattern(),(*digiIt).getStripType(),(*digiIt).getBend(),(*digiIt).getStrip(),(*digiIt).getCFEB(),(*digiIt).getBX(), (*digiIt).getTrknmb());
    
    }// for digis in layer
   }// end of for (detUnitIt=...
    printf("CSC CLCT count:  %3d \n", count);
}

void testCSCDigis::readCSCCFEBStatusDigi(CSCCFEBStatusDigiCollection & collection) {

  CSCCFEBStatusDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt=collection.begin();
       detUnitIt!=collection.end();
       ++detUnitIt){
 
    const CSCDetId& id = (*detUnitIt).first;
 
    const CSCCFEBStatusDigiCollection::Range& range = (*detUnitIt).second;

    int cfebcount=0;
    for (CSCCFEBStatusDigiCollection::const_iterator digiIt =
           range.first; digiIt!=range.second; ++digiIt){
      cfebcount++;
      CPPUNIT_ASSERT((*digiIt).getCFEBNmb()==cfebcount);
      printf("CSC CFEBStatus - endcap station ring csc cfeb: %3d %3d %3d %3d %3d \n",id.endcap(),id.station(),id.ring(),id.chamber(),(*digiIt).getCFEBNmb());
/*  Commented because the latest changes in CFEBStatusdDigi are not reflected
      std::cout<<"CSC CFEBStatus - SCA Full Condition:";
      for(int i=0;i<4;i++) std::cout<<" "<<(*digiIt).getSCAFullCond()[i];
      std::cout<<std::endl;
      std::cout<<"CSC CFEBStatus - CRC:";
      for(int i=0;i<8;i++) std::cout<<" "<<(*digiIt).getCRC()[i];
      std::cout<<std::endl;
      std::cout<<"CSC CFEBStatus - TS_FLAG:";
      for(int i=0;i<8;i++) std::cout<<" "<<(*digiIt).getTS_FLAG()[i];
      std::cout<<std::endl;
      std::cout<<"CSC CFEBStatus - SCA_FULL:";
      for(int i=0;i<8;i++) std::cout<<" "<<(*digiIt).getSCA_FULL()[i];
      std::cout<<std::endl;
      std::cout<<"CSC CFEBStatus - LCT_PHASE:";
      for(int i=0;i<8;i++) std::cout<<" "<<(*digiIt).getLCT_PHASE()[i];
      std::cout<<std::endl;
      std::cout<<"CSC CFEBStatus - L1A_PHASE:";
      for(int i=0;i<8;i++) std::cout<<" "<<(*digiIt).getL1A_PHASE()[i];
      std::cout<<std::endl;
      std::cout<<"CSC CFEBStatus - SCA_BLK:";
      for(int i=0;i<8;i++) std::cout<<" "<<(*digiIt).getSCA_BLK()[i];
      std::cout<<std::endl;
      std::cout<<"CSC CFEBStatus - TRIGGER_TIME:";
      for(int i=0;i<8;i++) std::cout<<" "<<(*digiIt).getTRIG_TIME()[i];
      std::cout<<std::endl;
*/
    }// for digis in collection
  }// end of for (detUnitIt=...
   
}

void testCSCDigis::readCSCTMBStatusDigi(CSCTMBStatusDigiCollection & collection) {

  CSCTMBStatusDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt=collection.begin();
       detUnitIt!=collection.end();
       ++detUnitIt){

    const CSCTMBStatusDigiCollection::Range& range = (*detUnitIt).second;

    for (CSCTMBStatusDigiCollection::const_iterator digiIt =
           range.first; digiIt!=range.second; ++digiIt){

      /*      CPPUNIT_ASSERT((*digiIt).getDAV()==7);
      //      CPPUNIT_ASSERT((*digiIt).getDAV()==3);
      CPPUNIT_ASSERT((*digiIt).getHALF()==0);
      CPPUNIT_ASSERT((*digiIt).getEMPTY()==1);
      CPPUNIT_ASSERT((*digiIt).getStart_Timeout()==0);
      CPPUNIT_ASSERT((*digiIt).getEnd_Timeout()==1);
      CPPUNIT_ASSERT((*digiIt).getFULL()==0);
      CPPUNIT_ASSERT((*digiIt).getBoardId()==5);
      CPPUNIT_ASSERT((*digiIt).getCscId()==10);
      CPPUNIT_ASSERT((*digiIt).getBXNCntL1A()==512);
      CPPUNIT_ASSERT((*digiIt).getBXNCntPretr()==256);
      CPPUNIT_ASSERT((*digiIt).getNmbTbinsPretr()==8);

      printf("CSC TMBStatus - endcap station ring csc TMB_DAV: %3d %3d %3d %3d%3d \n",id.endcap(),id.station(),id.ring(),id.chamber(),(*digiIt).getDAV());
      */

    }// for digis in collection
  }// end of for (detUnitIt=...

}
/*
void testCSCDigis::readCSCDCCFormatStatusDigi(CSCDCCFormatStatusDigiCollection & collection) {
  CSCDCCFormatStatusDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt=collection.begin();
       detUnitIt!=collection.end();
       ++detUnitIt){
 
    const CSCDetId& id = (*detUnitIt).first;
    std::cout<<"CSCDCCFormatStatus id "<<id<<std::endl;

    const CSCDCCFormatStatusDigiCollection::Range& range = (*detUnitIt).second;

    int count=0;
    for (CSCDCCFormatStatusDigiCollection::const_iterator digiIt = range.first; digiIt!=range.second; ++digiIt){
      
      CPPUNIT_ASSERT((*digiIt).getDCCNmb()==(count/4+1));
      count++;
      std::cout<<"CSCDCCFormatStatusDigi - dcc ddu exam unpack er "
               <<(*digiIt).getDCCNmb()<<" "<<(*digiIt).getDDUNmb()<<" "
               <<(*digiIt).getCSCDCCExaminerInfo()<<" "
               <<(*digiIt).getUnpackerInfo()<<" "
               <<(*digiIt).getErInfo()<<std::endl;

      std::cout<<"CSCDCCFormatStatusDigi - CSCId:";
      for(unsigned int i=0;i<(*digiIt).getCSCId().size();i++) 
      std::cout<<" "<<(*digiIt).getCSCId()[i];
      std::cout<<std::endl;

    }// for digis in collection
  }// end of for (detUnitIt=...
}
*/
void testCSCDigis::readCSCDCCFormatStatusDigi(CSCDCCFormatStatusDigiCollection & collection){
  CSCDCCFormatStatusDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt=collection.begin();
       detUnitIt!=collection.end();
       ++detUnitIt){
       
//   const CSCDetId& id = (*detUnitIt).first;

    const CSCDCCFormatStatusDigiCollection::Range& range = (*detUnitIt).second;

    for (CSCDCCFormatStatusDigiCollection::const_iterator digiIt = range.first; digiIt!=range.second; ++digiIt){
	digiIt->print();


    }// for digis in collection 

  }// end of for (detUnitIt=...
}

void testCSCDigis::testDigiCollectionPut(){

/************           Filling collections             *****************/

       CSCWireDigiCollection wiredigiCollection;
       fillCSCWireDigi(wiredigiCollection);

       CSCComparatorDigiCollection comparatordigiCollection;
       fillCSCComparatorDigi(comparatordigiCollection);

       CSCStripDigiCollection stripdigiCollection;
       fillCSCStripDigi(stripdigiCollection);


       CSCRPCDigiCollection rpcdigiCollection;
       fillCSCRPCDigi(rpcdigiCollection);

       CSCALCTDigiCollection alctdigiCollection;
       fillCSCALCTDigi(alctdigiCollection);

       CSCCLCTDigiCollection clctdigiCollection;
       fillCSCCLCTDigi(clctdigiCollection);

       CSCCorrelatedLCTDigiCollection corrlctdigiCollection;
       fillCSCCorrLCTDigi(corrlctdigiCollection);

       CSCCFEBStatusDigiCollection cfebstatusdigiCollection;
       fillCSCCFEBStatusDigi(cfebstatusdigiCollection);

       CSCTMBStatusDigiCollection tmbstatusdigiCollection;
       fillCSCTMBStatusDigi(tmbstatusdigiCollection);

       CSCDCCFormatStatusDigiCollection dccformatstatusdigiCollection;
       fillCSCDCCFormatStatusDigi(dccformatstatusdigiCollection);

//       CSCEventFormatStatusDigiCollection eventformatstatusdigiCollection;
//       fillCSCEventFormatStatusDigi(eventformatstatusdigiCollection);
      
/************           Reading collections             *****************/

       readCSCWireDigi(wiredigiCollection);
       readCSCComparatorDigi(comparatordigiCollection);
       readCSCStripDigi(stripdigiCollection);
       readCSCRPCDigi(rpcdigiCollection);
       readCSCALCTDigi(alctdigiCollection);
       readCSCCLCTDigi(clctdigiCollection);
       readCSCCorrLCTDigi(corrlctdigiCollection);
       readCSCCFEBStatusDigi(cfebstatusdigiCollection);
       readCSCTMBStatusDigi(tmbstatusdigiCollection);
       readCSCDCCFormatStatusDigi(dccformatstatusdigiCollection);
//       readCSCEventFormatStatusDigi(eventformatstatusdigiCollection);
}
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
