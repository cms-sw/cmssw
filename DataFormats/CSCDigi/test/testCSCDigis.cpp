/**\testCSCDigis.cpp
 *
 * Test suit for CSCDigi.
 * Based on testDTDigis.cpp
 *
 * $Date:$
 * $Revision:$
 *
 * \author N. Terentiev, CMU (for CSCWireDigi, CSCRPCDigi, 
 *                                CSCALCTDigi, CSCCLCTDigi)
 * \author M. Schmitt,    Northwestern (for CSCComparatorDigi, CSCStripDigi)
 * \author A. Tumanov, Rice U.
 */

static const char CVSId[] = "$Id:$";

#include <cppunit/extensions/HelperMacros.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <DataFormats/CSCDigi/interface/CSCWireDigi.h>
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>
                                                                                
#include <DataFormats/CSCDigi/interface/CSCComparatorDigi.h>
#include <DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h>

#include <DataFormats/CSCDigi/interface/CSCStripDigi.h>
#include <DataFormats/CSCDigi/interface/CSCStripDigiCollection.h>

#include <DataFormats/CSCDigi/interface/CSCRPCDigi.h>
#include <DataFormats/CSCDigi/interface/CSCRPCDigiCollection.h>

#include <DataFormats/CSCDigi/interface/CSCALCTDigi.h>
#include <DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h>

#include <DataFormats/CSCDigi/interface/CSCCLCTDigi.h>
#include <DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h>

#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h>
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>

#include <stdio.h>

class testCSCDigis: public CppUnit::TestFixture {

  CPPUNIT_TEST_SUITE(testCSCDigis);
  CPPUNIT_TEST(testDigiPacking);
  CPPUNIT_TEST(testDigiCollectionPut);
  CPPUNIT_TEST_SUITE_END();

public:

  void setUp(){}
  void tearDown(){}  
  void testDigiPacking();

  void fillCSCWireDigi(CSCWireDigiCollection &);
  void fillCSCComparatorDigi(CSCComparatorDigiCollection &);
  void fillCSCStripDigi(CSCStripDigiCollection &);
  void fillCSCRPCDigi(CSCRPCDigiCollection &);
  void fillCSCALCTDigi(CSCALCTDigiCollection &);
  void fillCSCCLCTDigi(CSCCLCTDigiCollection &);
  void fillCSCCorrLCTDigi(CSCCorrelatedLCTDigiCollection &);

  void readCSCWireDigi(CSCWireDigiCollection &);
  void readCSCComparatorDigi(CSCComparatorDigiCollection &);
  void readCSCStripDigi(CSCStripDigiCollection &);
  void readCSCRPCDigi(CSCRPCDigiCollection &);
  void readCSCALCTDigi(CSCALCTDigiCollection &);
  void readCSCCLCTDigi(CSCCLCTDigiCollection &);
  void readCSCCorrLCTDigi(CSCCorrelatedLCTDigiCollection &);

  void testDigiCollectionPut();
}; 

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testCSCDigis);

void testCSCDigis::testDigiPacking(){

  CPPUNIT_ASSERT(sizeof(CSCWireDigi::PersistentPacking)==sizeof(CSCWireDigi::PackedDigiType));
  CPPUNIT_ASSERT(sizeof(CSCWireDigi::ChannelPacking)==sizeof(int));

//  CPPUNIT_ASSERT(sizeof(CSCComparatorDigi::PersistentPacking)==sizeof(CSCComparatorDigi::PackedDigiType));
//  CPPUNIT_ASSERT(sizeof(CSCComparatorDigi::ChannelPacking)==sizeof(int));

//  CPPUNIT_ASSERT(sizeof(CSCStripDigi::PersistentPacking)==sizeof(CSCStripDigi::PackedDigiType));
//  CPPUNIT_ASSERT(sizeof(CSCStripDigi::ChannelPacking)==sizeof(int));

  CPPUNIT_ASSERT (sizeof(CSCRPCDigi::PersistentPacking)==
		  sizeof(CSCRPCDigi::PackedDigiType));

  CPPUNIT_ASSERT (sizeof(CSCALCTDigi::PersistentPacking)==sizeof(CSCALCTDigi::PackedDigiType));
  CPPUNIT_ASSERT(sizeof(CSCALCTDigi::ChannelPacking)==sizeof(int));

  CPPUNIT_ASSERT (sizeof(CSCCLCTDigi::PersistentPacking)==sizeof(CSCCLCTDigi::PackedDigiType));
  CPPUNIT_ASSERT(sizeof(CSCCLCTDigi::ChannelPacking)==sizeof(int));
}

void testCSCDigis::fillCSCWireDigi(CSCWireDigiCollection & collection){

  for(int endcp=1; endcp<3; endcp++)
   for(int stn=1; stn<5; stn++)
    for(int rng=1; rng<4; rng++)
     for(int csc=1; csc<37; csc++)
      for(int pln=1; pln<7; pln++) {
  
       CSCDetId detid(endcp,stn,rng,csc,pln);
                                                                                
       std::vector<CSCWireDigi> digivec;
       for (int i=10; i<11; ++i){
           CSCWireDigi::PackedDigiType pd;
           pd.wire=i;
           pd.tbin=5;
  
           CSCWireDigi digi(pd);
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
           CSCComparatorDigi::theComparatorDigi cd;
           cd.strip = aStrip;
           cd.comparator = aComparator;
           CSCComparatorDigi digi(cd);
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
           CSCStripDigi::theStripDigi sd;
           sd.strip = aStrip;
           sd.ADCCounts = someADCCounts;
           CSCStripDigi digi(sd);
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
	      CSCRPCDigi::PackedDigiType pd;
	      pd.rpc=i;
	      pd.bxn=1;
	      pd.pad=1;
	      pd.tbin=3;
 
	      CSCRPCDigi digi(pd);
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
           CSCALCTDigi::PackedDigiType pd;

           pd.valid=1;
           pd.quality=3;
           pd.accel=0;
           pd.patternb=1;
           pd.keywire=16;
           pd.bx=4;
           pd.trknmb=1;

           CSCALCTDigi digi(pd);
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
           CSCCLCTDigi::PackedDigiType pd;

           pd.valid=1;
           pd.quality=3;
           pd.patshape=2;
           pd.striptype=1;
           pd.bend=0;
           pd.strip=16;
           pd.cfeb=5;
           pd.bx=3;
           pd.trknmb=1;
 
           CSCCLCTDigi digi(pd);
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
		  CSCCorrelatedLCTDigi::PackedDigiType pd;
		  
		  pd.trknmb = trknmb;
		  pd.valid = 1;
		  pd.quality = 15;
		  pd.pattern = 2;
		  pd.strip = 16;
		  pd.keywire = 10;
		  pd.bend = 1;
		  pd.bx = 0;

		  CSCCorrelatedLCTDigi digi(pd);
		  digivec.push_back(digi);
		}
	      collection.put(std::make_pair(digivec.begin(),digivec.end()),detid);
	    }
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
        for (detUnitIt=collection.begin();
             detUnitIt!=collection.end();
           ++detUnitIt){
  
           const CSCDetId& id = (*detUnitIt).first;
  
           const CSCWireDigiCollection::Range& range =(*detUnitIt).second;
           for (CSCWireDigiCollection::const_iterator digiIt =
                range.first; digiIt!=range.second;
              ++digiIt){
  
              count++;
              CPPUNIT_ASSERT((*digiIt).getWireGroup()==10);
              CPPUNIT_ASSERT((*digiIt).getBeamCrossingTag()==5);
 printf("CSC Wire - endcap station ring csc plane wire tbin: %3d %3d %3d %3d %3d %3d  %3d\n",id.endcap(),id.station(),id.ring(),id.chamber(),id.layer(),(*digiIt).getWireGroup(),(*digiIt).getBeamCrossingTag());
  
    }// for digis in layer
   }// end of for (detUnitIt=...
    printf("CSC Wire count:  %3d \n", count);
} 

void testCSCDigis::readCSCComparatorDigi(CSCComparatorDigiCollection & collection){

  int count=0;
  CSCComparatorDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt=collection.begin();
       detUnitIt!=collection.end();
       ++detUnitIt){
 
    const CSCDetId& id = (*detUnitIt).first;

    const CSCComparatorDigiCollection::Range& range = (*detUnitIt).second;
    for (CSCComparatorDigiCollection::const_iterator digiIt =
           range.first; digiIt!=range.second; ++digiIt){
      count++;
      CPPUNIT_ASSERT((*digiIt).getComparator()==2);
      CPPUNIT_ASSERT((*digiIt).getStrip()==10);
      printf("CSC Comparator - endcap station ring csc plane strip comparator: %3d %3d %3d %3d %3d %3d  %3d\n",id.endcap(),id.station(),id.ring(),id.chamber(),id.layer(),(*digiIt).getStrip(),(*digiIt).getComparator());
 
    }// for digis in layer
  }// end of for (detUnitIt=...
   
  printf("CSC Comparator count:  %3d \n", count);
 
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
      printf("CSC Strip - endcap station ring csc plane: %3d %3d %3d %3d %3d strip: %3d  ADC: %4d %4d %4d %4d %4d %4d %4d %4d\n",
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

	      printf("RPC digi - endcap station ring csc plane: %3d %3d %3d %3d %3d %3d  %4d \n",id.endcap(),id.station(),id.ring(),id.chamber(),id.layer(),(*digiIt).getRpc(),(*digiIt).getBXN());
   
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

 printf("CSC ALCT - endcap station ring csc plane valid quality accel pattern wire bx track: %3d %3d %3d %3d %3d %3d  %3d %3d %3d %3d %3d %3d\n",id.endcap(),id.station(),id.ring(),id.chamber(),id.layer(),(*digiIt).isValid(),(*digiIt).getQuality(),(*digiIt).getAccelerator(),(*digiIt).getCollisionB(),(*digiIt).getKeyWG(),(*digiIt).getBX(),(*digiIt).getTrknmb());
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

 printf("CSC CLCT - endcap station ring csc plane valid quality pattern striptype bend strip cfeb bx tracknmb %3d %3d %3d %3d %3d %3d  %3d %3d %3d %3d %3d %3d %3d %3d\n",id.endcap(),id.station(),id.ring(),id.chamber(),id.layer(),(*digiIt).isValid(),(*digiIt).getQuality(),(*digiIt).getPattern(),(*digiIt).getStripType(),(*digiIt).getBend(),(*digiIt).getStrip(),(*digiIt).getCFEB(),(*digiIt).getBX(), (*digiIt).getTrknmb());
    
    }// for digis in layer
   }// end of for (detUnitIt=...
    printf("CSC CLCT count:  %3d \n", count);
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

/************           Reading collections             *****************/

       readCSCWireDigi(wiredigiCollection);
       readCSCComparatorDigi(comparatordigiCollection);
       readCSCStripDigi(stripdigiCollection);
       readCSCRPCDigi(rpcdigiCollection);
       readCSCALCTDigi(alctdigiCollection);
       readCSCCLCTDigi(clctdigiCollection);
       readCSCCorrLCTDigi(corrlctdigiCollection);
}
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
