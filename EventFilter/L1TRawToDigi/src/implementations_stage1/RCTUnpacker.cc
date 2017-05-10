#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "FWCore/Framework/interface/one/EDProducerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

// RCT data formats

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "CaloCollections.h"

#include "RCTInfo.hh"

#define EVENT_HEADER_WORDS 6
#define CHANNEL_HEADER_WORDS 2
#define CHANNEL_DATA_WORDS_PER_BX 6
#define NLinks 36
#define NIntsPerLink 6

namespace l1t {
  namespace stage1 {
    class RCTUnpacker : public Unpacker {
    public:
      //RCTUnpacker(const edm::ParameterSet&, edm::Event&);
      virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
    private:
      virtual bool produceRCTInfo(const std::vector <unsigned int> evenFiberData,
				  const std::vector <unsigned int> oddFiberData,
				  const uint32_t crateID,
				  std::vector <RCTInfo> &rctInfoData);
      virtual bool getBXNumbers(uint32_t L1aBCID, uint32_t BXsInCapture, unsigned int BCs[5], uint32_t firstBX, uint32_t lastBX);
      virtual bool decodeCapturedLinkID(const uint32_t capturedValue, unsigned int &crateNumber, unsigned int &linkNumber, bool &even);
      
    };

    /*    
    class RCTUnpackerFactory : public BaseUnpackerFactory {
    public:
      RCTUnpackerFactory(const edm::ParameterSet&, edm::one::EDProducerBase&);
      virtual std::vector<UnpackerItem> create(edm::Event&, const unsigned& fw, const int fedid);
      
    private:
      const edm::ParameterSet& cfg_;
      edm::one::EDProducerBase& prod_;
    };
    */
  }
}

// Implementation

namespace l1t {
  namespace stage1 {
    /*    RCTUnpacker::RCTUnpacker(const edm::ParameterSet& cfg, edm::Event& ev) :
      ev_(ev),
      res1_(new CaloEmCandBxCollection()),
      res2_(new CaloRegionBxCollection())
    {
    };
    
    RCTUnpacker::~RCTUnpacker()
    {
      ev_.put(res1_);
      ev_.put(res2_);
    };
*/
    bool RCTUnpacker::unpack(const Block& block, UnpackerCollections *coll){
      auto resRCTEmCands_ = static_cast<CaloCollections*>(coll)->getCaloEmCands();
      auto resRCTRegions_ = static_cast<CaloCollections*>(coll)->getCaloRegions();
      //bool RCTUnpacker::unpack(const unsigned char *data, const unsigned block_id, const unsigned size){
      //LogDebug("L1T") << "Block ID  = " << block_id << " size = " << size;
      
      uint32_t nBX = 0; 
      uint32_t ctp7FWVersion;
      //uint32_t CRCErrCnt;
      //uint32_t linkStatus;
      uint32_t L1ID, L1aBCID;
      uint32_t BCs[5] = {0}; //5 BCs is max readout
      
      L1ID          =  block.payload()[1];                      // extract the L1 ID number
      L1aBCID       =  block.payload()[5] & 0x00000FFF;         // extract the BCID number of L1A
      nBX           = (block.payload()[5] & 0x00FF0000) >> 16;  // extract number of BXs readout per L1A 
      ctp7FWVersion =  block.payload()[4];
      
      LogDebug("L1T") << "L1ID = " << L1ID << " L1A BCID = " << L1aBCID << " BXs in capture = " << nBX << " CTP7 DAQ FW = " << ctp7FWVersion;
      
      //getBXNumbers handles special cases, for example:
      //if L1A is 3563, nBX = 3 then BCs = 3562, 3563, 0
      uint32_t firstBX = 0; uint32_t lastBX = 1; 
      getBXNumbers(L1aBCID, nBX, BCs, firstBX, lastBX);
	
      //resRCTEmCands_->setBXRange(0, nBX);
      //resRCTRegions_->setBXRange(0, nBX);
      
      std::vector<RCTInfo> allCrateRCTInfo[nBX];
      
      struct link_data{
	unsigned int ctp7LinkNumber;
	//full 32-bit linkID
	  unsigned int capturedLinkID;
	//decoded linkID from the oRSC
	unsigned int capturedLinkNumber;
	bool even;
	unsigned int crateID;
	std::vector <unsigned int> uint;
      };
      
      link_data allLinks[nBX][NLinks];
      
      uint32_t iDAQBuffer = 0;
      
      //Step 1: Grab all data from ctp7 buffer and put into link format
      for(unsigned int iLink = 0; iLink < NLinks; iLink++ ){
	
	iDAQBuffer = EVENT_HEADER_WORDS + iLink * (CHANNEL_HEADER_WORDS + 
						   nBX * CHANNEL_DATA_WORDS_PER_BX);
	
	//first decode linkID 
	uint32_t linkID     = block.payload()[iDAQBuffer++];
	uint32_t tmp        = block.payload()[iDAQBuffer++];
	//uint32_t CRCErrCnt  =  tmp & 0x0000FFFF;
	//uint32_t linkStatus = (tmp & 0xFFFF0000) >> 16;
	
	uint32_t capturedLinkID = 0;
	uint32_t crateID = 0;
	uint32_t capturedLinkNumber = 0;
	bool even = false;
	
	//Using normal method for decoding oRSC Captured LinkID
       //Input linkID, output: crateID, capturedLinkNumber, Even or Odd
	decodeCapturedLinkID(linkID, crateID, capturedLinkNumber, even);
	
	// Loop over multiple BX                                                                        
	for (uint32_t iBX=0; iBX<nBX; iBX++){
	  allLinks[iLink][iBX].uint.reserve(6);
	  allLinks[iLink][iBX].ctp7LinkNumber     = iLink;
	  allLinks[iLink][iBX].capturedLinkID     = capturedLinkID;
	  allLinks[iLink][iBX].crateID            = crateID;
	  allLinks[iLink][iBX].capturedLinkNumber = capturedLinkNumber;
	  allLinks[iLink][iBX].even               = even;
	  
	  for(unsigned int iWord = 0; iWord < 6 ; iWord++ ){
	    allLinks[iLink][iBX].uint.push_back(block.payload()[iDAQBuffer+iWord]);
	  }
	}
      }
      
     //Step 2: Dynamically match links and create RCTInfo Objects 
      uint32_t nCratesFound = 0;
      for(unsigned int iCrate = 0; iCrate < 18 ; iCrate++){
       
	bool foundEven = false, foundOdd = false;
	link_data even[nBX];
	link_data odd[nBX];
	
	for(unsigned int iLink = 0; iLink < NLinks; iLink++){
	  
	  if( (allLinks[iLink][0].crateID==iCrate) && (allLinks[iLink][0].even == true) ){
	   
	    foundEven = true;
	    for (unsigned int iBX=0; iBX<nBX; iBX++)
	      even[iBX] = allLinks[iLink][iBX];
	    
	 }
	  else if( (allLinks[iLink][0].crateID==iCrate) && (allLinks[iLink][0].even == false) ){
	    
	    foundOdd = true;
	    for (unsigned int iBX=0; iBX<nBX; iBX++)
	      odd[iBX] = allLinks[iLink][iBX];
	    
	  } 
	  //if success then create RCTInfoVector and fill output object
	  if(foundEven && foundOdd){
	    nCratesFound++;
	    
	    //fill rctInfoVector for all BX read out
	   for (unsigned int iBX=0; iBX<nBX; iBX++){
	     std::vector <RCTInfo> rctInfoData;
	     produceRCTInfo(even[iBX].uint, odd[iBX].uint, iCrate, rctInfoData);
	     allCrateRCTInfo[iBX].push_back(rctInfoData.at(0));
	   }
	   break;
	  }
	}
      }
      
      if(nCratesFound != 18)
	LogDebug("L1T") << "Warning -- only found "<< nCratesFound << " valid crates";
      
     //Step 3: Create Collections from RCTInfo Objects  
      for (uint32_t iBX=0; iBX<nBX; iBX++){
	
	int bx = BCs[iBX];
	
	for(unsigned int iCrate = 0; iCrate < nCratesFound; iCrate++ ){
	 
	 RCTInfo rctInfo = allCrateRCTInfo[iBX].at(iCrate);
	 
	 //Use Crate ID to identify eta/phi of candidate
	 for(int j = 0; j < 4; j++) {
	   
	   L1CaloEmCand em = L1CaloEmCand(rctInfo.neRank[j], 
					  rctInfo.neRegn[j], 
					  rctInfo.neCard[j], 
					  rctInfo.crateID, 
					  iBX,
					  bx);
	   
	   ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 =
	     new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();
	   
	   //l1t::CaloStage1Cluster cluster;
	   CaloEmCand EmCand(*p4,
			     (int) em.rank(),
			     (int) em.regionId().ieta(),
			     (int) em.regionId().iphi(),
			     0);

	   resRCTEmCands_->push_back( iBX, EmCand );
	   
	 }

	 for(int j = 0; j < 4; j++) {

	   L1CaloEmCand em = L1CaloEmCand(rctInfo.ieRank[j], 
					  rctInfo.ieRegn[j], 
					  rctInfo.ieCard[j], 
					  rctInfo.crateID, 
					  iBX,
					  bx);
	   
	   ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 =
	     new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();
	   
	   //l1t::CaloStage1Cluster cluster;
	   CaloEmCand EmCand(*p4,
			     (int) em.rank(),
			     (int) em.regionId().ieta(),
			     (int) em.regionId().iphi(),
			     0);

	   resRCTEmCands_->push_back( iBX, EmCand );
	 }

	 for(int j = 0; j < 7; j++) {

	   for(int k = 0; k < 2; k++) {
	     
	     bool o = (((rctInfo.oBits >> (j * 2 + k)) & 0x1) == 0x1);
	     bool t = (((rctInfo.tBits >> (j * 2 + k)) & 0x1) == 0x1);
	     bool m = (((rctInfo.mBits >> (j * 2 + k)) & 0x1) == 0x1);
	     bool q = (((rctInfo.qBits >> (j * 2 + k)) & 0x1) == 0x1);

	     L1CaloRegion rgn = L1CaloRegion(rctInfo.rgnEt[j][k], o, t, m, q, rctInfo.crateID , j, k);
	     rgn.setBx(bx);
	     ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 =
	       new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();
	     
	     CaloRegion region(*p4, 
			       0., 
			       0., 
			       (int) rgn.et(), 
			       (int) rgn.id().ieta(), 
			       (int) rgn.id().ieta(), 
			       (int) rgn.id().iphi(), 
			       0,
			       0);
	     //region->setBx(bx);
	     // add to ouput
	     resRCTRegions_->push_back( iBX, region );
	   }
	 }

	 for(int j = 0; j < 2; j++) {

	   for(int k = 0; k < 4; k++) {

	     bool fg=(((rctInfo.hfQBits>> (j * 4 + k)) & 0x1)  == 0x1); 

	     L1CaloRegion rgn = L1CaloRegion(rctInfo.hfEt[j][k], fg,  rctInfo.crateID , (j * 4 +  k));
	     rgn.setBx(bx);
	     ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 =
	       new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();
	     
	     CaloRegion region(*p4, 
			       0., 
			       0., 
			       (int) rgn.et(), 
			       (int) rgn.id().ieta(), 
			       (int) rgn.id().ieta(), 
			       (int) rgn.id().iphi(), 
			       0.,
			       0.);

	     resRCTRegions_->push_back( iBX, region );
	   }
	 }
       }
     }
     
     return true;
   }

  bool 
  RCTUnpacker::getBXNumbers(const uint32_t L1aBCID, const uint32_t BXsInCapture, unsigned int BCs[5], uint32_t firstBX, uint32_t lastBX) {

  if (BXsInCapture > 3) {
    if (L1aBCID == 0){
      BCs[0]=3562; BCs[1]=3563; BCs[2]=0; BCs[3]=1; BCs[4]=2;
    }
    else if (L1aBCID == 1){
      BCs[0]=3563; BCs[1]=0; BCs[2]=1; BCs[3]=2; BCs[4]=3;
    }
    else{
      BCs[0]= L1aBCID - 2;
      BCs[1]= L1aBCID - 1;
      BCs[2]= L1aBCID - 0;
      BCs[3]= L1aBCID + 1;
      BCs[4]= L1aBCID + 2;
    }
  } else if (BXsInCapture > 1) {
    if (L1aBCID == 0){
      BCs[0]=3563;
      BCs[1]=0;
      BCs[2]=1;

    }
    else{
      BCs[0]= L1aBCID - 1;
      BCs[1]= L1aBCID - 0;
      BCs[2]= L1aBCID + 1;

    }
  }
  else{
    BCs[0]= L1aBCID;
  }

  firstBX = BCs[0];

  //check if BXs in Capture greater than 5
  if(BXsInCapture<5) lastBX  = BCs[BXsInCapture-1];
  else lastBX = 0;

  return true;
}

  bool RCTUnpacker::decodeCapturedLinkID(uint32_t capturedValue, uint32_t &crateNumber, uint32_t &linkNumber, bool &even)
  {
    
    //if crateNumber not valid set to 0xFF
    crateNumber = ( capturedValue >> 8 ) & 0xFF;
    if(crateNumber > 17)
      crateNumber = 0xFF;
    
    //if linkNumber not valid set to 0xFF
    linkNumber = ( capturedValue ) & 0xFF;

    if(linkNumber > 12)
      linkNumber = 0xFF;
    
    if((linkNumber&0x1) == 0)
      even=true;
    else
      even=false;
    
    return true;
  };
  
  bool RCTUnpacker::produceRCTInfo(const std::vector <unsigned int> evenFiberData, 
				      const std::vector <unsigned int> oddFiberData,
				      const uint32_t crateID,
				      std::vector <RCTInfo> &rctInfoData) {
    
    // Ensure that there is data to process
    unsigned int nWordsToProcess = evenFiberData.size();
    unsigned int remainder = nWordsToProcess%6;

    if(nWordsToProcess != oddFiberData.size()) {
      LogDebug("L1T") << "Error  -- even and odd fiber sizes are different! Aborting";
      return false;
    }

    if(nWordsToProcess == 0|| nWordsToProcess/6 == 0) {
      LogDebug("L1T") << "Error -- evenFiberData is null! Aborting";
      return false;
    }
    else if((nWordsToProcess % 6) != 0) {
      LogDebug("L1T") << "Warning -- Correct Protocol Expects 6x32-bit words! Remainder of "<< nWordsToProcess % 6;
      nWordsToProcess=nWordsToProcess-remainder;
    }
    
    // Extract RCTInfo
    
    unsigned int nBXToProcess = nWordsToProcess / 6;
    
    for(unsigned int iBX = 0; iBX < nBXToProcess; iBX++) {

      RCTInfo rctInfo;
      // extract into rctInfo the data from RCT crate
      // contact RCT Group for bit field data
      rctInfo.rgnEt[0][0]  = (evenFiberData[iBX * 6 + 0] & 0x0003FF00) >>  8;
      rctInfo.rgnEt[0][1]  = (evenFiberData[iBX * 6 + 0] & 0x0FFC0000) >> 18;
      rctInfo.rgnEt[1][0]  = (evenFiberData[iBX * 6 + 0] & 0xF0000000) >> 28;
      rctInfo.rgnEt[1][0] |= (evenFiberData[iBX * 6 + 1] & 0x0000003F) <<  4;
      rctInfo.rgnEt[1][1]  = (evenFiberData[iBX * 6 + 1] & 0x0000FFC0) >>  6;
      rctInfo.rgnEt[2][0]  = (evenFiberData[iBX * 6 + 1] & 0x03FF0000) >> 16;
      rctInfo.rgnEt[2][1]  = (evenFiberData[iBX * 6 + 1] & 0xFC000000) >> 26;
      rctInfo.rgnEt[2][1] |= (evenFiberData[iBX * 6 + 2] & 0x0000000F) <<  6;
      rctInfo.rgnEt[3][0]  = (evenFiberData[iBX * 6 + 2] & 0x00003FF0) >>  4;
      rctInfo.rgnEt[3][1]  = (evenFiberData[iBX * 6 + 2] & 0x00FFC000) >> 14;
      rctInfo.rgnEt[4][0]  = (evenFiberData[iBX * 6 + 2] & 0xFF000000) >> 24;
      rctInfo.rgnEt[4][0] |= (evenFiberData[iBX * 6 + 3] & 0x00000003) <<  8;
      rctInfo.rgnEt[4][1]  = (evenFiberData[iBX * 6 + 3] & 0x00000FFC) >>  2;
      rctInfo.rgnEt[5][0]  = (evenFiberData[iBX * 6 + 3] & 0x003FF000) >> 12;
      rctInfo.rgnEt[5][1]  = (evenFiberData[iBX * 6 + 3] & 0xFFC00000) >> 22;
      rctInfo.rgnEt[6][0]  = (evenFiberData[iBX * 6 + 4] & 0x000003FF) >>  0;
      rctInfo.rgnEt[6][1]  = (evenFiberData[iBX * 6 + 4] & 0x000FFC00) >> 10;
      rctInfo.tBits  = (evenFiberData[iBX * 6 + 4] & 0xFFF00000) >> 20;
      rctInfo.tBits |= (evenFiberData[iBX * 6 + 5] & 0x00000003) << 12;
      rctInfo.oBits  = (evenFiberData[iBX * 6 + 5] & 0x0000FFFC) >>  2;
      rctInfo.c4BC0  = (evenFiberData[iBX * 6 + 5] & 0x000C0000) >> 18;
      rctInfo.c5BC0  = (evenFiberData[iBX * 6 + 5] & 0x00300000) >> 20;
      rctInfo.c6BC0  = (evenFiberData[iBX * 6 + 5] & 0x00C00000) >> 22;
      // Odd fiber bits contain 2x1, HF and other miscellaneous information
      rctInfo.hfEt[0][0]  = (oddFiberData[iBX * 6 + 0] & 0x0000FF00) >>  8;
      rctInfo.hfEt[0][1]  = (oddFiberData[iBX * 6 + 0] & 0x00FF0000) >> 16;
      rctInfo.hfEt[1][0]  = (oddFiberData[iBX * 6 + 0] & 0xFF000000) >> 24;
      rctInfo.hfEt[1][1]  = (oddFiberData[iBX * 6 + 1] & 0x000000FF) >>  0;
      rctInfo.hfEt[0][2]  = (oddFiberData[iBX * 6 + 1] & 0x0000FF00) >>  8;
      rctInfo.hfEt[0][3]  = (oddFiberData[iBX * 6 + 1] & 0x00FF0000) >> 16;
      rctInfo.hfEt[1][2]  = (oddFiberData[iBX * 6 + 1] & 0xFF000000) >> 24;
      rctInfo.hfEt[1][3]  = (oddFiberData[iBX * 6 + 2] & 0x000000FF) >>  0;
      rctInfo.hfQBits     = (oddFiberData[iBX * 6 + 2] & 0x0000FF00) >>  8;
      rctInfo.ieRank[0]   = (oddFiberData[iBX * 6 + 2] & 0x003F0000) >> 16;
      rctInfo.ieRegn[0]   = (oddFiberData[iBX * 6 + 2] & 0x00400000) >> 22;
      rctInfo.ieCard[0]   = (oddFiberData[iBX * 6 + 2] & 0x03800000) >> 23;
      rctInfo.ieRank[1]   = (oddFiberData[iBX * 6 + 2] & 0xFC000000) >> 26;
      rctInfo.ieRegn[1]   = (oddFiberData[iBX * 6 + 3] & 0x00000001) >>  0;
      rctInfo.ieCard[1]   = (oddFiberData[iBX * 6 + 3] & 0x0000000E) >>  1;
      rctInfo.ieRank[2]   = (oddFiberData[iBX * 6 + 3] & 0x000003F0) >>  4;
      rctInfo.ieRegn[2]   = (oddFiberData[iBX * 6 + 3] & 0x00000400) >> 10;
      rctInfo.ieCard[2]   = (oddFiberData[iBX * 6 + 3] & 0x00003800) >> 11;
      rctInfo.ieRank[3]   = (oddFiberData[iBX * 6 + 3] & 0x000FC000) >> 14;
      rctInfo.ieRegn[3]   = (oddFiberData[iBX * 6 + 3] & 0x00100000) >> 20;
      rctInfo.ieCard[3]   = (oddFiberData[iBX * 6 + 3] & 0x00E00000) >> 21;
      rctInfo.neRank[0]   = (oddFiberData[iBX * 6 + 3] & 0x3F000000) >> 24; 
      rctInfo.neRegn[0]   = (oddFiberData[iBX * 6 + 3] & 0x40000000) >> 30;
      rctInfo.neCard[0]   = (oddFiberData[iBX * 6 + 3] & 0x80000000) >> 31; 
      rctInfo.neCard[0]  |= (oddFiberData[iBX * 6 + 4] & 0x00000003) <<  1;
      rctInfo.neRank[1]   = (oddFiberData[iBX * 6 + 4] & 0x000000FC) >>  2;
      rctInfo.neRegn[1]   = (oddFiberData[iBX * 6 + 4] & 0x00000100) >>  8;
      rctInfo.neCard[1]   = (oddFiberData[iBX * 6 + 4] & 0x00000E00) >>  9;
      rctInfo.neRank[2]   = (oddFiberData[iBX * 6 + 4] & 0x0003F000) >> 12;
      rctInfo.neRegn[2]   = (oddFiberData[iBX * 6 + 4] & 0x00040000) >> 18;
      rctInfo.neCard[2]   = (oddFiberData[iBX * 6 + 4] & 0x00380000) >> 19;
      rctInfo.neRank[3]   = (oddFiberData[iBX * 6 + 4] & 0x0FC00000) >> 22;
      rctInfo.neRegn[3]   = (oddFiberData[iBX * 6 + 4] & 0x10000000) >> 28;
      rctInfo.neCard[3]   = (oddFiberData[iBX * 6 + 4] & 0xE0000000) >> 29;
      rctInfo.mBits       = (oddFiberData[iBX * 6 + 5] & 0x00003FFF) >>  0;
      rctInfo.c1BC0       = (oddFiberData[iBX * 6 + 5] & 0x00030000) >> 16;
      rctInfo.c2BC0       = (oddFiberData[iBX * 6 + 5] & 0x000C0000) >> 18;
      rctInfo.c3BC0       = (oddFiberData[iBX * 6 + 5] & 0x00300000) >> 20;
      unsigned int oddFiberc4BC0 = (oddFiberData[iBX * 6 + 5] & 0x00C00000) >> 22;
      if(oddFiberc4BC0 != rctInfo.c4BC0) {
	std::cerr << "Even and odd fibers do not agree on cable 4 BC0 mark :(" << std::endl;
      }
      rctInfo.crateID = crateID;
      rctInfoData.push_back(rctInfo);
    }
    return true;
    
  }
    
  }
}

DEFINE_L1T_UNPACKER(l1t::stage1::RCTUnpacker);
