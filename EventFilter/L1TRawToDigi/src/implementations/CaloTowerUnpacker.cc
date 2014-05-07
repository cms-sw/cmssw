#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

#include "CaloTowerUnpacker.h"

namespace l1t {
   class CaloTowerUnpacker : public BaseUnpacker {
      public:

     virtual bool unpack(const unsigned char *data, const unsigned block_id, const unsigned size);

         virtual void setCollections(UnpackerCollections& coll) {
            res = coll.getCaloTowerCollection();
         };
      private:
         CaloTowerBxCollection* res;
   };

  std::vector<UnpackerItem> CaloTowerUnpackerFactory::create(unsigned fw, const int fedid) {
    std::vector<UnpackerItem> towersMap;
    
     // Map all even number links, which are Rx links and need unpacking to the same instance of the CaloTowerUnpacker
     // which receives the block_ID and can convert this to phi

     auto unpacker = std::shared_ptr<BaseUnpacker>(new CaloTowerUnpacker());

     for (int link = 0; link < 144; link++){
       if (link % 2 == 0) towersMap.push_back(std::make_pair(link, unpacker)); 
     }
     
     return towersMap;

   };

  bool CaloTowerUnpacker::unpack(const unsigned char *data, const unsigned block_id, const unsigned size) {
    
     int nBX = size/82; // Since there is one Rx link per block with 2*28 slices in barrel and endcap + 2*13 for upgraded HF - check this!!

     // Find the first and last BXs
     int firstBX = -(std::ceil((double)nBX/2.)-1);
     int lastBX;
     if (nBX % 2 == 0) {
       lastBX = std::ceil((double)nBX/2.)+1;
     } else {
       lastBX = std::ceil((double)nBX/2.);
     }
     
     // Initialise index
     int unsigned i = 0;

     // Loop over multiple BX and fill towers collection
     for (int bx=firstBX; bx<lastBX; bx++){

       for (int frame=0; frame<82; frame++){

	 uint32_t raw_data = pop(data,i); // pop advances the index i internally

	 l1t::CaloTower tower1 = l1t::CaloTower();
    
	 // First calo tower is in the LSW with phi
	 tower1.setHwPt(raw_data & 0x1FF);
	 tower1.setHwQual(raw_data & 0xF300);
         tower1.setHwEtRatio(raw_data & 0xE00);
	 tower1.setHwPhi((block_id/2)+1); // iPhi starts at 1
	 
	 if (frame % 2==0) { // Even number links carry Eta+
	   tower1.setHwEta(1+frame/2); // iEta starts at 1
	 } else { // Odd number links carry Eta-
	   tower1.setHwEta(-1*(1+frame/2));
	 }
	 
	 res->push_back(bx,tower1);

	 // Second calo tower is in the MSW with phi+1
	 l1t::CaloTower tower2 = l1t::CaloTower();
	 
	 tower2.setHwPt((raw_data >> 16) & 0x1FF);
	 tower2.setHwQual((raw_data >> 16 )& 0xF300);
         tower2.setHwEtRatio((raw_data >> 16) & 0xE00);
	 tower2.setHwPhi((block_id/2)+2);

	 if (frame % 2==0) {
	   tower1.setHwEta(1+frame/2);
	 } else {
	   tower1.setHwEta(-1*(1+frame/2));
	 }
	 
	 res->push_back(bx,tower2);
	 
       }
     }
     
     return true;

  }
};
