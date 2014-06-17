#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

#include "FWCore/Framework/interface/one/EDProducerBase.h"
#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

namespace l1t {
   class CaloTowerUnpacker : public BaseUnpacker {
      public:
         CaloTowerUnpacker(const edm::ParameterSet&, edm::Event&);
         ~CaloTowerUnpacker();
         virtual bool unpack(const unsigned char *data, const unsigned block_id, const unsigned size);
      private:
         edm::Event& ev_;
         std::auto_ptr<CaloTowerBxCollection> res_;
   };

   class CaloTowerUnpackerFactory : public BaseUnpackerFactory {
      public:
         CaloTowerUnpackerFactory(const edm::ParameterSet&, edm::one::EDProducerBase&);
         virtual std::vector<UnpackerItem> create(edm::Event&, const unsigned& fw, const int fedid);

      private:
         const edm::ParameterSet& cfg_;
         edm::one::EDProducerBase& prod_;
   };
}

// Implementation

namespace l1t {
   CaloTowerUnpacker::CaloTowerUnpacker(const edm::ParameterSet& cfg, edm::Event& ev) :
      ev_(ev),
      res_(new CaloTowerBxCollection())
   {
   };

   CaloTowerUnpacker::~CaloTowerUnpacker()
   {
      ev_.put(res_);
   };

   bool
   CaloTowerUnpacker::unpack(const unsigned char *data, const unsigned block_id, const unsigned size)
   {
      int nBX = int(ceil(size/82.)); // Since there is one Rx link per block with 2*28 slices in barrel and endcap + 2*13 for upgraded HF - check this!!

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

       for (unsigned frame=0; frame<82 && frame<size; frame++){

	 uint32_t raw_data = pop(data,i); // pop advances the index i internally

	 l1t::CaloTower tower1 = l1t::CaloTower();
    
	 // First calo tower is in the LSW with phi
	 tower1.setHwPt(raw_data & 0x1FF);
	 tower1.setHwQual((raw_data >> 12) & 0xF);
         tower1.setHwEtRatio((raw_data >>9) & 0x7);
	 tower1.setHwPhi((block_id/2)+1); // iPhi starts at 1
	 
	 if (frame % 2==0) { // Even number links carry Eta+
	   tower1.setHwEta(1+frame/2); // iEta starts at 1
	 } else { // Odd number links carry Eta-
	   tower1.setHwEta(-1*(1+frame/2));
	 }
	 
	 res_->push_back(bx,tower1);

	 // Second calo tower is in the MSW with phi+1
	 l1t::CaloTower tower2 = l1t::CaloTower();
	 
	 tower2.setHwPt((raw_data >> 16) & 0x1FF);
	 tower2.setHwQual((raw_data >> 28 ) & 0xF);
         tower2.setHwEtRatio((raw_data >> 25) & 0x7);
	 tower2.setHwPhi((block_id/2)+2);

	 if (frame % 2==0) {
	   tower1.setHwEta(1+frame/2);
	 } else {
	   tower1.setHwEta(-1*(1+frame/2));
	 }
	 
	 res_->push_back(bx,tower2);
	 
       }
     }
     
     return true;

  }

   CaloTowerUnpackerFactory::CaloTowerUnpackerFactory(const edm::ParameterSet& cfg, edm::one::EDProducerBase& prod) : cfg_(cfg), prod_(prod)
   {
      prod_.produces<CaloTowerBxCollection>();
   }

   std::vector<UnpackerItem>
   CaloTowerUnpackerFactory::create(edm::Event& ev, const unsigned& fw, const int fedid)
   {
    std::vector<UnpackerItem> towersMap;
    
     // Map all even number links, which are Rx links and need unpacking to the same instance of the CaloTowerUnpacker
     // which receives the block_ID and can convert this to phi

     auto unpacker = std::shared_ptr<BaseUnpacker>(new CaloTowerUnpacker(cfg_, ev));

     for (int link = 0; link < 144; link++){
       if (link % 2 == 0) towersMap.push_back(std::make_pair(link, unpacker)); 
     }
     
     return towersMap;

   };
};

DEFINE_L1TUNPACKER(l1t::CaloTowerUnpackerFactory);
