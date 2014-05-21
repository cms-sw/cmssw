#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

#include "EtSumUnpacker.h"

namespace l1t {
   class EtSumUnpacker : public BaseUnpacker {
      public:

     virtual bool unpack(const unsigned char *data, const unsigned block_id, const unsigned size);

         virtual void setCollections(UnpackerCollections& coll) {
            res = coll.getEtSumCollection();
         };
      private:
         EtSumBxCollection* res;
   };

   std::vector<UnpackerItem> EtSumUnpackerFactory::create(unsigned fw, const int fedid) {
      return {std::make_pair(3, std::shared_ptr<BaseUnpacker>(new EtSumUnpacker()))};
   };

   bool EtSumUnpacker::unpack(const unsigned char *data, const unsigned block_id, const unsigned size) {

     int nBX = size / 4; // Since there are 4 EtSum objects reported per event (see CMS IN-2013/005)

     // Find the central, first and last BXs
     int firstBX = -(ceil((double)nBX/2.)-1);
     int lastBX;
     if (nBX % 2 == 0) {
       lastBX = ceil((double)nBX/2.)+1;
     } else {
       lastBX = ceil((double)nBX/2.);
     }

     // Initialise index
     int unsigned i = 0;

     // Loop over multiple BX and fill EtSums collection
     for (int bx=firstBX; bx<lastBX; bx++){

       //  MET

       uint32_t raw_data = pop(data,i); // pop advances the index i internally

       l1t::EtSum met = l1t::EtSum();
    
       met.setHwPt(raw_data & 0xFFF);
       met.setHwPhi(raw_data & 0xFF000);
       met.setType(l1t::EtSum::kMissingEt);       

       res->push_back(bx,met);

       // MHT

       raw_data = pop(data,i); // pop advances the index i internally

       l1t::EtSum mht = l1t::EtSum();
    
       mht.setHwPt(raw_data & 0xFFF);
       mht.setHwPhi(raw_data & 0xFF000);
       mht.setType(l1t::EtSum::kMissingHt);       

       res->push_back(bx,mht);       

       // ET

       raw_data = pop(data,i); // pop advances the index i internally

       l1t::EtSum et = l1t::EtSum();
    
       et.setHwPt(raw_data & 0xFFF);
       et.setType(l1t::EtSum::kTotalEt);       

       res->push_back(bx,et);

       // HT

       raw_data = pop(data,i); // pop advances the index i internally

       l1t::EtSum ht = l1t::EtSum();
    
       ht.setHwPt(raw_data & 0xFFF);
       ht.setType(l1t::EtSum::kTotalHt);       

       res->push_back(bx,ht);

     }

     return true;
   }

}
