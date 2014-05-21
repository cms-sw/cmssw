#include "DataFormats/L1Trigger/interface/Tau.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

#include "TauUnpacker.h"

namespace l1t {
   class TauUnpacker : public BaseUnpacker {
      public:
     
     virtual bool unpack(const unsigned char *data, const unsigned block_id, const unsigned size);

         virtual void setCollections(UnpackerCollections& coll) {
            res = coll.getTauCollection();
         };
      private:
         TauBxCollection* res;
   };

  std::vector<UnpackerItem> TauUnpackerFactory::create(unsigned fw, const int fedid) {
      return {std::make_pair(7, std::shared_ptr<BaseUnpacker>(new TauUnpacker()))};
   };

   bool TauUnpacker::unpack(const unsigned char *data, const unsigned block_id, const unsigned size) {

     int nBX = size / 8; // Since there are 8 Tau objects reported per event (see CMS IN-2013/005)

     // Find the first and last BXs
     int firstBX = -(ceil((double)nBX/2.)-1);
     int lastBX;
     if (nBX % 2 == 0) {
       lastBX = ceil((double)nBX/2.)+1;
     } else {
       lastBX = ceil((double)nBX/2.);
     }

     // Initialise index
     int unsigned i = 0;

     // Loop over multiple BX and then number of Tau cands filling collection
     for (int bx=firstBX; bx<lastBX; bx++){

       for (unsigned nTau=0; nTau < 12; nTau++){

         uint32_t raw_data = pop(data,i); // pop advances the index i internally

         l1t::Tau tau = l1t::Tau();
       
         tau.setHwPt(raw_data & 0x1FF);
         tau.setHwPhi(raw_data & 0x1FE0000);
         if (raw_data & 0x10000) {
           tau.setHwEta(-1*(raw_data & 0xFE00));
         } else {
           tau.setHwEta(raw_data & 0xFE00);
         }
         tau.setHwIso(raw_data & 0x2000000); // Assume one bit for now?
         tau.setHwQual(raw_data & 0x1C000000); // Assume 3 bits for now? leaves 3 spare bits

         res->push_back(bx,tau);

       }

     }

     return true;
   }

}
