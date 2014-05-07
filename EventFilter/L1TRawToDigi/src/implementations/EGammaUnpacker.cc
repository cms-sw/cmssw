#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

#include "EGammaUnpacker.h"

namespace l1t {
   class EGammaUnpacker : public BaseUnpacker {
      public:
     
     virtual bool unpack(const unsigned char *data, const unsigned block_id, const unsigned size);

         virtual void setCollections(UnpackerCollections& coll) {
            res = coll.getEGammaCollection();
         };
      private:
         EGammaBxCollection* res;
   };

  std::vector<UnpackerItem> EGammaUnpackerFactory::create(unsigned fw, const int fedid) {
      return {std::make_pair(1, std::shared_ptr<BaseUnpacker>(new EGammaUnpacker()))};
   };

   bool EGammaUnpacker::unpack(const unsigned char *data, const unsigned block_id, const unsigned size) {

     int nBX = size / 12; // Since there are 12 EGamma objects reported per event (see CMS IN-2013/005)

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

     // Loop over multiple BX and then number of EG cands filling collection
     for (int bx=firstBX; bx<lastBX; bx++){

       for (unsigned nEG=0; nEG < 12; nEG++){

         uint32_t raw_data = pop(data,i); // pop advances the index i internally

         l1t::EGamma eg = l1t::EGamma();
    
         eg.setHwPt(raw_data & 0x1FF);
         eg.setHwPhi(raw_data & 0x1FE0000);
         if (raw_data & 0x10000) {
           eg.setHwEta(-1*(raw_data & 0xFE00));
         } else {
           eg.setHwEta(raw_data & 0xFE00);
         }
         eg.setHwIso(raw_data & 0x2000000); // Assume one bit for now?
         eg.setHwQual(raw_data & 0x1C000000); // Assume 3 bits for now? leaves 3 spare bits
       
         res->push_back(bx,eg);

       }

     }

     return true;
   }

}
