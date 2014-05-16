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

  std::vector<UnpackerItem> EGammaUnpackerFactory::create(const unsigned fw, const int fedid) {
      return {std::make_pair(1, std::shared_ptr<BaseUnpacker>(new EGammaUnpacker()))};
   };

   bool EGammaUnpacker::unpack(const unsigned char *data, const unsigned block_id, const unsigned size) {

     int nBX = int(ceil(size / 12.)); // Since there are 12 EGamma objects reported per event (see CMS IN-2013/005)

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

       for (unsigned nEG=0; nEG < 12 && nEG < size; nEG++){

         uint32_t raw_data = pop(data,i); // pop advances the index i internally

         // skip padding to bring EG candidates up to 12 pre BX
         if (raw_data == 0)
            continue;

         l1t::EGamma eg = l1t::EGamma();
    
         eg.setHwPt(raw_data & 0x1FF);

	 int abs_eta = (raw_data >> 9) & 0x7F;
         if ((raw_data >> 16) & 0x1) {
           eg.setHwEta(-1*abs_eta);
         } else {
           eg.setHwEta(abs_eta);
         }

         eg.setHwPhi((raw_data >> 17) & 0xFF);
	 eg.setHwIso((raw_data >> 25) & 0x1); // Assume one bit for now?
	 eg.setHwQual((raw_data >> 26) & 0x7); // Assume 3 bits for now? leaves 3 spare bits
       
         res->push_back(bx,eg);

       }

     }

     return true;
   }

}
