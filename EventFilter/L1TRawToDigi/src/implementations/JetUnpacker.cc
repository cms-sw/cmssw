#include "DataFormats/L1Trigger/interface/Jet.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

#include "JetUnpacker.h"

namespace l1t {
   class JetUnpacker : public BaseUnpacker {
      public:
         virtual bool unpack(const unsigned char *data, const unsigned block_id, const unsigned size);

         virtual void setCollections(UnpackerCollections& coll) {
            res = coll.getJetCollection();
         };
      private:
         JetBxCollection* res;
   };

   std::vector<UnpackerItem> JetUnpackerFactory::create(const FirmwareVersion& fw, const int fedid) {
      return {std::make_pair(0xF, std::shared_ptr<BaseUnpacker>(new JetUnpacker()))};
   };

   bool JetUnpacker::unpack(const unsigned char *data, const unsigned block_id, const unsigned size) {

     int nBX = size / 12; // Since there are 12 jets reported per event (see CMS IN-2013/005)

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

     // Loop over multiple BX and then number of jets filling jet collection
     for (int bx=firstBX; bx<lastBX; bx++){

       for (unsigned nJet=0; nJet < 12; nJet++){

         uint32_t raw_data = pop(data,i); // pop advances the index i internally

         l1t::Jet jet = l1t::Jet();
    
         jet.setHwPt(raw_data & 0x7FF);
         jet.setHwPhi(raw_data & 0x7F80000);
         if (raw_data & 0x40000) {
           jet.setHwEta(-1*(raw_data & 0x3F800));
         } else {
           jet.setHwEta(raw_data & 0x3F800);
         }

         jet.setHwQual(raw_data & 0x38000000); // Assume 3 bits for now? Leaves 4 bits spare
       
         res->push_back(bx,jet);

       }

     }

     return true;
   }

}
