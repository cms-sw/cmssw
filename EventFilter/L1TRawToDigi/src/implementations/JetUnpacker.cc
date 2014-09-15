#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

#include "L1TCollections.h"

namespace l1t {
   class JetUnpacker : public BaseUnpacker {
      public:
         JetUnpacker(UnpackerCollections* c) : BaseUnpacker(c) {};
         virtual bool unpack(const unsigned char *data, const unsigned block_id, const unsigned size) override;
   };

   class JetUnpackerFactory : public BaseUnpackerFactory {
      public:
         virtual std::vector<UnpackerItem> create(const unsigned& fw, const int fedid, UnpackerCollections*) override;
   };
}

// Implementation

namespace l1t {
   bool
   JetUnpacker::unpack(const unsigned char *data, const unsigned block_id, const unsigned size)
   {

     LogDebug("L1T") << "Block ID  = " << block_id << " size = " << size;

     int nBX = int(ceil(size / 12.)); // Since there are 12 jets reported per event (see CMS IN-2013/005)

     // Find the first and last BXs
     int firstBX = -(ceil((double)nBX/2.)-1);
     int lastBX;
     if (nBX % 2 == 0) {
       lastBX = ceil((double)nBX/2.)+1;
     } else {
       lastBX = ceil((double)nBX/2.);
     }

     auto res_ = static_cast<L1TCollections*>(collections_)->getJets();
     res_->setBXRange(firstBX, lastBX);

     LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

     // Initialise index
     int unsigned i = 0;

     // Loop over multiple BX and then number of jets filling jet collection
     for (int bx=firstBX; bx<lastBX; bx++){
       for (unsigned nJet=0; nJet < 12 && nJet < size; nJet++){
         uint32_t raw_data = pop(data,i); // pop advances the index i internally

         if (raw_data == 0)
            continue;

         l1t::Jet jet = l1t::Jet();

         jet.setHwPt(raw_data & 0x7FF);
         
         int abs_eta = (raw_data >> 11) & 0x7F;
         if ((raw_data >> 18) & 0x1) {
           jet.setHwEta(-1 * abs_eta);
         } else {
           jet.setHwEta(abs_eta);
         }

	 jet.setHwPhi((raw_data >> 19) & 0xFF);
         jet.setHwQual((raw_data >> 27) & 0x7); // Assume 3 bits for now? Leaves 2 bits spare

         LogDebug("L1T") << "Jet: eta " << jet.hwEta() << " phi " << jet.hwPhi() << " pT " << jet.hwPt() << " qual " << jet.hwQual();

         res_->push_back(bx,jet);
       }
     }

     return true;
   }

   std::vector<UnpackerItem>
   JetUnpackerFactory::create(const unsigned& fw, const int fedid, UnpackerCollections* coll) {

     // This unpacker is only appropriate for the Demux card output (FED ID=1). Anything else should not be unpacked.                                                     
     if (fedid==1){

       return {std::make_pair(5, std::shared_ptr<BaseUnpacker>(new JetUnpacker(coll)))};

     } else {

       return {};

     }

   };
}

DEFINE_L1TUNPACKER(l1t::JetUnpackerFactory);
