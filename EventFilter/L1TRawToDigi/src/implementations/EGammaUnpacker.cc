#include "DataFormats/L1Trigger/interface/EGamma.h"

#include "FWCore/Framework/interface/one/EDProducerBase.h"
#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

namespace l1t {
   class EGammaUnpacker : public BaseUnpacker {
      public:
         EGammaUnpacker(const edm::ParameterSet&, edm::Event&);
         ~EGammaUnpacker();
         virtual bool unpack(const unsigned char *data, const unsigned block_id, const unsigned size) override;
      private:
         edm::Event& ev_;
         std::auto_ptr<EGammaBxCollection> res_;
   };

   class EGammaUnpackerFactory : public BaseUnpackerFactory {
      public:
         EGammaUnpackerFactory(const edm::ParameterSet&, edm::one::EDProducerBase&);
         virtual std::vector<UnpackerItem> create(edm::Event&, const unsigned& fw, const int fedid) override;

      private:
         const edm::ParameterSet& cfg_;
         edm::one::EDProducerBase& prod_;
   };
}

// Implementation

namespace l1t {
   EGammaUnpacker::EGammaUnpacker(const edm::ParameterSet& cfg, edm::Event& ev) :
      ev_(ev),
      res_(new EGammaBxCollection())
   {
   };

   EGammaUnpacker::~EGammaUnpacker()
   {
      ev_.put(res_);
   };

   bool
   EGammaUnpacker::unpack(const unsigned char *data, const unsigned block_id, const unsigned size)
   {
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
       
         res_->push_back(bx,eg);
       }

     }

     return true;
   }

   EGammaUnpackerFactory::EGammaUnpackerFactory(const edm::ParameterSet& cfg, edm::one::EDProducerBase& prod) : cfg_(cfg), prod_(prod)
   {
      prod_.produces<EGammaBxCollection>();
   }

   std::vector<UnpackerItem>
   EGammaUnpackerFactory::create(edm::Event& ev, const unsigned& fw, const int fedid)
   {
      return {std::make_pair(1, std::shared_ptr<BaseUnpacker>(new EGammaUnpacker(cfg_, ev)))};
   };
}

DEFINE_L1TUNPACKER(l1t::EGammaUnpackerFactory);
