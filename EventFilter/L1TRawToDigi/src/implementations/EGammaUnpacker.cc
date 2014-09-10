#include "DataFormats/L1Trigger/interface/EGamma.h"

#include "FWCore/Framework/interface/one/EDProducerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

namespace l1t {
   class EGammaUnpacker : public BaseUnpacker {
      public:
         EGammaUnpacker(EGammaBxCollection*);
         virtual bool unpack(const unsigned char *data, const unsigned block_id, const unsigned size) override;
      private:
         EGammaBxCollection* res_;
   };

   class EGammaUnpackerFactory : public BaseUnpackerFactory {
      public:
         EGammaUnpackerFactory(const edm::ParameterSet&, edm::one::EDProducerBase&);
         virtual std::vector<UnpackerItem> create(const unsigned& fw, const int fedid) override;
         virtual void beginEvent(edm::Event&) override;
         virtual void endEvent(edm::Event&) override;

      private:
         const edm::ParameterSet& cfg_;
         edm::one::EDProducerBase& prod_;

         std::auto_ptr<EGammaBxCollection> res_;
   };
}

// Implementation

namespace l1t {
   EGammaUnpacker::EGammaUnpacker(EGammaBxCollection* coll) :
      res_(coll)
   {
   };

   bool
   EGammaUnpacker::unpack(const unsigned char *data, const unsigned block_id, const unsigned size)
   {

     LogDebug("L1T") << "Block ID  = " << block_id << " size = " << size;

     int nBX = int(ceil(size / 12.)); // Since there are 12 EGamma objects reported per event (see CMS IN-2013/005)

     // Find the central, first and last BXs
     int firstBX = -(ceil((double)nBX/2.)-1);
     int lastBX;
     if (nBX % 2 == 0) {
       lastBX = ceil((double)nBX/2.)+1;
     } else {
       lastBX = ceil((double)nBX/2.);
     }

     res_->setBXRange(firstBX, lastBX);

     LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

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
       
         LogDebug("L1T") << "EG: eta " << eg.hwEta() << " phi " << eg.hwPhi() << " pT " << eg.hwPt() << " iso " << eg.hwIso() << " qual " << eg.hwQual();

         res_->push_back(bx,eg);
       }

     }

     return true;
   }

   EGammaUnpackerFactory::EGammaUnpackerFactory(const edm::ParameterSet& cfg, edm::one::EDProducerBase& prod) : cfg_(cfg), prod_(prod)
   {
      prod_.produces<EGammaBxCollection>();
   }

   void
   EGammaUnpackerFactory::beginEvent(edm::Event& ev)
   {
      res_.reset(new EGammaBxCollection());
   }

   void
   EGammaUnpackerFactory::endEvent(edm::Event& ev)
   {
      ev.put(res_);
      res_.reset();
   }

  std::vector<UnpackerItem>
  EGammaUnpackerFactory::create(const unsigned& fw, const int fedid)
   {
     
     // This unpacker is only appropriate for the Demux card output (FED ID=1). Anything else should not be unpacked.
     
     if (fedid==1){
       
       return {std::make_pair(1, std::shared_ptr<BaseUnpacker>(new EGammaUnpacker(res_.get())))};
       
     } else {
       
       return {};
     }
     
   };
}

DEFINE_L1TUNPACKER(l1t::EGammaUnpackerFactory);
