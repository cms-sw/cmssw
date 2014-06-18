#include "DataFormats/L1Trigger/interface/Jet.h"

#include "FWCore/Framework/interface/one/EDProducerBase.h"
#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

namespace l1t {
   class JetUnpacker : public BaseUnpacker {
      public:
         JetUnpacker(const edm::ParameterSet&, edm::Event&);
         ~JetUnpacker();
         virtual bool unpack(const unsigned char *data, const unsigned block_id, const unsigned size) override;
      private:
         edm::Event& ev_;
         std::auto_ptr<JetBxCollection> res_;
   };

   class JetUnpackerFactory : public BaseUnpackerFactory {
      public:
         JetUnpackerFactory(const edm::ParameterSet&, edm::one::EDProducerBase&);
         virtual std::vector<UnpackerItem> create(edm::Event&, const unsigned& fw, const int fedid);

      private:
         const edm::ParameterSet& cfg_;
         edm::one::EDProducerBase& prod_;
   };
}

// Implementation

namespace l1t {
   JetUnpacker::JetUnpacker(const edm::ParameterSet& cfg, edm::Event& ev) :
      ev_(ev),
      res_(new JetBxCollection())
   {
   };

   JetUnpacker::~JetUnpacker()
   {
      ev_.put(res_);
   };

   bool
   JetUnpacker::unpack(const unsigned char *data, const unsigned block_id, const unsigned size)
   {
     int nBX = int(ceil(size / 12.)); // Since there are 12 jets reported per event (see CMS IN-2013/005)

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

         res_->push_back(bx,jet);
       }
     }

     return true;
   }

   JetUnpackerFactory::JetUnpackerFactory(const edm::ParameterSet& cfg, edm::one::EDProducerBase& prod) : cfg_(cfg), prod_(prod)
   {
      prod_.produces<JetBxCollection>();
   }

   std::vector<UnpackerItem>
   JetUnpackerFactory::create(edm::Event& ev, const unsigned& fw, const int fedid) {
      return {std::make_pair(5, std::shared_ptr<BaseUnpacker>(new JetUnpacker(cfg_, ev)))};
   };
}

DEFINE_L1TUNPACKER(l1t::JetUnpackerFactory);
