#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "FWCore/Framework/interface/one/EDProducerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

namespace l1t {
   class MPLink4Unpacker : public BaseUnpacker {
      public:
         MPLink4Unpacker(const edm::ParameterSet&, edm::Event&);
         ~MPLink4Unpacker();
         virtual bool unpack(const unsigned char *data, const unsigned block_id, const unsigned size) override;
      private:
         edm::Event& ev_;
         std::auto_ptr<JetBxCollection> res1_;
         std::auto_ptr<EtSumBxCollection> res2_;

   };

   class MPLink4UnpackerFactory : public BaseUnpackerFactory {
      public:
         MPLink4UnpackerFactory(const edm::ParameterSet&, edm::one::EDProducerBase&);
         virtual std::vector<UnpackerItem> create(edm::Event&, const unsigned& fw, const int fedid);

      private:
         const edm::ParameterSet& cfg_;
         edm::one::EDProducerBase& prod_;
   };
}

// Implementation

namespace l1t {
   MPLink4Unpacker::MPLink4Unpacker(const edm::ParameterSet& cfg, edm::Event& ev) :
      ev_(ev),
      res1_(new JetBxCollection()),
      res2_(new EtSumBxCollection())
   {
   };

   MPLink4Unpacker::~MPLink4Unpacker()
   {
      ev_.put(res1_);
      ev_.put(res2_);
   };

   bool
   MPLink4Unpacker::unpack(const unsigned char *data, const unsigned block_id, const unsigned size)
   {

     LogDebug("L1T") << "Block ID  = " << block_id << " size = " << size;

     res1_->setBXRange(0,1);
     res2_->setBXRange(0,1);

     // Initialise index
     int unsigned i = 0;

     // MET(x)

     uint32_t raw_data = pop(data,i); // pop advances the index i internally

     l1t::EtSum metx = l1t::EtSum();
    
     metx.setHwPt(raw_data & 0xFFFFF);
     metx.setType(l1t::EtSum::kTotalEt);       

     LogDebug("L1T") << "MET(X): pT " << metx.hwPt();

     res2_->push_back(0,metx);


     // Skip 9 empty frames
     for (int j=0; j<9; j++) raw_data=pop(data,i); 

     // HT

     raw_data = pop(data,i); // pop advances the index i internally

     l1t::EtSum mhtx = l1t::EtSum();
    
     mhtx.setHwPt(raw_data & 0xFFFFF);
     mhtx.setType(l1t::EtSum::kTotalHt);       

     LogDebug("L1T") << "MHT(x): pT " << mhtx.hwPt();

     res2_->push_back(0,mhtx);

     // Skip 26 empty frames                                                                                                                                             
     for (int j=0; j<26; j++) raw_data=pop(data,i);

     // Two jets
     for (unsigned nJet=0; nJet < 2; nJet++){
       raw_data = pop(data,i); // pop advances the index i internally

       if (raw_data == 0)
            continue;

       l1t::Jet jet = l1t::Jet();

       jet.setHwEta(-1 * (raw_data & 0x3F));
       jet.setHwPhi((raw_data >> 6) & 0x7F);
       jet.setHwPt((raw_data >> 13) & 0xFFFF);
         
       LogDebug("L1T") << "Jet: eta " << jet.hwEta() << " phi " << jet.hwPhi() << " pT " << jet.hwPt() << " qual " << jet.hwQual();

       res1_->push_back(0,jet);
     }

     return true;
   }

   MPLink4UnpackerFactory::MPLink4UnpackerFactory(const edm::ParameterSet& cfg, edm::one::EDProducerBase& prod) : cfg_(cfg), prod_(prod)
   {
      prod_.produces<JetBxCollection>("MPLink4");
      prod_.produces<EtSumBxCollection>("MPLink4");
   }

   std::vector<UnpackerItem>
   MPLink4UnpackerFactory::create(edm::Event& ev, const unsigned& fw, const int fedid) {
     return {std::make_pair(9, std::shared_ptr<BaseUnpacker>(new MPLink4Unpacker(cfg_, ev)))};
   };
}

DEFINE_L1TUNPACKER(l1t::MPLink4UnpackerFactory);
