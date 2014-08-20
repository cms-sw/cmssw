#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "FWCore/Framework/interface/one/EDProducerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

namespace l1t {
   class MPUnpacker : public BaseUnpacker {
      public:
         MPUnpacker(const edm::ParameterSet&, JetBxCollection* coll1, EtSumBxCollection* coll2);
         virtual bool unpack(const unsigned char *data, const unsigned block_id, const unsigned size) override;
      private:
         JetBxCollection* res1_;
         EtSumBxCollection* res2_;

   };

   class MPUnpackerFactory : public BaseUnpackerFactory {
      public:
         MPUnpackerFactory(const edm::ParameterSet&, edm::one::EDProducerBase&);
         virtual std::vector<UnpackerItem> create(const unsigned& fw, const int fedid) override;
         virtual void beginEvent(edm::Event&) override;
         virtual void endEvent(edm::Event&) override;

      private:
         const edm::ParameterSet& cfg_;
         edm::one::EDProducerBase& prod_;

         std::auto_ptr<JetBxCollection> res1_;
         std::auto_ptr<EtSumBxCollection> res2_;
   };
}

// Implementation

namespace l1t {
   MPUnpacker::MPUnpacker(const edm::ParameterSet& cfg, JetBxCollection* coll1, EtSumBxCollection* coll2) :
      res1_(coll1),
      res2_(coll2)
   {
   };

   bool
   MPUnpacker::unpack(const unsigned char *data, const unsigned block_id, const unsigned size)
   {

     LogDebug("L1T") << "Block ID  = " << block_id << " size = " << size;

     res1_->setBXRange(0,1);
     res2_->setBXRange(0,1);

     // Initialise index
     int unsigned i = 0;

     // ET / MET(x) / MET (y)

     uint32_t raw_data = pop(data,i); // pop advances the index i internally

     l1t::EtSum et = l1t::EtSum();
    
     et.setHwPt(raw_data & 0xFFFFF);
     et.setType(l1t::EtSum::kTotalEt);       

     LogDebug("L1T") << "ET/METx/METy: pT " << et.hwPt();

     res2_->push_back(0,et);

     // Skip 9 empty frames
     for (int j=0; j<9; j++) raw_data=pop(data,i); 

     // HT / MHT(x)/ MHT (y)

     raw_data = pop(data,i); // pop advances the index i internally

     l1t::EtSum ht = l1t::EtSum();
    
     ht.setHwPt(raw_data & 0xFFFFF);
     ht.setType(l1t::EtSum::kTotalHt);       

     LogDebug("L1T") << "HT/MHTx/MHTy: pT " << ht.hwPt();

     res2_->push_back(0,ht);

     // Skip 26 empty frames                                                                                                                                             
     for (int j=0; j<26; j++) raw_data=pop(data,i);

     // Two jets
     for (unsigned nJet=0; nJet < 2; nJet++){
       raw_data = pop(data,i); // pop advances the index i internally

       if (raw_data == 0)
            continue;

       l1t::Jet jet = l1t::Jet();

       int etasign = 1;
       if ((block_id == 7) ||
           (block_id == 9) ||
           (block_id == 11)) {
         etasign = -1;
       }

       LogDebug("L1") << "block ID=" << block_id << " etasign=" << etasign;

       jet.setHwEta(etasign*(raw_data & 0x3F));
       jet.setHwPhi((raw_data >> 6) & 0x7F);
       jet.setHwPt((raw_data >> 13) & 0xFFFF);
         
       LogDebug("L1T") << "Jet: eta " << jet.hwEta() << " phi " << jet.hwPhi() << " pT " << jet.hwPt() << " qual " << jet.hwQual();

       res1_->push_back(0,jet);
     }

     return true;
   }

   MPUnpackerFactory::MPUnpackerFactory(const edm::ParameterSet& cfg, edm::one::EDProducerBase& prod) : cfg_(cfg), prod_(prod)
   {
      prod_.produces<JetBxCollection>("MP");
      prod_.produces<EtSumBxCollection>("MP");
   }

   void
   MPUnpackerFactory::beginEvent(edm::Event& ev)
   {
      res1_.reset(new JetBxCollection());
      res2_.reset(new EtSumBxCollection());
   }

   void
   MPUnpackerFactory::endEvent(edm::Event& ev)
   {
      ev.put(res1_);
      ev.put(res2_);
      res1_.reset();
      res2_.reset();
   }

   std::vector<UnpackerItem>
   MPUnpackerFactory::create(const unsigned& fw, const int fedid) {

     // This unpacker is only appropriate for the Main Processor output (FED ID=2). Anything else should not be unpacked.
     
     if (fedid==2){

       std::vector<UnpackerItem> linkMap;
    
       auto unpacker = std::shared_ptr<BaseUnpacker>(new MPUnpacker(cfg_, res1_.get(), res2_.get()));

       // Six links are used to output the data
       
       linkMap.push_back(std::make_pair(1, unpacker));
       linkMap.push_back(std::make_pair(3, unpacker));
       linkMap.push_back(std::make_pair(5, unpacker));
       linkMap.push_back(std::make_pair(7, unpacker));
       linkMap.push_back(std::make_pair(9, unpacker));
       linkMap.push_back(std::make_pair(11, unpacker));

       return linkMap;

     } else {

       return {};

     }

   };
}

DEFINE_L1TUNPACKER(l1t::MPUnpackerFactory);
