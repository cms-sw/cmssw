#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "FWCore/Framework/interface/one/EDProducerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

namespace l1t {
   class MPUnpacker : public BaseUnpacker {
      public:
         MPUnpacker(const edm::ParameterSet&, edm::Event&);
         ~MPUnpacker();
         virtual bool unpack(const unsigned char *data, const unsigned block_id, const unsigned size) override;
      private:
         edm::Event& ev_;
         std::auto_ptr<JetBxCollection> res1_;
         std::auto_ptr<EtSumBxCollection> res2_;

   };

   class MPUnpackerFactory : public BaseUnpackerFactory {
      public:
         MPUnpackerFactory(const edm::ParameterSet&, edm::one::EDProducerBase&);
         virtual std::vector<UnpackerItem> create(edm::Event&, const unsigned& fw, const int fedid);

      private:
         const edm::ParameterSet& cfg_;
         edm::one::EDProducerBase& prod_;
   };
}

// Implementation

namespace l1t {
   MPUnpacker::MPUnpacker(const edm::ParameterSet& cfg, edm::Event& ev) :
      ev_(ev),
      res1_(new JetBxCollection()),
      res2_(new EtSumBxCollection())
   {
   };

   MPUnpacker::~MPUnpacker()
   {
      ev_.put(res1_);
      ev_.put(res2_);
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

   std::vector<UnpackerItem>
   MPUnpackerFactory::create(edm::Event& ev, const unsigned& fw, const int fedid) {

     // This unpacker is only appropriate for the Main Processor output (FED ID=2). Anything else should not be unpacked.
     
     if (fedid==2){

       std::vector<UnpackerItem> linkMap;
    
       auto unpacker = std::shared_ptr<BaseUnpacker>(new MPUnpacker(cfg_, ev));

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
