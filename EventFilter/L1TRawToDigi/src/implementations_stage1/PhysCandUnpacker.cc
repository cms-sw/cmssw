#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "CaloCollections.h"

template<typename T, typename F>
bool
process(const l1t::Block& block, BXVector<T> * coll, F modify) {
   LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

   int nBX = int(ceil(block.header().getSize() / 2.)); 

   // Find the first and last BXs
   int firstBX = -(ceil((double)nBX/2.)-1);
   int lastBX;
   if (nBX % 2 == 0) {
      lastBX = ceil((double)nBX/2.)+1;
   } else {
      lastBX = ceil((double)nBX/2.);
   }

   coll->setBXRange(firstBX, lastBX);

   LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

   // Initialise index
   int unsigned i = 0;

   // Loop over multiple BX and then number of jets filling jet collection
   for (int bx=firstBX; bx<lastBX; bx++){
      uint32_t raw_data0 = block.payload()[i++];
      uint32_t raw_data1 = block.payload()[i++];        

      uint16_t candbit[4];
      candbit[0] = raw_data0 & 0xFFFF;
      candbit[1] = (raw_data0 >> 16) & 0xFFFF;
      candbit[2] = raw_data1 & 0xFFFF;
      candbit[3] = (raw_data1 >> 16) & 0xFFFF;

      for (int icand=0;icand<4;icand++){

         int candPt=candbit[icand] & 0x3F;
         int candEta=(candbit[icand]>>6 ) & 0x7;
         int candEtasign=(candbit[icand]>>9) & 0x1;
         int candPhi=(candbit[icand]>>10) & 0x1F;

         T cand;
         cand.setHwPt(candPt);
         cand.setHwEta((candEtasign << 3) | candEta);
         cand.setHwPhi(candPhi);
         //int qualflag=cand.hwQual();
         //qualflag|= (candPt == 0x3F);
         //cand.setHwQual(qualflag);

         /* std::cout << "cand: eta " << cand.hwEta() << " phi " << cand.hwPhi() << " pT " << cand.hwPt() << " qual " << cand.hwQual() << std::endl; */
         //std::cout << cand.hwPt() << " @ " << cand.hwEta() << ", " << cand.hwPhi() << " > " << cand.hwQual() << " > " << cand.hwIso() << std::endl;
         coll->push_back(bx, modify(cand));
      }
   }

   return true;
}

namespace l1t {
  namespace stage1 {
    class IsoEGammaUnpacker : public Unpacker {
      public:
        virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
    };

    class NonIsoEGammaUnpacker : public Unpacker {
      public:
        virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
    };

    class CentralJetUnpacker : public Unpacker {
      public:
        virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
    };

    class ForwardJetUnpacker : public Unpacker {
      public:
        virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
    };

    class TauUnpacker : public Unpacker {
       public:
        virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
    };

    class IsoTauUnpacker : public Unpacker {
       public:
        virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
    };
  }
}

// Implementation

namespace l1t {
  namespace stage1 {
    bool
    IsoEGammaUnpacker::unpack(const Block& block, UnpackerCollections *coll)
    {
       auto res = static_cast<CaloCollections*>(coll)->getEGammas();
       return process(block, res, [](l1t::EGamma eg) { eg.setHwIso(1); return eg; });
    }

    bool
    NonIsoEGammaUnpacker::unpack(const Block& block, UnpackerCollections *coll)
    {
       auto res = static_cast<CaloCollections*>(coll)->getEGammas();
       return process(block, res, [](const l1t::EGamma& eg) { return eg; });
    }

    bool
    CentralJetUnpacker::unpack(const Block& block, UnpackerCollections *coll)
    {
       auto res = static_cast<CaloCollections*>(coll)->getJets();

       if (res->size(0) != 0)
          edm::LogWarning("L1T") << "Need to unpack central jets before forward ones";

       return process(block, res, [](const l1t::Jet& j) { return j; });
    }

    bool
    ForwardJetUnpacker::unpack(const Block& block, UnpackerCollections *coll)
    {
       auto res = static_cast<CaloCollections*>(coll)->getJets();

       if (res->size(0) != 4)
          edm::LogWarning("L1T") << "Need to unpack central jets before forward ones";

       return process(block, res, [](l1t::Jet j) { j.setHwQual(j.hwQual() | 2); return j; });
    }

    bool
    TauUnpacker::unpack(const Block& block, UnpackerCollections *coll)
    {
       auto res = static_cast<CaloCollections*>(coll)->getTaus();
       return process(block, res, [](const l1t::Tau& t) { return t; });
    }

    bool
    IsoTauUnpacker::unpack(const Block& block, UnpackerCollections *coll)
    {
       auto res = static_cast<CaloCollections*>(coll)->getIsoTaus();
       return process(block, res, [](const l1t::Tau& t) { return t; });
    }
  }
}

DEFINE_L1T_UNPACKER(l1t::stage1::IsoEGammaUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage1::NonIsoEGammaUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage1::CentralJetUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage1::ForwardJetUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage1::TauUnpacker);
DEFINE_L1T_UNPACKER(l1t::stage1::IsoTauUnpacker);
