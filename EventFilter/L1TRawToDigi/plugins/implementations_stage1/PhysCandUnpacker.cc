#include "FWCore/Framework/interface/MakerMacros.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "CaloCollections.h"
#include "PhysCandUnpacker.h"

template <typename T, typename F>
bool process(const l1t::Block& block, BXVector<T>* coll, F modify, bool isleft, bool isfirst, bool istau) {
  LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

  int nBX, firstBX, lastBX;
  nBX = int(ceil(block.header().getSize() / 2.));
  l1t::getBXRange(nBX, firstBX, lastBX);

  coll->setBXRange(firstBX, lastBX);

  LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

  // Initialise index
  int unsigned i = 0;

  // Loop over multiple BX and then number of jets filling jet collection
  for (int bx = firstBX; bx <= lastBX; bx++) {
    if (!istau)
      coll->resize(bx, 8);
    else
      coll->resize(bx, 4);

    uint32_t raw_data0 = block.payload()[i++];
    uint32_t raw_data1 = block.payload()[i++];

    uint16_t candbit[2];
    candbit[0] = raw_data0 & 0xFFFF;
    candbit[1] = raw_data1 & 0xFFFF;

    for (int icand = 0; icand < 2; icand++) {
      int candPt = candbit[icand] & 0x3F;
      int candEta = (candbit[icand] >> 6) & 0x7;
      int candEtasign = (candbit[icand] >> 9) & 0x1;
      int candPhi = (candbit[icand] >> 10) & 0x1F;

      T cand;
      cand.setHwPt(candPt);
      cand.setHwEta((candEtasign << 3) | candEta);
      cand.setHwPhi(candPhi);
      //int qualflag=cand.hwQual();
      //qualflag|= (candPt == 0x3F);
      //cand.setHwQual(qualflag);

      /* std::cout << "cand: eta " << cand.hwEta() << " phi " << cand.hwPhi() << " pT " << cand.hwPt() << " qual " << cand.hwQual() << std::endl; */
      //std::cout << cand.hwPt() << " @ " << cand.hwEta() << ", " << cand.hwPhi() << " > " << cand.hwQual() << " > " << cand.hwIso() << std::endl;

      if (isfirst) {
        if (isleft) {
          coll->set(bx, 2 * icand, modify(cand));
        } else if (!isleft) {
          coll->set(bx, 2 * icand + 1, modify(cand));
        }
      } else if (!isfirst) {
        if (isleft) {
          coll->set(bx, 2 * icand + 4, modify(cand));
        } else if (!isleft) {
          coll->set(bx, 2 * icand + 5, modify(cand));
        }
      }
    }
  }

  return true;
}

namespace l1t {
  namespace stage1 {
    bool IsoEGammaUnpackerLeft::unpack(const Block& block, UnpackerCollections* coll) {
      auto res = static_cast<CaloCollections*>(coll)->getEGammas();
      return process(
          block,
          res,
          [](l1t::EGamma eg) {
            eg.setHwIso(1);
            return eg;
          },
          true,
          true,
          false);
    }

    bool NonIsoEGammaUnpackerLeft::unpack(const Block& block, UnpackerCollections* coll) {
      auto res = static_cast<CaloCollections*>(coll)->getEGammas();
      return process(
          block, res, [](const l1t::EGamma& eg) { return eg; }, true, false, false);
    }

    bool CentralJetUnpackerLeft::unpack(const Block& block, UnpackerCollections* coll) {
      auto res = static_cast<CaloCollections*>(coll)->getJets();
      return process(
          block, res, [](const l1t::Jet& j) { return j; }, true, true, false);
    }

    bool ForwardJetUnpackerLeft::unpack(const Block& block, UnpackerCollections* coll) {
      auto res = static_cast<CaloCollections*>(coll)->getJets();
      return process(
          block,
          res,
          [](l1t::Jet j) {
            j.setHwQual(j.hwQual() | 2);
            return j;
          },
          true,
          false,
          false);
    }

    bool TauUnpackerLeft::unpack(const Block& block, UnpackerCollections* coll) {
      auto res = static_cast<CaloCollections*>(coll)->getTaus();
      return process(
          block, res, [](const l1t::Tau& t) { return t; }, true, true, true);
    }

    bool IsoTauUnpackerLeft::unpack(const Block& block, UnpackerCollections* coll) {
      auto res = static_cast<CaloCollections*>(coll)->getIsoTaus();
      return process(
          block, res, [](const l1t::Tau& t) { return t; }, true, true, true);
    }

    bool IsoEGammaUnpackerRight::unpack(const Block& block, UnpackerCollections* coll) {
      auto res = static_cast<CaloCollections*>(coll)->getEGammas();
      return process(
          block,
          res,
          [](l1t::EGamma eg) {
            eg.setHwIso(1);
            return eg;
          },
          false,
          true,
          false);
    }

    bool NonIsoEGammaUnpackerRight::unpack(const Block& block, UnpackerCollections* coll) {
      auto res = static_cast<CaloCollections*>(coll)->getEGammas();
      return process(
          block, res, [](const l1t::EGamma& eg) { return eg; }, false, false, false);
    }

    bool CentralJetUnpackerRight::unpack(const Block& block, UnpackerCollections* coll) {
      auto res = static_cast<CaloCollections*>(coll)->getJets();
      return process(
          block, res, [](const l1t::Jet& j) { return j; }, false, true, false);
    }

    bool ForwardJetUnpackerRight::unpack(const Block& block, UnpackerCollections* coll) {
      auto res = static_cast<CaloCollections*>(coll)->getJets();
      return process(
          block,
          res,
          [](l1t::Jet j) {
            j.setHwQual(j.hwQual() | 2);
            return j;
          },
          false,
          false,
          false);
    }

    bool TauUnpackerRight::unpack(const Block& block, UnpackerCollections* coll) {
      auto res = static_cast<CaloCollections*>(coll)->getTaus();
      return process(
          block, res, [](const l1t::Tau& t) { return t; }, false, true, true);
    }

    bool IsoTauUnpackerRight::unpack(const Block& block, UnpackerCollections* coll) {
      auto res = static_cast<CaloCollections*>(coll)->getIsoTaus();
      return process(
          block, res, [](const l1t::Tau& t) { return t; }, false, true, true);
    }
  }  // namespace stage1
}  // namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage1::IsoEGammaUnpackerLeft);
DEFINE_L1T_UNPACKER(l1t::stage1::NonIsoEGammaUnpackerLeft);
DEFINE_L1T_UNPACKER(l1t::stage1::CentralJetUnpackerLeft);
DEFINE_L1T_UNPACKER(l1t::stage1::ForwardJetUnpackerLeft);
DEFINE_L1T_UNPACKER(l1t::stage1::TauUnpackerLeft);
DEFINE_L1T_UNPACKER(l1t::stage1::IsoTauUnpackerLeft);
DEFINE_L1T_UNPACKER(l1t::stage1::IsoEGammaUnpackerRight);
DEFINE_L1T_UNPACKER(l1t::stage1::NonIsoEGammaUnpackerRight);
DEFINE_L1T_UNPACKER(l1t::stage1::CentralJetUnpackerRight);
DEFINE_L1T_UNPACKER(l1t::stage1::ForwardJetUnpackerRight);
DEFINE_L1T_UNPACKER(l1t::stage1::TauUnpackerRight);
DEFINE_L1T_UNPACKER(l1t::stage1::IsoTauUnpackerRight);
