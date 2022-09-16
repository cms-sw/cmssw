#ifndef PHASE2GMT_TEMPORARY_ASSOCIATOR
#define PHASE2GMT_TEMPORARY_ASSOCIATOR

#include "ap_int.h"
#include "MuonROI.h"
#include "DataFormats/L1TMuonPhase2/interface/MuonStub.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace Phase2L1GMT {

  class ROITempAssociator {
  public:
    ROITempAssociator(const edm::ParameterSet& iConfig) {}
    ~ROITempAssociator() {}

    std::vector<MuonROI> associate(int bx,
                                   const l1t::ObjectRefBxCollection<l1t::RegionalMuonCand>& muons,
                                   const l1t::MuonStubRefVector& stubs) {
      std::vector<MuonROI> out;
      l1t::MuonStubRefVector usedStubs;

      if (muons.size() > 0) {
        for (unsigned int i = 0; i < muons.size(bx); ++i) {
          const l1t::RegionalMuonCandRef& mu = muons.at(bx, i);
          uint pt = mu->hwPt();
          uint charge = mu->hwSign();

          float eta = mu->hwEta() * 0.010875;

          int globalPhi = 0;
          if (mu->trackFinderType() == l1t::bmtf) {
            globalPhi = mu->processor() * 48 + mu->hwPhi() - 24;
          } else {
            globalPhi = mu->processor() * 96 + mu->hwPhi() + 24;
          }

          float phi = globalPhi * 2 * M_PI / 576.0;
          if (phi > (M_PI))
            phi = phi - 2 * M_PI;
          else if (phi < (-M_PI))
            phi = phi + 2 * M_PI;

          MuonROI roi(bx, charge, pt, 1);
          roi.setMuonRef(mu);
          l1t::MuonStubRefVector cleanedStubs = clean(stubs, usedStubs);

          for (unsigned int layer = 0; layer <= 4; ++layer) {
            l1t::MuonStubRefVector selectedStubs = selectLayerBX(cleanedStubs, bx, layer);
            int bestStubINT = -1;
            float dPhi = 1000.0;

            for (uint i = 0; i < selectedStubs.size(); ++i) {
              const l1t::MuonStubRef& stub = selectedStubs[i];
              float deltaPhi = (stub->quality() & 0x1) ? stub->offline_coord1() - phi : stub->offline_coord2() - phi;
              if (deltaPhi > M_PI)
                deltaPhi = deltaPhi - 2 * M_PI;
              else if (deltaPhi < -M_PI)
                deltaPhi = deltaPhi + 2 * M_PI;
              deltaPhi = fabs(deltaPhi);
              float deltaEta = (stub->etaQuality() == 0 || (stub->etaQuality() & 0x1))
                                   ? fabs(stub->offline_eta1() - eta)
                                   : fabs(stub->offline_eta2() - eta);
              if (deltaPhi < (M_PI / 6.0) && deltaEta < 0.3 && deltaPhi < dPhi) {
                dPhi = deltaPhi;
                bestStubINT = i;
              }
            }
            if (bestStubINT >= 0) {
              roi.addStub(selectedStubs[bestStubINT]);
              usedStubs.push_back(selectedStubs[bestStubINT]);
            }
          }
          if (out.size() < 16 && !roi.stubs().empty())
            out.push_back(roi);
        }
      }
      //Now the stubs only . Find per layer

      l1t::MuonStubRefVector cleanedStubs = clean(stubs, usedStubs);

      while (!cleanedStubs.empty()) {
        MuonROI roi(bx, 0, 0, 0);
        roi.addStub(cleanedStubs[0]);
        usedStubs.push_back(cleanedStubs[0]);
        for (unsigned int layer = 0; layer <= 4; ++layer) {
          if (layer == cleanedStubs[0]->tfLayer())
            continue;
          l1t::MuonStubRefVector selectedStubs = selectLayerBX(cleanedStubs, bx, layer);
          if (!selectedStubs.empty()) {
            roi.addStub(selectedStubs[0]);
            usedStubs.push_back(selectedStubs[0]);
          }
        }
        if (!roi.stubs().empty())
          if (out.size() < 16)
            out.push_back(roi);
        cleanedStubs = clean(cleanedStubs, usedStubs);
      }
      return out;
    }

  private:
    l1t::MuonStubRefVector selectLayerBX(const l1t::MuonStubRefVector& all, int bx, uint layer) {
      l1t::MuonStubRefVector out;
      for (const auto& stub : all) {
        if (stub->bxNum() == bx && stub->tfLayer() == layer)
          out.push_back(stub);
      }
      return out;
    }

    l1t::MuonStubRefVector clean(const l1t::MuonStubRefVector& all, const l1t::MuonStubRefVector& used) {
      l1t::MuonStubRefVector out;
      for (const auto& stub : all) {
        bool keep = true;
        for (const auto& st : used) {
          if (st == stub) {
            keep = false;
            break;
          }
        }
        if (keep)
          out.push_back(stub);
      }
      return out;
    }
  };
}  // namespace Phase2L1GMT

#endif
