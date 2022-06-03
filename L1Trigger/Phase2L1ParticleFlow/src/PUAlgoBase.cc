#include "L1Trigger/Phase2L1ParticleFlow/interface/PUAlgoBase.h"

#include <TH1F.h>

using namespace l1tpf_impl;

PUAlgoBase::PUAlgoBase(const edm::ParameterSet &iConfig)
    : debug_(iConfig.getUntrackedParameter<int>("debug", 0)),
      etaCharged_(iConfig.getParameter<double>("etaCharged")),
      vtxRes_(iConfig.getParameter<double>("vtxRes")),
      vtxAdaptiveCut_(iConfig.getParameter<bool>("vtxAdaptiveCut")),
      nVtx_(iConfig.getParameter<int>("nVtx")) {}

PUAlgoBase::~PUAlgoBase() {}

void PUAlgoBase::runChargedPV(Region &r, float z0) const {
  int16_t iZ0 = round(z0 * InputTrack::Z0_SCALE);
  int16_t iDZ = round(1.5 * vtxRes_ * InputTrack::Z0_SCALE);
  int16_t iDZ2 = vtxAdaptiveCut_ ? round(4.0 * vtxRes_ * InputTrack::Z0_SCALE) : iDZ;
  for (PFParticle &p : r.pf) {
    bool barrel = std::abs(p.track.hwVtxEta) < InputTrack::VTX_ETA_1p3;
    if (r.relativeCoordinates)
      barrel =
          (std::abs(r.globalAbsEta(p.track.floatVtxEta())) < 1.3);  // FIXME could make a better integer implementation
    p.chargedPV = (p.hwId <= 1 && std::abs(p.track.hwZ0 - iZ0) < (barrel ? iDZ : iDZ2));
  }
}

void PUAlgoBase::runChargedPV(Region &r, std::vector<float> &z0s) const {
  int16_t iDZ = round(1.5 * vtxRes_ * InputTrack::Z0_SCALE);
  int16_t iDZ2 = vtxAdaptiveCut_ ? round(4.0 * vtxRes_ * InputTrack::Z0_SCALE) : iDZ;
  for (PFParticle &p : r.pf) {
    bool barrel = std::abs(p.track.hwVtxEta) < InputTrack::VTX_ETA_1p3;
    if (r.relativeCoordinates)
      barrel =
          (std::abs(r.globalAbsEta(p.track.floatVtxEta())) < 1.3);  // FIXME could make a better integer implementation
    //p.chargedPV = (p.hwId <= 1 && std::abs(p.track.hwZ0 - iZ0) < (barrel ? iDZ : iDZ2));
    bool pFromPV = false;
    for (int v = 0; v < nVtx_; v++) {
      int16_t iZ0 = round(z0s[v] * InputTrack::Z0_SCALE);
      if (std::abs(p.track.hwZ0 - iZ0) < (barrel ? iDZ : iDZ2))
        pFromPV = true;
    }
    p.chargedPV = pFromPV;
  }
}

void PUAlgoBase::doVertexing(std::vector<Region> &rs, VertexAlgo algo, float &pvdz) const {
  int lNBins = int(40. / vtxRes_);
  if (algo == VertexAlgo::TP)
    lNBins *= 3;
  std::unique_ptr<TH1F> h_dz(new TH1F("h_dz", "h_dz", lNBins, -20, 20));
  if (algo != VertexAlgo::External) {
    for (const Region &r : rs) {
      for (const PropagatedTrack &p : r.track) {
        if (rs.size() > 1) {
          if (!r.fiducialLocal(p.floatVtxEta(), p.floatVtxPhi()))
            continue;  // skip duplicates
        }
        h_dz->Fill(p.floatDZ(), std::min(p.floatPt(), 50.f));
      }
    }
  }
  switch (algo) {
    case VertexAlgo::External:
      break;
    case VertexAlgo::Old: {
      int imaxbin = h_dz->GetMaximumBin();
      pvdz = h_dz->GetXaxis()->GetBinCenter(imaxbin);
    }; break;
    case VertexAlgo::TP: {
      float max = 0;
      int bmax = -1;
      for (int b = 1; b <= lNBins; ++b) {
        float sum3 = h_dz->GetBinContent(b) + h_dz->GetBinContent(b + 1) + h_dz->GetBinContent(b - 1);
        if (bmax == -1 || sum3 > max) {
          max = sum3;
          bmax = b;
        }
      }
      pvdz = h_dz->GetXaxis()->GetBinCenter(bmax);
    }; break;
  }
  int16_t iZ0 = round(pvdz * InputTrack::Z0_SCALE);
  int16_t iDZ = round(1.5 * vtxRes_ * InputTrack::Z0_SCALE);
  int16_t iDZ2 = vtxAdaptiveCut_ ? round(4.0 * vtxRes_ * InputTrack::Z0_SCALE) : iDZ;
  for (Region &r : rs) {
    for (PropagatedTrack &p : r.track) {
      bool central = std::abs(p.hwVtxEta) < InputTrack::VTX_ETA_1p3;
      if (r.relativeCoordinates)
        central =
            (std::abs(r.globalAbsEta(p.floatVtxEta())) < 1.3);  // FIXME could make a better integer implementation
      p.fromPV = (std::abs(p.hwZ0 - iZ0) < (central ? iDZ : iDZ2));
    }
  }
}

void PUAlgoBase::doVertexings(std::vector<Region> &rs, VertexAlgo algo, std::vector<float> &pvdz) const {
  int lNBins = int(40. / vtxRes_);
  if (algo == VertexAlgo::TP || algo == VertexAlgo::External)
    lNBins *= 3;
  std::unique_ptr<TH1F> h_dz(new TH1F("h_dz", "h_dz", lNBins, -20, 20));
  for (const Region &r : rs) {
    for (const PropagatedTrack &p : r.track) {
      if (rs.size() > 1) {
        if (!r.fiducialLocal(p.floatVtxEta(), p.floatVtxPhi()))
          continue;  // skip duplicates
      }
      h_dz->Fill(p.floatDZ(), std::min(p.floatPt(), 50.f));
    }
  }
  switch (algo) {
    case VertexAlgo::External: {
      int lBin[nVtx_];
      for (int vtx = 0; vtx < int(nVtx_); vtx++)
        lBin[vtx] = -1;
      for (int vtx = 0; vtx < int(nVtx_) - int(pvdz.size()); vtx++) {
        float max = 0;
        for (int b = 1; b <= lNBins; ++b) {
          bool pPass = false;
          for (int v = 0; v < vtx; v++) {
            if (lBin[v] == b)
              pPass = true;
          }
          if (pPass)
            continue;
          float sum3 = h_dz->GetBinContent(b) + h_dz->GetBinContent(b + 1) + h_dz->GetBinContent(b - 1);
          if (lBin[vtx] == -1 || sum3 > max) {
            max = sum3;
            lBin[vtx] = b;
          }
        }
        float tmpdz = h_dz->GetXaxis()->GetBinCenter(lBin[vtx]);
        pvdz.push_back(tmpdz);
      }
    } break;
    case VertexAlgo::Old: {
      int lBin[nVtx_];
      for (int vtx = 0; vtx < nVtx_; ++vtx) {
        float pMax = 0;
        lBin[vtx] = -1;
        for (int b = 1; b <= lNBins; ++b) {
          bool pPass = false;
          for (int v = 0; v < vtx; v++) {
            if (lBin[v] == b)
              pPass = true;
          }
          if (pPass)
            continue;
          float pVal = h_dz->GetBinContent(b);
          if (pMax < pVal || lBin[vtx] == -1) {
            pMax = pVal;
            lBin[vtx] = b;
          }
        }
        float tmpdz = h_dz->GetXaxis()->GetBinCenter(lBin[vtx]);
        pvdz.push_back(tmpdz);
      }
    }; break;
    case VertexAlgo::TP: {
      int lBin[nVtx_];
      for (int vtx = 0; vtx < int(nVtx_); vtx++)
        lBin[vtx] = -1;
      for (int vtx = 0; vtx < nVtx_; vtx++) {
        float max = 0;
        for (int b = 1; b <= lNBins; ++b) {
          bool pPass = false;
          for (int v = 0; v < vtx; v++) {
            if (lBin[v] == b)
              pPass = true;
          }
          if (pPass)
            continue;
          float sum3 = h_dz->GetBinContent(b) + h_dz->GetBinContent(b + 1) + h_dz->GetBinContent(b - 1);
          if (lBin[vtx] == -1 || sum3 > max) {
            max = sum3;
            lBin[vtx] = b;
          }
        }
        float tmpdz = h_dz->GetXaxis()->GetBinCenter(lBin[vtx]);
        pvdz.push_back(tmpdz);
      }
    }; break;
  }

  int16_t iDZ = round(1.5 * vtxRes_ * InputTrack::Z0_SCALE);
  int16_t iDZ2 = vtxAdaptiveCut_ ? round(4.0 * vtxRes_ * InputTrack::Z0_SCALE) : iDZ;
  for (Region &r : rs) {
    for (PropagatedTrack &p : r.track) {
      bool central = std::abs(p.hwVtxEta) < InputTrack::VTX_ETA_1p3;
      if (r.relativeCoordinates)
        central =
            (std::abs(r.globalAbsEta(p.floatVtxEta())) < 1.3);  // FIXME could make a better integer implementation
      bool pFromPV = false;
      for (int v = 0; v < nVtx_; v++) {
        int16_t iZ0 = round(pvdz[v] * InputTrack::Z0_SCALE);
        if (std::abs(p.hwZ0 - iZ0) < (central ? iDZ : iDZ2))
          pFromPV = true;
      }
      p.fromPV = pFromPV;
    }
  }
}

const std::vector<std::string> &PUAlgoBase::puGlobalNames() const {
  static const std::vector<std::string> empty_;
  return empty_;
}
