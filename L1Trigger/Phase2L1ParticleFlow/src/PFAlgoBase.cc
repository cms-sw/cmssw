#include "L1Trigger/Phase2L1ParticleFlow/interface/PFAlgoBase.h"

#include "DataFormats/Phase2L1ParticleFlow/interface/PFCandidate.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "Math/ProbFunc.h"
#include <TH1F.h>

namespace { 
    std::vector<float> vd2vf(const std::vector<double> & vd) {
        std::vector<float> ret;
        ret.insert(ret.end(), vd.begin(), vd.end());
        return ret;
    }
}

using namespace l1tpf_impl;

PFAlgoBase::PFAlgoBase( const edm::ParameterSet & iConfig ) :
    etaCharged_(iConfig.getParameter<double>("etaCharged")),
    puppiDr_(iConfig.getParameter<double>("puppiDr")),
    puppiEtaCuts_(vd2vf(iConfig.getParameter<std::vector<double>>("puppiEtaCuts"))),
    puppiPtCuts_(vd2vf(iConfig.getParameter<std::vector<double>>("puppiPtCuts"))),
    puppiPtCutsPhotons_(vd2vf(iConfig.getParameter<std::vector<double>>("puppiPtCutsPhotons"))),
    vtxRes_(iConfig.getParameter<double>("vtxRes")),
    vtxAdaptiveCut_(iConfig.getParameter<bool>("vtxAdaptiveCut"))
{
    if (puppiEtaCuts_.size() != puppiPtCuts_.size() || puppiPtCuts_.size() != puppiPtCutsPhotons_.size()) {
        throw cms::Exception("Configuration", "Bad PUPPI config");
    }
    for (unsigned int i = 0, n = puppiEtaCuts_.size(); i < n; ++i) {
        intPuppiEtaCuts_.push_back( std::round(puppiEtaCuts_[i] * CaloCluster::ETAPHI_SCALE) );
        intPuppiPtCuts_.push_back( std::round(puppiPtCuts_[i] * CaloCluster::PT_SCALE) );
        intPuppiPtCutsPhotons_.push_back( std::round(puppiPtCutsPhotons_[i] * CaloCluster::PT_SCALE) );
    }
}

void PFAlgoBase::initRegion(Region &r) const {
    r.inputSort();
    r.pf.clear(); r.puppi.clear();
    for (auto & c : r.calo) c.used = false;
    for (auto & c : r.emcalo) c.used = false;
    for (auto & t : r.track) { t.used = false; t.muonLink = false; }
}

PFParticle & PFAlgoBase::addTrackToPF(std::vector<PFParticle> &pfs, const PropagatedTrack &tk) const {
    PFParticle pf;
    pf.hwPt = tk.hwPt;
    pf.hwEta = tk.hwEta;
    pf.hwPhi = tk.hwPhi;
    pf.hwVtxEta = tk.hwEta; // FIXME: get from the track
    pf.hwVtxPhi = tk.hwPhi; // before propagation
    pf.track = tk;
    pf.cluster.hwPt = 0;
    pf.cluster.src = nullptr;
    pf.muonsrc = nullptr;
    pf.hwId = (tk.muonLink ? l1t::PFCandidate::Muon : l1t::PFCandidate::ChargedHadron);
    pf.hwStatus = 0;
    pfs.push_back(pf);
    return pfs.back();
}

PFParticle & PFAlgoBase::addCaloToPF(std::vector<PFParticle> &pfs, const CaloCluster &calo) const {
    PFParticle pf;
    pf.hwPt = calo.hwPt;
    pf.hwEta = calo.hwEta;
    pf.hwPhi = calo.hwPhi;
    pf.hwVtxEta = calo.hwEta; 
    pf.hwVtxPhi = calo.hwPhi; 
    pf.track.hwPt = 0;
    pf.track.src = nullptr;
    pf.cluster = calo;
    pf.muonsrc = nullptr;
    pf.hwId = (calo.isEM ? l1t::PFCandidate::Photon : l1t::PFCandidate::NeutralHadron);
    pf.hwStatus = 0;
    pfs.push_back(pf);
    return pfs.back();
}

void PFAlgoBase::runPuppi(Region &r, float npu, float alphaCMed, float alphaCRms, float alphaFMed, float alphaFRms) const {
    computePuppiWeights(r, alphaCMed, alphaCRms, alphaFMed, alphaFRms);
    fillPuppi(r);
}

void PFAlgoBase::runChargedPV(Region &r, float z0) const {
    int16_t iZ0 = round(z0 * InputTrack::Z0_SCALE);
    int16_t iDZ  = round(1.5 * vtxRes_ * InputTrack::Z0_SCALE);
    int16_t iDZ2 = vtxAdaptiveCut_ ? round(4.0 * vtxRes_ * InputTrack::Z0_SCALE) : iDZ;
    for (PFParticle & p : r.pf) {
        bool barrel = std::abs(p.track.hwVtxEta) < InputTrack::VTX_ETA_1p3;
        if (r.relativeCoordinates) barrel = (std::abs(r.globalAbsEta(p.track.floatVtxEta())) < 1.3); // FIXME could make a better integer implementation
        p.chargedPV = (p.hwId <= 1 && std::abs(p.track.hwZ0 - iZ0) < (barrel ? iDZ : iDZ2));
    }
}

void PFAlgoBase::computePuppiWeights(Region &r, float alphaCMed, float alphaCRms, float alphaFMed, float alphaFRms) const {
    int16_t ietacut = std::round(etaCharged_ * CaloCluster::ETAPHI_SCALE);
    // FIXME floats for now
    float puppiDr2 = std::pow(puppiDr_,2);
    for (PFParticle & p : r.pf) {
        // charged
        if (p.hwId <= 1) {
            p.setPuppiW(p.chargedPV ? 1.0 : 0); 
            if (debug_) printf("PUPPI \t charged id %1d pt %7.2f eta %+5.2f phi %+5.2f  alpha %+6.2f x2 %+6.2f --> puppi weight %.3f   puppi pt %7.2f \n", p.hwId, p.floatPt(), p.floatEta(), p.floatPhi(), 0., 0., p.floatPuppiW(), p.floatPt()*p.floatPuppiW());
            continue;
        }
        // neutral
        float alphaC = 0, alphaF = 0;
        for (const PFParticle & p2 : r.pf) {
            float dr2 = ::deltaR2(p.floatEta(), p.floatPhi(), p2.floatEta(), p2.floatPhi());
            if (dr2 > 0 && dr2 < puppiDr2) {
                float w = std::pow(p2.floatPt(),2) / dr2;
                alphaF += w;
                if (p2.chargedPV) alphaC += w;
            }
        }
        float alpha = -99, x2 = -99;
        bool central = std::abs(p.hwEta) < ietacut;
        if (r.relativeCoordinates) central = (std::abs(r.globalAbsEta(p.floatEta())) < etaCharged_); // FIXME could make a better integer implementation
        if (central) {
            if (alphaC > 0) {
                alpha = std::log(alphaC);
                x2 = (alpha - alphaCMed) * std::abs(alpha - alphaCMed) / std::pow(alphaCRms,2);
                p.setPuppiW( ROOT::Math::chisquared_cdf(x2,1) );
            } else {
                p.setPuppiW(0);
            }
        } else {
            if (alphaF > 0) {
                alpha = std::log(alphaF);
                x2 = (alpha - alphaFMed) * std::abs(alpha - alphaFMed) / std::pow(alphaFRms,2);
                p.setPuppiW( ROOT::Math::chisquared_cdf(x2,1) );
            } else {
                p.setPuppiW(0);
            }
        }
        if (debug_) printf("PUPPI \t neutral id %1d pt %7.2f eta %+5.2f phi %+5.2f  alpha %+6.2f x2 %+7.2f --> puppi weight %.3f   puppi pt %7.2f \n", p.hwId, p.floatPt(), p.floatEta(), p.floatPhi(), alpha, x2, p.floatPuppiW(), p.floatPt()*p.floatPuppiW());
    }
}

void PFAlgoBase::doVertexing(std::vector<Region> &rs, VertexAlgo algo, float &pvdz) const {
    int lNBins = int(40./vtxRes_);
    if (algo == TPVtxAlgo) lNBins *= 3;
    std::unique_ptr<TH1F> h_dz(new TH1F("h_dz","h_dz",lNBins,-20,20));
    for (const Region & r : rs) {
        for (const PropagatedTrack & p : r.track) {
            if (rs.size() > 1) {
                if (!r.fiducialLocal(p.floatVtxEta(), p.floatVtxPhi())) continue; // skip duplicates
            }
            h_dz->Fill( p.floatDZ(), std::min(p.floatPt(), 50.f) );
        }
    }
    switch(algo) {
        case OldVtxAlgo: {
                             int imaxbin = h_dz->GetMaximumBin();
                             pvdz = h_dz->GetXaxis()->GetBinCenter(imaxbin);
                         }; break;
        case TPVtxAlgo: {
                            float max = 0; int bmax = -1;
                            for (int b = 1; b <= lNBins; ++b) {
                                float sum3 = h_dz->GetBinContent(b) + h_dz->GetBinContent(b+1) + h_dz->GetBinContent(b-1);
                                if (bmax == -1 || sum3 > max) { max = sum3; bmax = b; }
                            }
                            pvdz = h_dz->GetXaxis()->GetBinCenter(bmax); 
                        }; break;
    }
    int16_t iZ0 = round(pvdz * InputTrack::Z0_SCALE);
    int16_t iDZ  = round(1.5 * vtxRes_ * InputTrack::Z0_SCALE);
    int16_t iDZ2 = vtxAdaptiveCut_ ? round(4.0 * vtxRes_ * InputTrack::Z0_SCALE) : iDZ;
    for (Region & r : rs) {
        for (PropagatedTrack & p : r.track) {
            bool central = std::abs(p.hwVtxEta) < InputTrack::VTX_ETA_1p3;
            if (r.relativeCoordinates) central = (std::abs(r.globalAbsEta(p.floatVtxEta())) < 1.3); // FIXME could make a better integer implementation
            p.fromPV = (std::abs(p.hwZ0 - iZ0) < (central ? iDZ : iDZ2));
        }
    }

}

void PFAlgoBase::computePuppiMedRMS(const std::vector<Region> &rs, float &alphaCMed, float &alphaCRms, float &alphaFMed, float &alphaFRms) const {
    std::vector<float> alphaFs;
    std::vector<float> alphaCs;
    int16_t ietacut = std::round(etaCharged_ * CaloCluster::ETAPHI_SCALE);
    float puppiDr2 = std::pow(puppiDr_,2);
    for (const Region & r : rs) {
        for (const PFParticle & p : r.pf) {
            bool central = std::abs(p.hwEta) < ietacut;
            if (r.relativeCoordinates) central = (r.globalAbsEta(p.floatEta()) < etaCharged_); // FIXME could make a better integer implementation
            if (central) {
                if (p.hwId > 1 || p.chargedPV) continue;
            }
            float alphaC = 0, alphaF = 0;
            for (const PFParticle & p2 : r.pf) {
                float dr2 = ::deltaR2(p.floatEta(), p.floatPhi(), p2.floatEta(), p2.floatPhi());
                if (dr2 > 0 && dr2 < puppiDr2) {
                    float w = std::pow(p2.floatPt(),2) / std::max<float>(0.01f, dr2);
                    alphaF += w;
                    if (p2.chargedPV) alphaC += w;
                }
            }
            if (central) {
                if (alphaC > 0) alphaCs.push_back(std::log(alphaC));
            } else {
                if (alphaF > 0) alphaFs.push_back(std::log(alphaF));
            }
        }
    }
  std::sort(alphaCs.begin(),alphaCs.end());
  std::sort(alphaFs.begin(),alphaFs.end());

  if (alphaCs.size() > 1){
      alphaCMed = alphaCs[alphaCs.size()/2+1];
      double sum = 0.0;
      for (float alpha : alphaCs) sum += std::pow(alpha-alphaCMed,2);
      alphaCRms = std::sqrt(float(sum)/alphaCs.size());
  } else {
      alphaCMed = 8.; alphaCRms = 8.;
  }

  if (alphaFs.size() > 1){
      alphaFMed = alphaFs[alphaFs.size()/2+1];
      double sum = 0.0;
      for (float alpha : alphaFs) sum += std::pow(alpha-alphaFMed,2);
      alphaFRms = std::sqrt(float(sum)/alphaFs.size());
  } else {
      alphaFMed = 6.; alphaFRms = 6.;
  }
  if (debug_) printf("PUPPI \t alphaC = %+6.2f +- %6.2f (%4lu), alphaF = %+6.2f +- %6.2f (%4lu)\n", alphaCMed, alphaCRms, alphaCs.size(), alphaFMed, alphaFRms, alphaFs.size());
}

void PFAlgoBase::fillPuppi(Region &r) const {
    constexpr uint16_t PUPPIW_0p01 = std::round(0.01 * PFParticle::PUPPI_SCALE);
    r.puppi.clear();
    for (PFParticle & p : r.pf) {
        if (p.hwId == l1t::PFCandidate::Muon) {
            r.puppi.push_back(p);
        } else if (p.hwId <= 1) { // charged
            if (p.hwPuppiWeight > 0) {
                r.puppi.push_back(p);
            }
        } else { // neutral
            if (p.hwPuppiWeight > PUPPIW_0p01) {
                // FIXME would work better with PUPPI_SCALE being a power of two, to do the shift
                // FIXME done with floats
                int16_t hwPt = ( float(p.hwPt) * float(p.hwPuppiWeight) / float(PFParticle::PUPPI_SCALE) );
                int16_t hwPtCut = 0, hwAbsEta = r.relativeCoordinates ? round(r.globalAbsEta(p.floatEta()) * CaloCluster::ETAPHI_SCALE) : std::abs(p.hwEta);
                for (unsigned int ietaBin = 0, nBins = intPuppiEtaCuts_.size(); ietaBin < nBins; ++ietaBin) {
                    if (hwAbsEta < intPuppiEtaCuts_[ietaBin]) {
                        hwPtCut = (p.hwId == l1t::PFCandidate::Photon ? intPuppiPtCutsPhotons_[ietaBin] : intPuppiPtCuts_[ietaBin]);
                        break;
                    }
                }
                if (hwPt > hwPtCut) {
                    r.puppi.push_back(p);
                    r.puppi.back().hwPt = hwPt;
                }
            }
        }
    }
}

