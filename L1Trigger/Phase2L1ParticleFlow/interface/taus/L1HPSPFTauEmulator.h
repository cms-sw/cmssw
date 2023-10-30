#ifndef L1HPSPFTAUEMULATOR_H
#define L1HPSPFTAUEMULATOR_H

#include "ap_int.h"
#include "ap_fixed.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

namespace L1HPSPFTauEmu {

  //mapping the eta phi onto bits

  static float etaphi_base = 720. / 3.14159;
  static float dz_base = 0.05;
  typedef ap_ufixed<16, 14> pt_t;

  typedef ap_int<12> etaphi_t;

  typedef ap_int<12> detaphi_t;

  typedef ap_uint<20> detaphi2_t;
  typedef ap_uint<5> count_t;  //type for multiplicity
  typedef ap_uint<3> type_t;   //type for particle type
  typedef ap_int<10> dz_t;     //type for z0 cut

  class Particle {
  public:
    pt_t hwPt;
    etaphi_t hwEta;
    etaphi_t hwPhi;
    type_t hwId;
    dz_t hwZ0;

    Particle() : hwPt(0), hwEta(0), hwPhi(0), hwId(0), hwZ0() {}
    Particle(pt_t pt, etaphi_t eta, etaphi_t phi, type_t id, dz_t z0)
        : hwPt(pt), hwEta(eta), hwPhi(phi), hwId(id), hwZ0(z0) {}

    pt_t pt() const { return hwPt; }

    etaphi_t phi() const { return hwPhi; }

    etaphi_t eta() const { return hwEta; }
    type_t id() const { return hwId; }
    dz_t z0() const { return hwZ0; }
  };

  //template classes
  class Tau : public Particle {
  public:
    Tau(pt_t pt, etaphi_t eta, etaphi_t phi, type_t id, dz_t z0) : Particle(pt, eta, phi, id, z0) {}
    ap_uint<5> seed_index;
    pt_t seed_pt;
    etaphi_t seed_eta;
    etaphi_t seed_phi;
    dz_t seed_z0;
  };

  //constants
  static const detaphi_t TWOPI = 3.14159 * 2. * etaphi_base;
  static const detaphi_t PI = 3.14159 * etaphi_base;

  static const detaphi_t strip_phi = 0.20 * etaphi_base;
  static const detaphi_t strip_eta = 0.05 * etaphi_base;

  static const pt_t min_leadChargedPfCand_pt = 4;
  static const detaphi_t isoConeSize = 0.4 * etaphi_base;
  static const detaphi_t delta_Rclean = 0.4 * etaphi_base;
  static const dz_t dzCut = 0.4 / dz_base;
  static const detaphi_t isoConeSizeRound = 92;
  static const etaphi_t etaCutoff = 2.4 * etaphi_base;

  template <int W, int I, ap_q_mode _AP_Q, ap_o_mode _AP_O>
  ap_ufixed<W, I> fp_abs(ap_fixed<W, I, _AP_Q, _AP_O> x) {
    ap_ufixed<W, I> result;
    if (x < 0) {
      result = -x;
    } else {
      result = x;
    }
    return result;
  }

  template <int W>
  inline ap_uint<W> int_abs(ap_int<W> x) {
    ap_uint<W> result;
    if (x < 0) {
      result = -x;
    } else {
      result = x;
    }
    return result;
  }

  template <class inP>
  inline bool is_charged(inP part) {
    bool charge = false;
    if ((part.hwId == 0) || (part.hwId == 1) || (part.hwId == 4)) {
      charge = true;
    } else {
      charge = false;
    }
    return charge;
  }

  inline ap_uint<20> setSConeSize2(pt_t total_pt) {
    ap_uint<20> SignalConeSizeSquare;
    ap_uint<10> SignalConeSize;
    if (total_pt < 115) {
      SignalConeSizeSquare = 23 * 23;
    } else if (total_pt < 120) {
      SignalConeSizeSquare = 22 * 22;
    } else if (total_pt < 126) {
      SignalConeSizeSquare = 21 * 21;
    } else if (total_pt < 132) {
      SignalConeSizeSquare = 20 * 20;
    } else if (total_pt < 139) {
      SignalConeSizeSquare = 19 * 19;
    } else if (total_pt < 147) {
      SignalConeSizeSquare = 18 * 18;
    } else if (total_pt < 156) {
      SignalConeSizeSquare = 17 * 17;
    } else if (total_pt < 166) {
      SignalConeSizeSquare = 16 * 16;
    } else if (total_pt < 178) {
      SignalConeSizeSquare = 15 * 15;
    } else if (total_pt < 191) {
      SignalConeSizeSquare = 14 * 14;
    } else if (total_pt < 206) {
      SignalConeSizeSquare = 13 * 13;
    } else if (total_pt < 224) {
      SignalConeSizeSquare = 12 * 12;
    } else {
      SignalConeSizeSquare = 12 * 12;
    }
    return SignalConeSizeSquare;
  }

  inline bool inIsolationCone(Particle part, Particle seed) {
    bool inCone = false;
    detaphi2_t isoConeSize2 = isoConeSize * isoConeSize;

    if (part.pt() != 0) {
      detaphi_t deltaEta = part.eta() - seed.eta();
      detaphi_t deltaPhi = part.phi() - seed.phi();
      if ((deltaEta * deltaEta + deltaPhi * deltaPhi) < isoConeSize2) {
        inCone = true;
      } else {
        inCone = false;
      }
    } else {
      inCone = false;
    }
    return inCone;
  }

  inline bool inSignalCone(
      Particle part, Particle seed, const int track_count, ap_uint<20> cone2, pt_t& iso_pt, bool& isLead) {
    //finds the signal cone candidates (including strip pt check

    bool isPotentialLead = false;

    isPotentialLead =
        is_charged(part) && part.id() != 4 && part.pt() > min_leadChargedPfCand_pt && int_abs(part.eta()) < etaCutoff;

    //calculate the deta and dphi
    bool inCone = false;

    if (part.pt() != 0) {
      detaphi_t deltaEta = part.eta() - seed.eta();
      detaphi_t deltaPhi = part.phi() - seed.phi();
      detaphi2_t deltaEta2 = deltaEta * deltaEta;
      detaphi2_t deltaPhi2 = deltaPhi * deltaPhi;
      dz_t deltaZ = 0;
      if (part.z0() && seed.z0()) {
        dz_t deltaZ = part.z0() - seed.z0();
      }

      if ((int_abs(deltaEta) < strip_eta) && (int_abs(deltaPhi) < strip_phi) && (part.id() == 3 || (part.id() == 1))) {
        if (isPotentialLead) {
          isLead = true;
        }
        inCone = true;

      } else if (((deltaEta2 + deltaPhi2) < cone2) && !((part.id() == 0) && (track_count > 3)) &&
                 !(is_charged(part) && int_abs(deltaZ) > dzCut)) {
        if (isPotentialLead) {
          isLead = true;
        }
        inCone = true;
      } else {
        if (is_charged(part) && int_abs(deltaZ) > dzCut) {
          iso_pt += part.pt();
          inCone = false;
        }
      }
    }
    return inCone;
  }

  inline Tau makeHPSTauHW(const std::vector<Particle>& parts,
                          const Particle seed,
                          const pt_t total_pt /*, Config config*/) {
    using namespace L1HPSPFTauEmu;

    ap_uint<20> scone2 = setSConeSize2(total_pt);

    pt_t isocone_pt = 0;

    pt_t sum_pt = 0;

    ap_fixed<22, 20> sum_eta = 0;
    ap_fixed<22, 20> sum_phi = 0;

    pt_t tau_pt = 0;
    etaphi_t tau_eta = 0;
    etaphi_t tau_phi = 0;

    pt_t chargedIsoPileup = 0;
    std::vector<Particle> signalParts;
    std::vector<Particle> outsideParts;

    int trct = 0;
    bool leadCand = false;
    bool leadSet = false;
    Particle lead;
    for (std::vector<int>::size_type i = 0; i != parts.size(); i++) {
      bool isSignal = inSignalCone(parts.at(i), seed, trct, scone2, isocone_pt, leadCand);
      if (isSignal) {
        signalParts.push_back(parts.at(i));
        if (parts[i].id() == 0) {
          trct++;
        }
        if (leadCand) {
          if (leadSet == false) {
            lead = parts[i];
            leadCand = false;
            leadSet = true;
          } else {
            if (parts[i].pt() > lead.pt()) {
              lead = parts[i];
              leadCand = false;
            } else {
              leadCand = false;
            }
          }
        }
      } else {
        outsideParts.push_back(parts.at(i));
      }
    }

    for (std::vector<int>::size_type i = 0; i != signalParts.size(); i++) {
      Particle sigP = signalParts.at(i);
      if (is_charged(sigP) || (sigP.id() == 3)) {
        sum_pt += sigP.pt();
        sum_eta += sigP.pt() * sigP.eta();
        sum_phi += sigP.pt() * sigP.phi();
      }
    }

    pt_t div_pt = 1;
    if (sum_pt == 0) {
      div_pt = 1;
    } else {
      div_pt = sum_pt;
    }

    tau_pt = sum_pt;
    tau_eta = sum_eta / div_pt;
    tau_phi = sum_phi / div_pt;

    if (tau_pt > 80. && int_abs(tau_eta) < etaCutoff && leadSet == true && isocone_pt < tau_pt) {
      Tau tau(tau_pt, tau_eta, tau_phi, 0, 0);
      return tau;
    } else {
      Tau tau(0., 0., 0., 0, 0.);
      return tau;
    }
  }

  inline std::vector<Tau> emulateEvent(std::vector<Particle>& parts, std::vector<Particle>& jets, bool jEnable) {
    using namespace L1HPSPFTauEmu;

    std::vector<Particle> parts_copy;
    parts_copy.resize(parts.size());
    std::transform(parts.begin(), parts.end(), parts_copy.begin(), [](const Particle& part) { return part; });
    //sorting by pt
    std::sort(parts_copy.begin(), parts_copy.end(), [](Particle i, Particle j) { return (i.pt() > j.pt()); });

    //sorting jets by pt
    std::vector<Particle> jets_copy;
    jets_copy.resize(jets.size());
    std::transform(jets.begin(), jets.end(), jets_copy.begin(), [](const Particle& jet) { return jet; });
    std::sort(jets_copy.begin(), jets_copy.end(), [](Particle i, Particle j) { return (i.pt() > j.pt()); });

    std::vector<Tau> taus;
    std::vector<Tau> cleaned_taus;
    taus.reserve(20);
    std::vector<Particle> preseed;
    preseed.reserve(144);

    //jet seeds reserve
    //4 for now
    int jets_index = 0;
    int jets_max = jets_copy.size();

    std::vector<Particle> jseeds;
    jseeds.reserve(4);

    int parts_index = 0;
    int parts_max = parts_copy.size();
    //first find the seeds
    while (preseed.size() < 128 && parts_index != parts_max) {
      Particle pSeed = parts_copy.at(parts_index);

      if (pSeed.pt() > 20. && is_charged(pSeed) && int_abs(pSeed.eta()) < etaCutoff) {
        preseed.push_back(pSeed);
      }

      parts_index++;
    }

    std::vector<Particle> seeds;
    seeds.reserve(16);  //up to 16 track + 4 jet seeds right now
    std::vector<int>::size_type pseed_index = 0;
    while (seeds.size() < 16 && pseed_index < preseed.size()) {
      seeds.push_back(preseed.at(pseed_index));
      pseed_index++;
    }

    //With jets
    if (jEnable) {
      while (jseeds.size() < 4 && jets_index != jets_max) {
        Particle jSeed = jets_copy.at(jets_index);

        if (jSeed.pt() > 80. && int_abs(jSeed.eta()) < etaCutoff) {
          jseeds.push_back(jSeed);
        }
        jets_index++;
      }
    }
    for (std::vector<int>::size_type i = 0; i != seeds.size(); i++) {
      Particle seed = seeds.at(i);

      std::vector<Particle> iso_parts;

      iso_parts.reserve(30);
      pt_t total_pt = 0;
      std::vector<int>::size_type iso_index = 0;
      while (iso_index < parts_copy.size() && iso_parts.size() < 30) {
        Particle isoCand = parts_copy.at(iso_index);
        if (inIsolationCone(isoCand, seed)) {
          iso_parts.push_back(isoCand);
          total_pt += isoCand.pt();
        }
        iso_index++;
      }

      taus.push_back(makeHPSTauHW(iso_parts, seed, total_pt));
    }

    //add in the jet taus
    if (jEnable) {
      for (std::vector<int>::size_type i = 0; i != jseeds.size(); i++) {
        Particle jseed = jseeds.at(i);
        std::vector<Particle> iso_parts;
        iso_parts.reserve(30);
        pt_t total_pt = 0;
        std::vector<int>::size_type iso_index = 0;

        pt_t max_pt_j = 0;
        while (iso_index < parts_copy.size()) {
          Particle isoCand = parts_copy.at(iso_index);

          if (inIsolationCone(isoCand, jseed)) {
            if (is_charged(isoCand) && isoCand.pt() > max_pt_j) {
              if (isoCand.z0()) {
                jseed.z0() = isoCand.z0();
              }
              max_pt_j = isoCand.pt();
            }

            if (iso_parts.size() < 30) {
              iso_parts.push_back(isoCand);
              total_pt += isoCand.pt();
            }
          }
          iso_index++;
        }
        taus.push_back(makeHPSTauHW(iso_parts, jseed, total_pt));
      }
    }

    std::sort(taus.begin(), taus.end(), [](Tau i, Tau j) { return (i.pt() > j.pt()); });

    int taus_max = taus.size();

    bool matrix[380];

    for (int i = 0; i < (taus_max - 1); i++) {
      for (int j = i + 1; j < taus_max; j++) {
        etaphi_t deltaE = taus[i].eta() - taus[j].eta();
        etaphi_t deltaP = taus[i].phi() - taus[j].phi();
        if ((deltaE * deltaE + deltaP * deltaP) < (delta_Rclean * delta_Rclean)) {
          matrix[i * 19 + j] = true;
        } else {
          matrix[i * 19 + j] = false;
        }
      }
    }

    if (!taus.empty()) {
      if (taus[0].pt() > 0) {
        cleaned_taus.push_back(taus.at(0));
      }

      bool clean[20];

      for (int i = 0; i < 20; i++) {
        clean[i] = false;
      }
      for (int i = 1; i < taus_max; i++) {
        for (int j = i - 1; j >= 0; j--) {
          clean[i] |= (matrix[j * 15 + i] && !clean[j]);
        }
        if (!clean[i] && taus[i].pt() > 0) {
          cleaned_taus.push_back(taus.at(i));
        }
      }
    }

    return cleaned_taus;
  }

};  // namespace L1HPSPFTauEmu

#endif
