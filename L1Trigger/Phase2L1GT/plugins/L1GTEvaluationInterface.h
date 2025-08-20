#ifndef L1Trigger_Phase2L1GT_L1GTEvaluationInterface_h
#define L1Trigger_Phase2L1GT_L1GTEvaluationInterface_h

#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"

#include <ap_int.h>

#include <array>
#include <cstddef>
#include <cstdint>

/**
 *  Source: CMS level-1 trigger interface specification: Global trigger
 **/
namespace l1t {

  template <typename A, typename... Args>
  A l1t_pack_int(const Args&... args) {
    A result = 0;
    std::size_t shift = 0;
    (
        [&result, &shift](const auto& arg) {
          result(shift + arg.width - 1, shift) = arg;
          shift += arg.width;
        }(args),
        ...);

    return result;
  }

  struct L1TGT_BaseInterface {
    virtual std::size_t packed_width() const = 0;
    virtual P2GTCandidate to_GTObject() const = 0;
    virtual ~L1TGT_BaseInterface() {}
  };

  template <std::size_t N>
  struct L1TGT_Interface : public L1TGT_BaseInterface {
    virtual ap_uint<N> pack() const = 0;

    static constexpr std::size_t WIDTH = N;

    std::size_t packed_width() const override { return WIDTH; }
  };

  template <std::size_t N>
  struct L1TGT_Common3Vector : public L1TGT_Interface<N> {
    ap_uint<1> valid;
    ap_uint<16> pT;
    ap_int<13> phi;
    ap_int<14> eta;

    L1TGT_Common3Vector(int valid = 0, int pT = 0, int phi = 0, int eta = 0)
        : valid(valid), pT(pT), phi(phi), eta(eta) {};

    virtual ap_uint<44> pack_common() const { return l1t_pack_int<ap_uint<44>>(valid, pT, phi, eta); }

    ap_uint<N> pack() const override { return pack_common(); }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object;
      gt_object.setHwPT(pT);
      gt_object.setHwPhi(phi);
      gt_object.setHwEta(eta);

      return gt_object;
    }
  };

  struct L1TGT_CommonSum : public L1TGT_Interface<64> {
    ap_uint<1> valid;
    ap_uint<16> pT;
    ap_int<13> phi;
    ap_uint<16> scalarSumPT;

    L1TGT_CommonSum(int valid = 0, int pT = 0, int phi = 0, int scalarSumPT = 0)
        : valid(valid), pT(pT), phi{phi}, scalarSumPT(scalarSumPT) {}

    ap_uint<46> pack_common() const { return l1t_pack_int<ap_uint<46>>(valid, pT, phi, scalarSumPT); }

    ap_uint<64> pack() const override { return pack_common(); }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object;
      gt_object.setHwPT(pT);
      gt_object.setHwPhi(phi);
      gt_object.setHwScalarSumPT(scalarSumPT);

      return gt_object;
    }
  };

  // Global Calorimeter Trigger

  struct L1TGT_GCT_EgammaNonIsolated6p6 : public L1TGT_Common3Vector<64> {
    using L1TGT_Common3Vector::L1TGT_Common3Vector;
  };

  struct L1TGT_GCT_EgammaIsolated6p6 : public L1TGT_Common3Vector<64> {
    using L1TGT_Common3Vector::L1TGT_Common3Vector;
  };

  struct L1TGT_GCT_jet6p6 : public L1TGT_Common3Vector<64> {
    using L1TGT_Common3Vector::L1TGT_Common3Vector;
  };

  struct L1TGT_GCT_tau6p6 : public L1TGT_Common3Vector<64> {
    ap_uint<10> seed_pT;

    L1TGT_GCT_tau6p6(int valid = 0, int pT = 0, int phi = 0, int eta = 0, int seed_pT = 0)
        : L1TGT_Common3Vector(valid, pT, phi, eta), seed_pT(seed_pT) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(L1TGT_Common3Vector::pack_common(), seed_pT);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwSeed_pT(seed_pT);

      return gt_object;
    }
  };

  struct L1TGT_GCT_Sum2 : public L1TGT_CommonSum {
    using L1TGT_CommonSum::L1TGT_CommonSum;
  };

  // Global Muon Trigger

  struct L1TGT_GMT_PromptDisplacedMuon : public L1TGT_Common3Vector<64> {
    ap_int<5> z0;
    ap_int<7> d0;
    ap_uint<1> charge;
    ap_uint<4> qualityScore;

    L1TGT_GMT_PromptDisplacedMuon(int valid = 0,
                                  int pT = 0,
                                  int phi = 0,
                                  int eta = 0,
                                  int z0 = 0,
                                  int d0 = 0,
                                  int charge = 0,
                                  int qualityScore = 0)
        : L1TGT_Common3Vector(valid, pT, phi, eta), z0(z0), d0(d0), charge(charge), qualityScore(qualityScore) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(L1TGT_Common3Vector::pack_common(), z0, d0, charge, qualityScore);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwZ0(static_cast<int>(z0) << 12);
      gt_object.setHwD0(static_cast<int>(d0) << 5);
      gt_object.setHwCharge(charge);
      gt_object.setHwQualityScore(qualityScore);

      return gt_object;
    }
  };

  struct L1TGT_GMT_TrackMatchedmuon : public L1TGT_Common3Vector<96> {
    ap_int<10> z0;
    ap_int<10> d0;
    ap_uint<1> charge;
    ap_uint<6> qualityFlags;
    ap_uint<6> isolationPT;
    ap_uint<4> beta;

    L1TGT_GMT_TrackMatchedmuon(int valid = 0,
                               int pT = 0,
                               int phi = 0,
                               int eta = 0,
                               int z0 = 0,
                               int d0 = 0,
                               int charge = 0,
                               int qualityFlags = 0,
                               int isolationPT = 0,
                               int beta = 0)
        : L1TGT_Common3Vector(valid, pT, phi, eta),
          z0(z0),
          d0(d0),
          charge(charge),
          qualityFlags(qualityFlags),
          isolationPT(isolationPT),
          beta(beta) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(
          L1TGT_Common3Vector::pack_common(), z0, d0, charge, qualityFlags, isolationPT, beta);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwZ0(static_cast<int>(z0) << 7);
      gt_object.setHwD0(static_cast<int>(d0) << 2);
      gt_object.setHwCharge(charge);
      gt_object.setHwQualityFlags(static_cast<int>(qualityFlags));
      gt_object.setHwIsolationPT(static_cast<int>(isolationPT));
      gt_object.setHwBeta(beta);

      return gt_object;
    }
  };

  struct L1TGT_GMT_TopoObject : public L1TGT_Interface<64> {
    ap_uint<1> valid;
    ap_uint<8> pT;  // TODO
    ap_int<8> eta;
    ap_int<8> phi;
    ap_uint<8> mass;
    ap_uint<6> qualityFlags;
    // ap_uint<16> /* Index of 3 prongs */;
    // ap_uint<3> /* Some other quality */;

    L1TGT_GMT_TopoObject(int valid = 0, int pT = 0, int phi = 0, int eta = 0, int mass = 0, int qualityFlags = 0)
        : valid(valid), pT(pT), eta(eta), phi(phi), mass(mass), qualityFlags(qualityFlags) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(valid, pT, eta, phi, mass, qualityFlags);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object;
      gt_object.setHwPT(static_cast<int>(pT) * 5);  // TODO
      gt_object.setHwPhi(static_cast<int>(phi) << 5);
      gt_object.setHwEta(static_cast<int>(eta) << 5);
      gt_object.setHwMass(mass);
      gt_object.setHwQualityFlags(qualityFlags);

      return gt_object;
    }
  };

  // Global Track Trigger

  struct L1TGT_GTT_PromptJet : public L1TGT_Common3Vector<128> {
    ap_int<10> z0;
    ap_uint<5> number_of_tracks;
    ap_uint<4> number_of_displaced_tracks;

    L1TGT_GTT_PromptJet(int valid = 0,
                        int pT = 0,
                        int phi = 0,
                        int eta = 0,
                        int z0 = 0,
                        int number_of_tracks = 0,
                        int number_of_displaced_tracks = 0)
        : L1TGT_Common3Vector(valid, pT, phi, eta),
          z0(z0),
          number_of_tracks(number_of_tracks),
          number_of_displaced_tracks(number_of_displaced_tracks) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(
          L1TGT_Common3Vector::pack_common(), z0, number_of_tracks, number_of_displaced_tracks);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwZ0(static_cast<int>(z0) << 7);
      gt_object.setHwNumber_of_tracks(number_of_tracks);
      gt_object.setHwNumber_of_displaced_tracks(number_of_displaced_tracks);

      return gt_object;
    }
  };

  struct L1TGT_GTT_DisplacedJet : public L1TGT_Common3Vector<128> {
    ap_int<10> z0;
    ap_uint<5> number_of_tracks;
    ap_uint<4> number_of_displaced_tracks;

    L1TGT_GTT_DisplacedJet(int valid = 0,
                           int pT = 0,
                           int phi = 0,
                           int eta = 0,
                           int z0 = 0,
                           int number_of_tracks = 0,
                           int number_of_displaced_tracks = 0)
        : L1TGT_Common3Vector(valid, pT, phi, eta),
          z0(z0),
          number_of_tracks(number_of_tracks),
          number_of_displaced_tracks(number_of_displaced_tracks) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(
          L1TGT_Common3Vector::pack_common(), z0, number_of_tracks, number_of_displaced_tracks);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwZ0(static_cast<int>(z0) << 7);
      gt_object.setHwNumber_of_tracks(number_of_tracks);
      gt_object.setHwNumber_of_displaced_tracks(number_of_displaced_tracks);

      return gt_object;
    }
  };

  struct L1TGT_GTT_Sum : public L1TGT_CommonSum {
    using L1TGT_CommonSum::L1TGT_CommonSum;
  };

  struct L1TGT_GTT_HadronicTau : public L1TGT_Common3Vector<96> {
    ap_uint<10> seed_pT;
    ap_int<10> seed_z0;
    ap_uint<1> charge;
    ap_uint<2> type;

    L1TGT_GTT_HadronicTau(int valid = 0,
                          int pT = 0,
                          int phi = 0,
                          int eta = 0,
                          int seed_pT = 0,
                          int seed_z0 = 0,
                          int charge = 0,
                          int type = 0)
        : L1TGT_Common3Vector(valid, pT, phi, eta), seed_pT(seed_pT), seed_z0(seed_z0), charge(charge), type(type) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(L1TGT_Common3Vector::pack_common(), seed_pT, seed_z0, charge, type);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwSeed_pT(seed_pT);
      gt_object.setHwSeed_z0(seed_z0);
      gt_object.setHwCharge(charge);
      gt_object.setHwType(type);

      return gt_object;
    }
  };

  struct L1TGT_GTT_LightMeson : public L1TGT_Common3Vector<96> {
    ap_int<10> z0;
    //ap_uint<10> /* candidate mass */;
    //ap_uint<2> /* candidate type */;
    //ap_uint<3> /* nbr of tracks */;

    L1TGT_GTT_LightMeson(int valid = 0, int pT = 0, int phi = 0, int eta = 0, int z0 = 0)
        : L1TGT_Common3Vector(valid, pT, phi, eta), z0(z0) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(L1TGT_Common3Vector::pack_common(), z0);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwZ0(static_cast<int>(z0) << 7);

      return gt_object;
    }
  };

  struct L1TGT_GTT_Track : public L1TGT_Interface<96> {
    //TODO

    L1TGT_GTT_Track() {};

    ap_uint<WIDTH> pack() const override { return ap_uint<WIDTH>(0); }

    P2GTCandidate to_GTObject() const override { return P2GTCandidate(); }
  };

  struct L1TGT_GTT_PrimaryVert : public L1TGT_Interface<64> {
    ap_uint<1> valid;
    ap_int<15> z0;
    ap_uint<8> number_of_tracks_in_pv;
    ap_uint<12> sum_pT_pv;
    ap_uint<3> qualityScore;
    ap_uint<10> number_of_tracks_not_in_pv;
    // ap_uint<15> /* unassigned */;

    L1TGT_GTT_PrimaryVert(int valid = 0,
                          int z0 = 0,
                          int number_of_tracks_in_pv = 0,
                          int sum_pT_pv = 0,
                          int qualityScore = 0,
                          int number_of_tracks_not_in_pv = 0)
        : valid(valid),
          z0(z0),
          number_of_tracks_in_pv(number_of_tracks_in_pv),
          sum_pT_pv(sum_pT_pv),
          qualityScore(qualityScore),
          number_of_tracks_not_in_pv(number_of_tracks_not_in_pv) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(
          valid, z0, number_of_tracks_in_pv, sum_pT_pv, qualityScore, number_of_tracks_not_in_pv);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object;
      gt_object.setHwZ0(static_cast<int>(z0) * 5);
      gt_object.setHwNumber_of_tracks_in_pv(number_of_tracks_in_pv);
      gt_object.setHwSum_pT_pv(sum_pT_pv);
      gt_object.setHwQualityScore(qualityScore);
      gt_object.setHwNumber_of_tracks_not_in_pv(number_of_tracks_not_in_pv);

      return gt_object;
    }
  };

  // Correlator Layer-2

  struct L1TGT_CL2_Jet : public L1TGT_Common3Vector<128> {
    ap_int<10> z0;

    L1TGT_CL2_Jet(int valid = 0, int pT = 0, int phi = 0, int eta = 0, int z0 = 0)
        : L1TGT_Common3Vector(valid, pT, phi, eta), z0(z0) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(L1TGT_Common3Vector::pack_common(), z0);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwZ0(static_cast<int>(z0) << 7);

      return gt_object;
    }
  };

  struct L1TGT_CL2_Sum : public L1TGT_CommonSum {
    using L1TGT_CommonSum::L1TGT_CommonSum;
  };

  struct L1TGT_CL2_Tau : public L1TGT_Common3Vector<96> {
    ap_uint<10> seed_pT;
    ap_int<10> seed_z0;
    ap_uint<1> charge;
    ap_uint<2> type;
    //ap_uint<10> /* MVA Id / Isol */;
    //ap_uint<2> /* Id vs Mu */;
    //ap_uint<2> /* Id vs Mu */;

    L1TGT_CL2_Tau(int valid = 0,
                  int pT = 0,
                  int phi = 0,
                  int eta = 0,
                  int seed_pT = 0,
                  int seed_z0 = 0,
                  int charge = 0,
                  int type = 0)
        : L1TGT_Common3Vector(valid, pT, phi, eta), seed_pT(seed_pT), seed_z0(seed_z0), charge(charge), type(type) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(L1TGT_Common3Vector::pack_common(), seed_pT, seed_z0, charge, type);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwSeed_pT(seed_pT);
      gt_object.setHwSeed_z0(seed_z0);
      gt_object.setHwCharge(charge);
      gt_object.setHwType(type);

      return gt_object;
    }
  };

  struct L1TGT_CL2_Electron : public L1TGT_Common3Vector<96> {
    ap_uint<4> qualityFlags;
    ap_uint<11> isolationPT;
    ap_uint<1> charge;
    ap_int<10> z0;

    L1TGT_CL2_Electron(int valid = 0,
                       int pT = 0,
                       int phi = 0,
                       int eta = 0,
                       int qualityFlags = 0,
                       int isolationPT = 0,
                       int charge = 0,
                       int z0 = 0)
        : L1TGT_Common3Vector(valid, pT, phi, eta),
          qualityFlags(qualityFlags),
          isolationPT(isolationPT),
          charge(charge),
          z0(z0) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(L1TGT_Common3Vector::pack_common(), qualityFlags, isolationPT, charge, z0);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwQualityFlags(qualityFlags);
      gt_object.setHwIsolationPT(isolationPT);
      gt_object.setHwCharge(charge);
      gt_object.setHwZ0(static_cast<int>(z0) << 7);

      return gt_object;
    }
  };

  struct L1TGT_CL2_Photon : public L1TGT_Common3Vector<96> {
    ap_uint<4> qualityFlags;
    ap_uint<11> isolationPT;

    L1TGT_CL2_Photon(int valid = 0, int pT = 0, int phi = 0, int eta = 0, int qualityFlags = 0, int isolationPT = 0)
        : L1TGT_Common3Vector(valid, pT, phi, eta), qualityFlags(qualityFlags), isolationPT(isolationPT) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(L1TGT_Common3Vector::pack_common(), qualityFlags, isolationPT);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwQualityFlags(qualityFlags);
      gt_object.setHwIsolationPT(isolationPT);

      return gt_object;
    }
  };
}  // namespace l1t

#endif  // L1Trigger_Phase2L1GT_L1GTEvaluationInterface_h
