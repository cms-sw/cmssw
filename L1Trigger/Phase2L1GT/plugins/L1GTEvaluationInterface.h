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

  template <typename A, typename... Args>
  A l1t_unpack_int(const A& packed, Args&&... args) {
    A temp = packed;
    (
        [&temp](auto&& arg) {
          arg = temp(arg.width - 1, 0);
          temp >>= arg.width;
        }(std::forward<Args>(args)),
        ...);
    return temp;
  }

  struct L1TGT_BaseInterface {
    virtual std::size_t packed_width() const = 0;
    virtual P2GTCandidate to_GTObject() const = 0;
    virtual ~L1TGT_BaseInterface() {}
  };

  template <std::size_t N>
  struct L1TGT_Interface : public L1TGT_BaseInterface {
    virtual ap_uint<N> pack() const = 0;
    virtual ap_uint<N> unpack(const ap_uint<N>&) = 0;

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
        : valid(valid), pT(pT), phi(phi), eta(eta){};

    virtual ap_uint<44> pack_common() const { return l1t_pack_int<ap_uint<44>>(valid, pT, phi, eta); }

    ap_uint<N> pack() const override { return pack_common(); }

    ap_uint<N> unpack(const ap_uint<N>& packed) override { return l1t_unpack_int(packed, valid, pT, phi, eta); }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object;
      gt_object.setHwPT(pT);
      gt_object.setHwPhi(phi);
      gt_object.setHwEta(eta);

      return gt_object;
    }
  };

  template <std::size_t N>
  struct L1TGT_CommonSum : public L1TGT_Interface<N> {
    ap_uint<1> valid;
    ap_uint<16> pT;
    ap_int<13> phi;
    ap_uint<16> scalar_sum_pT;

    L1TGT_CommonSum(int valid = 0, int pT = 0, int phi = 0, int scalar_sum_pT = 0)
        : valid(valid), pT(pT), phi{phi}, scalar_sum_pT(scalar_sum_pT) {}

    ap_uint<46> pack_common() const { return l1t_pack_int<ap_uint<46>>(valid, pT, phi, scalar_sum_pT); }

    ap_uint<N> pack() const override { return pack_common(); }

    ap_uint<N> unpack(const ap_uint<N>& packed) override {
      return l1t_unpack_int(packed, valid, pT, phi, scalar_sum_pT);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object;
      gt_object.setHwPT(pT);
      gt_object.setHwPhi(phi);
      gt_object.setHwSca_sum(scalar_sum_pT);

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

    ap_uint<WIDTH> unpack(const ap_uint<WIDTH>& packed) override {
      return l1t_unpack_int(L1TGT_Common3Vector::unpack(packed), seed_pT);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwSeed_pT(seed_pT);

      return gt_object;
    }
  };

  struct L1TGT_GCT_Sum2 : public L1TGT_CommonSum<64> {
    using L1TGT_CommonSum::L1TGT_CommonSum;
  };

  // Global Muon Trigger

  struct L1TGT_GMT_PromptDisplacedMuon : public L1TGT_Common3Vector<64> {
    ap_uint<5> z0;
    ap_int<7> d0;
    ap_uint<1> charge;
    ap_uint<4> qual;

    L1TGT_GMT_PromptDisplacedMuon(
        int valid = 0, int pT = 0, int phi = 0, int eta = 0, int z0 = 0, int d0 = 0, int charge = 0, int qual = 0)
        : L1TGT_Common3Vector(valid, pT, phi, eta), z0(z0), d0(d0), charge(charge), qual(qual) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(L1TGT_Common3Vector::pack_common(), z0, d0, charge, qual);
    }

    ap_uint<WIDTH> unpack(const ap_uint<WIDTH>& packed) override {
      return l1t_unpack_int(L1TGT_Common3Vector::unpack(packed), z0, d0, charge, qual);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwZ0(static_cast<int>(z0) << 5);
      gt_object.setHwD0(static_cast<int>(d0) << 5);
      gt_object.setHwCharge(charge);
      gt_object.setHwQual(qual);

      return gt_object;
    }
  };

  struct L1TGT_GMT_TrackMatchedmuon : public L1TGT_Common3Vector<96> {
    ap_int<10> z0;
    ap_int<10> d0;
    ap_uint<1> charge;
    ap_uint<8> qual;
    ap_uint<4> iso;
    ap_uint<4> beta;

    L1TGT_GMT_TrackMatchedmuon(int valid = 0,
                               int pT = 0,
                               int phi = 0,
                               int eta = 0,
                               int z0 = 0,
                               int d0 = 0,
                               int charge = 0,
                               int qual = 0,
                               int iso = 0,
                               int beta = 0)
        : L1TGT_Common3Vector(valid, pT, phi, eta), z0(z0), d0(d0), charge(charge), qual(qual), iso(iso), beta(beta) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(L1TGT_Common3Vector::pack_common(), z0, d0, charge, qual, iso, beta);
    }

    ap_uint<WIDTH> unpack(const ap_uint<WIDTH>& packed) override {
      return l1t_unpack_int(L1TGT_Common3Vector::unpack(packed), z0, d0, charge, qual, iso, beta);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwZ0(z0);
      gt_object.setHwD0(static_cast<int>(d0) << 2);
      gt_object.setHwCharge(charge);
      gt_object.setHwQual(qual);
      gt_object.setHwIso(static_cast<int>(iso) << 7);
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
    ap_uint<6> qual;
    // ap_uint<16> /* Index of 3 prongs */;
    // ap_uint<3> /* Some other quality */;

    L1TGT_GMT_TopoObject(int valid = 0, int pT = 0, int phi = 0, int eta = 0, int mass = 0, int qual = 0)
        : valid(valid), pT(pT), eta(eta), phi(phi), mass(mass), qual(qual) {}

    ap_uint<WIDTH> pack() const override { return l1t_pack_int<ap_uint<WIDTH>>(valid, pT, eta, phi, mass, qual); }

    ap_uint<WIDTH> unpack(const ap_uint<WIDTH>& packed) override {
      return l1t_unpack_int(packed, valid, pT, eta, phi, mass, qual);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object;
      gt_object.setHwPT(static_cast<int>(pT) * 5);  // TODO
      gt_object.setHwPhi(static_cast<int>(phi) << 5);
      gt_object.setHwEta(static_cast<int>(eta) << 5);
      gt_object.setHwMass(mass);
      gt_object.setHwQual(qual);

      return gt_object;
    }
  };

  // Global Track Trigger

  struct L1TGT_GTT_PromptJet : public L1TGT_Common3Vector<128> {
    ap_int<10> z0;
    ap_uint<5> number_of_tracks;
    // ap_uint<5> /* unassigned */;

    L1TGT_GTT_PromptJet(int valid = 0, int pT = 0, int phi = 0, int eta = 0, int z0 = 0, int number_of_tracks = 0)
        : L1TGT_Common3Vector(valid, pT, phi, eta), z0(z0), number_of_tracks(number_of_tracks) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(L1TGT_Common3Vector::pack_common(), z0, number_of_tracks);
    }

    ap_uint<WIDTH> unpack(const ap_uint<WIDTH>& packed) override {
      return l1t_unpack_int<ap_uint<WIDTH>>(L1TGT_Common3Vector::unpack(packed), z0, number_of_tracks);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwZ0(z0);
      gt_object.setHwNumber_of_tracks(number_of_tracks);

      return gt_object;
    }
  };

  struct L1TGT_GTT_DisplacedJet : public L1TGT_Common3Vector<128> {
    ap_int<10> z0;
    ap_uint<5> number_of_tracks;
    // ap_uint<5> /* unassigned */;
    ap_int<12> d0;

    L1TGT_GTT_DisplacedJet(
        int valid = 0, int pT = 0, int phi = 0, int eta = 0, int z0 = 0, int number_of_tracks = 0, int d0 = 0)
        : L1TGT_Common3Vector(valid, pT, phi, eta), z0(z0), number_of_tracks(number_of_tracks), d0(d0) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(L1TGT_Common3Vector::pack_common(), z0, number_of_tracks, ap_uint<5>(0), d0);
    }

    ap_uint<WIDTH> unpack(const ap_uint<WIDTH>& packed) override {
      return l1t_unpack_int<ap_uint<WIDTH>>(
          L1TGT_Common3Vector::unpack(packed), z0, number_of_tracks, ap_uint<5>(0), d0);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwZ0(z0);
      gt_object.setHwNumber_of_tracks(number_of_tracks);
      gt_object.setHwD0(d0);

      return gt_object;
    }
  };

  struct L1TGT_GTT_Sum : public L1TGT_CommonSum<64> {
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

    ap_uint<WIDTH> unpack(const ap_uint<WIDTH>& packed) override {
      return l1t_unpack_int(L1TGT_Common3Vector::unpack(packed), seed_pT, seed_z0, charge, type);
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

    ap_uint<WIDTH> unpack(const ap_uint<WIDTH>& packed) override {
      return l1t_unpack_int(L1TGT_Common3Vector::unpack(packed), z0);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwZ0(z0);

      return gt_object;
    }
  };

  struct L1TGT_GTT_PrimaryVert : public L1TGT_Interface<64> {
    ap_uint<1> valid;
    ap_int<15> z0;
    ap_uint<8> number_of_tracks_in_pv;
    ap_uint<12> sum_pT_pv;
    ap_uint<3> qual;
    ap_uint<10> number_of_tracks_not_in_pv;
    // ap_uint<15> /* unassigned */;

    L1TGT_GTT_PrimaryVert(int valid = 0,
                          int z0 = 0,
                          int number_of_tracks_in_pv = 0,
                          int sum_pT_pv = 0,
                          int qual = 0,
                          int number_of_tracks_not_in_pv = 0)
        : valid(valid),
          z0(z0),
          number_of_tracks_in_pv(number_of_tracks_in_pv),
          sum_pT_pv(sum_pT_pv),
          qual(qual),
          number_of_tracks_not_in_pv(number_of_tracks_not_in_pv) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(
          valid, z0 /*, number_of_tracks_in_pv, sum_pT_pv, qual, number_of_tracks_not_in_pv */);  // TODO: Maybe later
    }

    ap_uint<WIDTH> unpack(const ap_uint<WIDTH>& packed) override {
      return l1t_unpack_int(packed, valid, z0, number_of_tracks_in_pv, sum_pT_pv, qual, number_of_tracks_not_in_pv);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object;
      gt_object.setHwZ0(z0);
      gt_object.setHwNumber_of_tracks_in_pv(number_of_tracks_in_pv);
      gt_object.setHwSum_pT_pv(sum_pT_pv);
      gt_object.setHwQual(qual);
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

    ap_uint<WIDTH> unpack(const ap_uint<WIDTH>& packed) override {
      return l1t_unpack_int(L1TGT_Common3Vector::unpack(packed), z0);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwZ0(z0);

      return gt_object;
    }
  };

  struct L1TGT_CL2_Sum : public L1TGT_CommonSum<64> {
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

    ap_uint<WIDTH> unpack(const ap_uint<WIDTH>& packed) override {
      return l1t_unpack_int(L1TGT_Common3Vector::unpack(packed), seed_pT, seed_z0, charge, type);
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
    ap_uint<4> qual;
    ap_uint<11> iso;
    ap_uint<1> charge;
    ap_int<10> z0;

    L1TGT_CL2_Electron(
        int valid = 0, int pT = 0, int phi = 0, int eta = 0, int qual = 0, int iso = 0, int charge = 0, int z0 = 0)
        : L1TGT_Common3Vector(valid, pT, phi, eta), qual(qual), iso(iso), charge(charge), z0(z0) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(L1TGT_Common3Vector::pack_common(), qual, iso, charge, z0);
    }

    ap_uint<WIDTH> unpack(const ap_uint<WIDTH>& packed) override {
      return l1t_unpack_int(L1TGT_Common3Vector::unpack(packed), qual, iso, charge, z0);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwQual(qual);
      gt_object.setHwIso(iso);
      gt_object.setHwCharge(charge);
      gt_object.setHwZ0(z0);

      return gt_object;
    }
  };

  struct L1TGT_CL2_Photon : public L1TGT_Common3Vector<96> {
    ap_uint<4> qual;
    ap_uint<11> iso;

    L1TGT_CL2_Photon(int valid = 0, int pT = 0, int phi = 0, int eta = 0, int qual = 0, int iso = 0)
        : L1TGT_Common3Vector(valid, pT, phi, eta), qual(qual), iso(iso) {}

    ap_uint<WIDTH> pack() const override {
      return l1t_pack_int<ap_uint<WIDTH>>(L1TGT_Common3Vector::pack_common(), qual, iso);
    }

    ap_uint<WIDTH> unpack(const ap_uint<WIDTH>& packed) override {
      return l1t_unpack_int(L1TGT_Common3Vector::unpack(packed), qual, iso);
    }

    P2GTCandidate to_GTObject() const override {
      P2GTCandidate gt_object(L1TGT_Common3Vector::to_GTObject());
      gt_object.setHwQual(qual);
      gt_object.setHwIso(iso);

      return gt_object;
    }
  };
}  // namespace l1t

#endif  // L1Trigger_Phase2L1GT_L1GTEvaluationInterface_h
