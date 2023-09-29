// ===========================================================================
//
//       Filename:  Isolation.h
//
//    Description:
//
//        Version:  1.0
//        Created:  02/23/2021 01:16:43 PM
//       Revision:  none
//       Compiler:  g++
//
//         Author:  Zhenbin Wu, zhenbin.wu@gmail.com
//
// ===========================================================================

#ifndef PHASE2GMT_ISOLATION
#define PHASE2GMT_ISOLATION

#include "TopologicalAlgorithm.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <atomic>

namespace Phase2L1GMT {
  class Isolation : public TopoAlgo {
  public:
    Isolation(const edm::ParameterSet &iConfig);
    ~Isolation();
    Isolation(const Isolation &cpy);

    unsigned compute_trk_iso(l1t::TrackerMuon &in_mu, ConvertedTTTrack &in_trk);

    void isolation_allmu_alltrk(std::vector<l1t::TrackerMuon> &trkMus, std::vector<ConvertedTTTrack> &convertedTracks);

  private:
    void DumpOutputs(std::vector<l1t::TrackerMuon> &trkMus);
    int SetAbsIsolationBits(int accum);
    int SetRelIsolationBits(int accum, int mupt);
    int OverlapRemoval(unsigned &ovrl, std::vector<unsigned> &overlaps);

    const static int c_iso_dangle_max = 260;  //@ <  260 x 2pi/2^13 = 0.2 rad
    const static int c_iso_dz_max = 17;       //@ <  17 x 60/2^10 = 1   cm
    const static int c_iso_pt_min = 120;      //@ >= , 120  x 25MeV = 3 GeV

    // Assuming 4 bits for Muon isolation
    int absiso_thrL;
    int absiso_thrM;
    int absiso_thrT;
    double reliso_thrL;
    double reliso_thrM;
    double reliso_thrT;
    bool verbose_;
    bool dumpForHLS_;
    std::ofstream dumpOutput;

    typedef ap_ufixed<9, 9, AP_TRN, AP_SAT> iso_accum_t;
    typedef ap_ufixed<9, 0> reliso_thresh_t;
  };

  inline Isolation::Isolation(const edm::ParameterSet &iConfig)
      : absiso_thrL(iConfig.getParameter<int>("AbsIsoThresholdL")),
        absiso_thrM(iConfig.getParameter<int>("AbsIsoThresholdM")),
        absiso_thrT(iConfig.getParameter<int>("AbsIsoThresholdT")),
        reliso_thrL(iConfig.getParameter<double>("RelIsoThresholdL")),
        reliso_thrM(iConfig.getParameter<double>("RelIsoThresholdM")),
        reliso_thrT(iConfig.getParameter<double>("RelIsoThresholdT")),
        verbose_(iConfig.getParameter<int>("verbose")),
        dumpForHLS_(iConfig.getParameter<int>("IsodumpForHLS")) {
    if (dumpForHLS_) {
      dumpInput.open("Isolation_Mu_Track_infolist.txt", std::ofstream::out);
      dumpOutput.open("Isolation_Mu_Isolation.txt", std::ofstream::out);
    }
  }

  inline Isolation::~Isolation() {
    if (dumpForHLS_) {
      dumpInput.close();
      dumpOutput.close();
    }
  }

  inline Isolation::Isolation(const Isolation &cpy) : TopoAlgo(cpy) {}

  inline void Isolation::DumpOutputs(std::vector<l1t::TrackerMuon> &trkMus) {
    static std::atomic<int> nevto = 0;
    auto evto = nevto++;
    for (unsigned int i = 0; i < trkMus.size(); ++i) {
      auto mu = trkMus.at(i);
      if (mu.hwPt() != 0) {
        double convertphi = mu.hwPhi() * LSBphi;
        if (convertphi > M_PI) {
          convertphi -= 2 * M_PI;
        }
        dumpOutput << evto << " " << i << " " << mu.hwPt() * LSBpt << " " << mu.hwEta() * LSBeta << " " << convertphi
                   << " " << mu.hwZ0() * LSBGTz0 << " " << mu.hwIso() << endl;
      }
    }
  }

  inline int Isolation::SetAbsIsolationBits(int accum) {
    int iso = (accum <= absiso_thrT ? 3 : accum <= absiso_thrM ? 2 : accum <= absiso_thrL ? 1 : 0);

    if (verbose_) {
      edm::LogInfo("Isolation") << " [DEBUG Isolation] : absiso_threshold L : " << absiso_thrL << " accum " << accum
                                << " bit set : " << (accum < absiso_thrL);
      edm::LogInfo("Isolation") << " [DEBUG Isolation] : absiso_threshold M : " << absiso_thrM << " accum " << accum
                                << " bit set : " << (accum < absiso_thrM);
      edm::LogInfo("Isolation") << " [DEBUG Isolation] : absiso_threshold T : " << absiso_thrT << " accum " << accum
                                << " bit set : " << (accum < absiso_thrT);
      edm::LogInfo("Isolation") << " [DEBUG Isolation] : absiso : " << (iso);
    }
    return iso;
  }

  inline int Isolation::SetRelIsolationBits(int accum, int mupt) {
    const static reliso_thresh_t relisoL(reliso_thrL);
    const static reliso_thresh_t relisoM(reliso_thrM);
    const static reliso_thresh_t relisoT(reliso_thrT);

    iso_accum_t thrL = relisoL * mupt;
    iso_accum_t thrM = relisoM * mupt;
    iso_accum_t thrT = relisoT * mupt;

    int iso = (accum <= thrT.to_int() ? 3 : accum <= thrM.to_int() ? 2 : accum <= thrL.to_int() ? 1 : 0);

    if (verbose_) {
      edm::LogInfo("Isolation") << " [DEBUG Isolation] : reliso_threshold L : " << thrL << " accum " << accum
                                << " bit set : " << (accum < thrL.to_int());
      edm::LogInfo("Isolation") << " [DEBUG Isolation] : reliso_threshold M : " << thrM << " accum " << accum
                                << " bit set : " << (accum < thrM.to_int());
      edm::LogInfo("Isolation") << " [DEBUG Isolation] : reliso_threshold T : " << thrT << " accum " << accum
                                << " bit set : " << (accum < thrT.to_int());
      edm::LogInfo("Isolation") << " [DEBUG Isolation] : reliso : " << (iso << 2) << " org " << iso;
    }

    return iso << 2;
  }

  inline void Isolation::isolation_allmu_alltrk(std::vector<l1t::TrackerMuon> &trkMus,
                                                std::vector<ConvertedTTTrack> &convertedTracks) {
    load(trkMus, convertedTracks);
    if (dumpForHLS_) {
      DumpInputs();
    }

    static std::atomic<int> itest = 0;
    if (verbose_) {
      edm::LogInfo("Isolation") << "........ RUNNING TEST NUMBER .......... " << itest++;
    }

    for (auto &mu : trkMus) {
      int accum = 0;
      int iso_ = 0;
      std::vector<unsigned> overlaps;
      for (auto t : convertedTracks) {
        unsigned ovrl = compute_trk_iso(mu, t);
        if (ovrl != 0) {
          accum += OverlapRemoval(ovrl, overlaps) * t.pt();
        }
      }

      // Only 8 bit for accumation?
      mu.setHwIsoSum(accum);

      iso_accum_t temp(accum);
      accum = temp.to_int();

      mu.setHwIsoSumAp(accum);

      iso_ |= SetAbsIsolationBits(accum);
      iso_ |= SetRelIsolationBits(accum, mu.hwPt());

      mu.setHwIso(iso_);
    }

    if (dumpForHLS_) {
      DumpOutputs(trkMus);
    }
  }

  // ===  FUNCTION  ============================================================
  //         Name:  Isolation::OverlapRemoval
  //  Description:
  // ===========================================================================
  inline int Isolation::OverlapRemoval(unsigned &ovrl, std::vector<unsigned> &overlaps) {
    for (auto i : overlaps) {
      // same tracks with Phi can be off by 1 LSB
      unsigned diff = ovrl - i;
      if (diff <= 1 || diff == unsigned(-1)) {
        // When Overlap, return 0 so that this track won't be consider
        return 0;
      }
    }
    overlaps.push_back(ovrl);
    return 1;
  }  // -----  end of function Isolation::OverlapRemoval  -----

  inline unsigned Isolation::compute_trk_iso(l1t::TrackerMuon &in_mu, ConvertedTTTrack &in_trk) {
    int dphi = deltaPhi(in_mu.hwPhi(), in_trk.phi());
    int deta = deltaEta(in_mu.hwEta(), in_trk.eta());
    int dz0 = deltaZ0(in_mu.hwZ0(), in_trk.z0());

    bool pass_deta = (deta < c_iso_dangle_max ? true : false);
    bool pass_dphi = (dphi < c_iso_dangle_max ? true : false);
    bool pass_dz0 = (dz0 < c_iso_dz_max ? true : false);
    bool pass_trkpt = (in_trk.pt() >= c_iso_pt_min ? true : false);
    bool pass_ovrl = (deta > 0 || dphi > 0 ? true : false);

    if (verbose_) {
      edm::LogInfo("Isolation") << " [DEBUG compute_trk_iso] : Start of debug msg for compute_trk_iso";
      edm::LogInfo("Isolation") << " [DEBUG compute_trk_iso] : incoming muon (pt / eta / phi / z0 / isvalid)";
      edm::LogInfo("Isolation") << " [DEBUG compute_trk_iso] : MU  =  " << in_mu.hwPt() << " / " << in_mu.hwEta()
                                << " / " << in_mu.hwPhi() << " / " << in_mu.hwZ0() << " / " << 1;
      edm::LogInfo("Isolation") << " [DEBUG compute_trk_iso] : incoming track (pt / eta / phi / z0 / isvalid)";
      edm::LogInfo("Isolation") << " [DEBUG compute_trk_iso] : TRK =  " << in_trk.pt() << " / " << in_trk.eta() << " / "
                                << in_trk.phi() << " / " << in_trk.z0() << " / " << 1;
      edm::LogInfo("Isolation") << " [DEBUG compute_trk_iso] : Delta phi : " << dphi;
      edm::LogInfo("Isolation") << " [DEBUG compute_trk_iso] : Delta eta : " << deta;
      edm::LogInfo("Isolation") << " [DEBUG compute_trk_iso] : Delta z0  : " << dz0;
      edm::LogInfo("Isolation") << " [DEBUG compute_trk_iso] : pass_deta      : " << pass_deta;
      edm::LogInfo("Isolation") << " [DEBUG compute_trk_iso] : pass_dphi      : " << pass_dphi;
      edm::LogInfo("Isolation") << " [DEBUG compute_trk_iso] : pass_dz0       : " << pass_dz0;
      edm::LogInfo("Isolation") << " [DEBUG compute_trk_iso] : pass_trkpt     : " << pass_trkpt;
      edm::LogInfo("Isolation") << " [DEBUG compute_trk_iso] : pass_ovrl      : " << pass_ovrl;
    }
    // match conditions
    if (pass_deta && pass_dphi && pass_dz0 && pass_trkpt && pass_ovrl) {
      if (verbose_) {
        edm::LogInfo("Isolation") << " [DEBUG compute_trk_iso] : THE TRACK WAS MATCHED";
        edm::LogInfo("Isolation") << " [DEBUG compute_trk_iso] : RETURN         : " << in_trk.pt();
      }

      //return in_trk.pt();
      // Return fixed bit output for duplication removal.
      // dZ0(8bis) + deta(10bits)+dphi(10bits)
      unsigned int retbits = 0;
      retbits |= (dz0 & ((1 << 9) - 1)) << 20;
      retbits |= (deta & ((1 << 11) - 1)) << 10;
      retbits |= (dphi & ((1 << 11) - 1));
      return retbits;
    } else {
      return 0;
    }
  }
}  // namespace Phase2L1GMT
#endif  // ----- #ifndef PHASE2GMT_ISOLATION -----
