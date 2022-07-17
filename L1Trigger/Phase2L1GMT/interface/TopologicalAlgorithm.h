// ===========================================================================
//
//       Filename:  TopologicalAlgorithm.h
//
//    Description:  A base class for all the topological algorithms
//
//        Version:  1.0
//        Created:  03/03/2021 10:14:23 AM
//       Revision:  none
//       Compiler:  g++
//
//         Author:  Zhenbin Wu, zhenbin.wu@gmail.com
//
// ===========================================================================

#ifndef PHASE2GMT_TOPOLOGICALALGORITHM
#define PHASE2GMT_TOPOLOGICALALGORITHM

#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "L1Trigger/Phase2L1GMT/interface/ConvertedTTTrack.h"
#include "L1Trigger/Phase2L1GMT/interface/Constants.h"

#include <fstream>

namespace Phase2L1GMT {

  class TopoAlgo {
  public:
    TopoAlgo();
    ~TopoAlgo();
    TopoAlgo(const TopoAlgo &cpy);
    void load(std::vector<l1t::TrackerMuon> &trkMus, std::vector<ConvertedTTTrack> &convertedTracks);
    void DumpInputs();

    int deltaEta(const int eta1, const int eta2);
    int deltaZ0(const int Z01, const int Z02);
    int deltaPhi(int phi1, int phi2);

  protected:
    std::vector<l1t::TrackerMuon> *trkMus;
    std::vector<ConvertedTTTrack> *convertedTracks;
    std::ofstream dumpInput;
  };

  inline TopoAlgo::TopoAlgo() {}

  inline TopoAlgo::~TopoAlgo() {}

  inline TopoAlgo::TopoAlgo(const TopoAlgo &cpy) {}

  // ===  FUNCTION  ============================================================
  //         Name:  TopoAlgo::load
  //  Description:
  // ===========================================================================
  inline void TopoAlgo::load(std::vector<l1t::TrackerMuon> &trkMus_, std::vector<ConvertedTTTrack> &convertedTracks_) {
    trkMus = &trkMus_;
    convertedTracks = &convertedTracks_;
  }  // -----  end of function TopoAlgo::load  -----

  inline void TopoAlgo::DumpInputs() {
    static int nevti = 0;
    int totalsize = 0;
    // Current setting
    int constexpr exptotal = 12 + 18 * 100;  // N_Muon + N_TRK_LINKS * NTRKperlinks
    for (unsigned int i = 0; i < 12; ++i) {
      if (i < trkMus->size())
        dumpInput << " " << nevti << " 0 " << i << " " << trkMus->at(i).hwPt() * LSBpt << " "
                  << trkMus->at(i).hwEta() * LSBeta << " " << trkMus->at(i).hwPhi() * LSBphi << " "
                  << trkMus->at(i).hwZ0() * LSBGTz0 << " " << trkMus->at(i).charge() << std::endl;
      else
        dumpInput << " " << nevti << " 0 " << i << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0
                  << std::endl;
      totalsize++;
    }
    for (unsigned int i = 0; i < convertedTracks->size(); ++i) {
      dumpInput << " " << nevti << " 1 " << i << " " << convertedTracks->at(i).pt() * LSBpt << " "
                << convertedTracks->at(i).eta() * LSBeta << " " << convertedTracks->at(i).phi() * LSBphi << " "
                << convertedTracks->at(i).z0() * LSBGTz0 << " " << convertedTracks->at(i).charge() << " "
                << convertedTracks->at(i).quality() << std::endl;
      totalsize++;
    }
    int ntrks = convertedTracks->size();
    // Pat the remaining
    while (totalsize < exptotal) {
      dumpInput << " " << nevti << " 1 " << ntrks++ << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " "
                << 0 << std::endl;
      totalsize++;
    }
    nevti++;
  }

  inline int TopoAlgo::deltaEta(const int eta1, const int eta2) {
    static const int maxbits = (1 << BITSETA) - 1;
    int deta = abs(eta1 - eta2);
    deta &= maxbits;
    return deta;
  }

  inline int TopoAlgo::deltaZ0(const int Z01, const int Z02) {
    static const int maxbits = (1 << BITSZ0) - 1;
    int dZ0 = abs(Z01 - Z02);
    dZ0 &= maxbits;
    return dZ0;
  }

  // Ideal the object should carry its own ap types once we finalize
  inline int TopoAlgo::deltaPhi(int phi1, int phi2) {
    static const int maxbits = (1 << BITSPHI) - 1;
    static const int pibits = (1 << (BITSPHI - 1));
    int dphi = abs(phi1 - phi2);
    if (dphi >= pibits)
      dphi = maxbits - dphi;
    return dphi;
  }
}  // namespace Phase2L1GMT

#endif  // ----- #ifndef PHASE2GMT_TOPOLOGICALALGORITHM -----
