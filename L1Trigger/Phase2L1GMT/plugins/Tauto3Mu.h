// ===========================================================================
//
//       Filename:  Tauto3Mu.h
//
//    Description:
//
//        Version:  1.0
//        Created:  03/15/2021 07:33:59 PM
//       Revision:  none
//       Compiler:  g++
//
//         Author:  Zhenbin Wu, zhenbin.wu@gmail.com
//
// ===========================================================================

#ifndef PHASE2GMT_TAUTO3MU
#define PHASE2GMT_TAUTO3MU

#include "TopologicalAlgorithm.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace Phase2L1GMT {
  class Tauto3Mu : public TopoAlgo {
  public:
    Tauto3Mu(const edm::ParameterSet &iConfig);
    ~Tauto3Mu();
    Tauto3Mu(const Tauto3Mu &cpy);
    // Interface function
    bool GetTau3Mu(std::vector<l1t::TrackerMuon> &trkMus, std::vector<ConvertedTTTrack> &convertedTracks);

  private:
    bool Find3MuComb(std::vector<l1t::TrackerMuon> &trkMus);

    bool FindCloset3Mu(std::vector<std::pair<int, unsigned int> > &mu_phis,
                       std::vector<std::pair<unsigned, unsigned> > &nearby3mu);

    int Get3MuDphi(unsigned target, unsigned obj1, unsigned obj2);

    int Get3MuMass(unsigned target, unsigned obj1, unsigned obj2);

    int GetDiMass(const l1t::TrackerMuon &mu1, const l1t::TrackerMuon &mu2);
  };

  inline Tauto3Mu::Tauto3Mu(const edm::ParameterSet &iConfig) {}

  inline Tauto3Mu::~Tauto3Mu() {}

  inline Tauto3Mu::Tauto3Mu(const Tauto3Mu &cpy) : TopoAlgo(cpy) {}

  // ===  FUNCTION  ============================================================
  //         Name:  Tauto3Mu::GetTau3Mu
  //  Description:
  // ===========================================================================
  inline bool Tauto3Mu::GetTau3Mu(std::vector<l1t::TrackerMuon> &trkMus,
                                  std::vector<ConvertedTTTrack> &convertedTracks) {
    Find3MuComb(trkMus);
    return true;
  }  // -----  end of function Tauto3Mu::GetTau3Mu  -----

  // ===  FUNCTION  ============================================================
  //         Name:  Tauto3Mu::Find3MuComb
  //  Description:
  // ===========================================================================
  inline bool Tauto3Mu::Find3MuComb(std::vector<l1t::TrackerMuon> &trkMus) {
    // vector< phi, index of trackerMuon >
    std::vector<std::pair<int, unsigned int> > mu_phis;
    for (unsigned i = 0; i < trkMus.size(); ++i) {
      mu_phis.push_back(std::make_pair(trkMus.at(i).hwPhi(), i));
    }

    std::sort(mu_phis.begin(), mu_phis.end());

    std::vector<std::pair<unsigned, unsigned> > nearby3mu;
    std::vector<int> mu3mass;
    FindCloset3Mu(mu_phis, nearby3mu);

    for (unsigned i = 0; i < trkMus.size(); ++i) {
      int trimass = Get3MuMass(i, nearby3mu.at(i).first, nearby3mu.at(i).second);
      mu3mass.push_back(trimass);
    }

    return true;
  }  // -----  end of function Tauto3Mu::Find3MuComb  -----

  // ===  FUNCTION  ============================================================
  //         Name:  Tauto3Mu::Get3MuMass
  //  Description:
  // ===========================================================================
  inline int Tauto3Mu::Get3MuMass(unsigned target, unsigned obj1, unsigned obj2) {
    int mass12 = GetDiMass(trkMus->at(target), trkMus->at(obj1));
    int mass23 = GetDiMass(trkMus->at(obj1), trkMus->at(obj2));
    int mass31 = GetDiMass(trkMus->at(obj2), trkMus->at(target));

    return mass12 + mass23 + mass31;
  }  // -----  end of function Tauto3Mu::Get3MuMass  -----

  // ===  FUNCTION  ============================================================
  //         Name:  Tauto3Mu::GetDiMass
  //  Description:
  // ===========================================================================
  inline int Tauto3Mu::GetDiMass(const l1t::TrackerMuon &mu1, const l1t::TrackerMuon &mu2) {
    int deta = deltaEta(mu1.hwEta(), mu2.hwEta());
    int dphi = deltaPhi(mu1.hwPhi(), mu2.hwPhi());
    int mass = 2 * mu1.hwPt() * mu2.hwPt() * (cosh(deta) - cos(dphi));
    return mass;
  }  // -----  end of function Tauto3Mu::GetDiMass  -----

  // ===  FUNCTION  ============================================================
  //         Name:  Tauto3Mu::FindCloset3Mu
  //  Description:
  // ===========================================================================
  inline bool Tauto3Mu::FindCloset3Mu(std::vector<std::pair<int, unsigned int> > &mu_phis,
                                      std::vector<std::pair<unsigned, unsigned> > &nearby3mu) {
    nearby3mu.clear();

    std::vector<std::pair<int, unsigned int> > temp(mu_phis);

    // Round the last 2 to first element of vector
    temp.insert(temp.begin(), mu_phis.back());
    temp.insert(temp.begin(), *(mu_phis.rbegin() - 1));
    // Append the first two element to vector
    temp.push_back(mu_phis.front());
    temp.push_back(*(mu_phis.begin() + 1));

    for (unsigned i = 2; i < temp.size() - 2; ++i) {
      int combleft = Get3MuDphi(temp[i].second, temp[i - 1].second, temp[i - 2].second);
      std::pair<unsigned, unsigned> neighbors(temp[i - 1].second, temp[i - 2].second);
      int mincomb(combleft);

      int combcenter = Get3MuDphi(temp[i].second, temp[i - 1].second, temp[i + 1].second);
      if (combcenter < mincomb) {
        neighbors = std::make_pair(temp[i - 1].second, temp[i + 1].second);
        mincomb = combcenter;
      }

      int combright = Get3MuDphi(temp[i].second, temp[i + 1].second, temp[i + 2].second);
      if (combright < mincomb) {
        neighbors = std::make_pair(temp[i + 1].second, temp[i + 2].second);
      }

      nearby3mu.push_back(neighbors);
    }

    return true;
  }  // -----  end of function Tauto3Mu::FindCloset3Mu  -----

  inline int Tauto3Mu::Get3MuDphi(unsigned target, unsigned obj1, unsigned obj2) {
    int dPhi1 = deltaPhi(trkMus->at(target).hwPhi(), trkMus->at(obj1).hwPhi());
    int dPhi2 = deltaPhi(trkMus->at(target).hwPhi(), trkMus->at(obj2).hwPhi());
    return dPhi1 + dPhi2;
  }
}  // namespace Phase2L1GMT

#endif  // ----- #ifndef PHASE2GMT_TAUTO3MU -----
