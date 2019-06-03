///
/// \class l1t::CaloParams
///
/// Description: Placeholder for calorimeter trigger parameters
///
/// Implementation:
///
///
/// \author: Jim Brooke
///

#ifndef CaloParams_h
#define CaloParams_h

#include <memory>
#include <iostream>
#include <vector>
#include <cmath>

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/L1TObjects/interface/LUT.h"

namespace l1t {

  class CaloParams {
  public:
    enum { Version = 2 };

    class Node {
    public:
      std::string type_;
      unsigned version_;
      l1t::LUT LUT_;
      std::vector<double> dparams_;
      std::vector<unsigned> uparams_;
      std::vector<int> iparams_;
      std::vector<std::string> sparams_;
      Node() {
        type_ = "unspecified";
        version_ = 1;
      }
      COND_SERIALIZABLE;
    };

    class TowerParams {
    public:
      /* Towers */

      // LSB of HCAL scale
      double lsbH_;

      // LSB of ECAL scale
      double lsbE_;

      // LSB of ECAL+HCAL sum scale
      double lsbSum_;

      // number of bits for HCAL encoding
      int nBitsH_;

      // number of bits for ECAL encoding
      int nBitsE_;

      // number of bits for ECAL+HCAL sum encoding
      int nBitsSum_;

      // number of bits for ECAL/HCAL ratio encoding
      int nBitsRatio_;

      // bitmask for storing HCAL Et in  object
      int maskH_;

      // bitmask for storing ECAL ET in  object
      int maskE_;

      // bitmask for storing ECAL+HCAL sum in  object
      int maskSum_;

      // bitmask for storing ECAL/HCAL ratio in  object
      int maskRatio_;

      // turn encoding on/off
      bool doEncoding_;

      TowerParams()
          : lsbH_(0),
            lsbE_(0),
            lsbSum_(0),
            nBitsH_(0),
            nBitsE_(0),
            nBitsSum_(0),
            nBitsRatio_(0),
            maskH_(0),
            maskE_(0),
            maskSum_(0),
            maskRatio_(0),
            doEncoding_(false) { /* no-op */
      }

      COND_SERIALIZABLE;
    };

    class EgParams {
    public:
      // EG LSB
      double lsb_;

      // Et threshold on EG seed tower
      double seedThreshold_;

      // Et threshold on EG neighbour tower(s)
      double neighbourThreshold_;

      // Et threshold on HCAL for H/E computation
      double hcalThreshold_;

      // EG maximum value of HCAL Et
      double maxHcalEt_;

      // Et threshold to remove the H/E cut from the EGammas
      double maxPtHOverE_;

      // Range of jet isolation for EG (in rank!) (Stage1Layer2)
      int minPtJetIsolation_;
      int maxPtJetIsolation_;

      // Range of 3x3 HoE isolation for EG (in rank!) (Stage1Layer2)
      int minPtHOverEIsolation_;
      int maxPtHOverEIsolation_;

      // isolation area in eta is seed tower +/- <=egIsoAreaNrTowersPhi
      unsigned isoAreaNrTowersEta_;

      // isolation area in phi is seed tower +/- <=egIsoAreaNrTowersPhi
      unsigned isoAreaNrTowersPhi_;

      // veto region is seed tower +/- <=egIsoVetoNrTowersPhi
      unsigned isoVetoNrTowersPhi_;

      EgParams()
          : lsb_(0),
            seedThreshold_(0),
            neighbourThreshold_(0),
            hcalThreshold_(0),
            maxHcalEt_(0),
            maxPtHOverE_(0),
            minPtJetIsolation_(0),
            maxPtJetIsolation_(0),
            minPtHOverEIsolation_(0),
            maxPtHOverEIsolation_(0),
            isoAreaNrTowersEta_(0),
            isoAreaNrTowersPhi_(0),
            isoVetoNrTowersPhi_(0) { /* no-op */
      }

      COND_SERIALIZABLE;
    };

    class TauParams {
    public:
      // Tau LSB
      double lsb_;

      // Et threshold on tau seed tower
      double seedThreshold_;

      // Et threshold on tau neighbour towers
      double neighbourThreshold_;

      // Et limit when to switch off tau veto requirement
      double maxPtTauVeto_;

      // Et limit when to switch off tau isolation requirement
      double minPtJetIsolationB_;

      // Et jet isolation limit for Taus (Stage1Layer2)
      double maxJetIsolationB_;

      // Relative jet isolation cut for Taus (Stage1Layer2)
      double maxJetIsolationA_;

      // Eta min and max for Iso-Tau collections (Stage1Layer2)
      int isoEtaMin_;
      int isoEtaMax_;

      // isolation area in eta is seed tower +/- <=tauIsoAreaNrTowersEta
      unsigned isoAreaNrTowersEta_;

      // isolation area in phi is seed tower +/- <=tauIsoAreaNrTowersPhi
      unsigned isoAreaNrTowersPhi_;

      // veto region is seed tower +/- <=tauIsoVetoNrTowersPhi
      unsigned isoVetoNrTowersPhi_;

      TauParams()
          : lsb_(0),
            seedThreshold_(0),
            neighbourThreshold_(0),
            maxPtTauVeto_(0),
            minPtJetIsolationB_(0),
            maxJetIsolationB_(0),
            maxJetIsolationA_(0),
            isoEtaMin_(0),
            isoEtaMax_(0),
            isoAreaNrTowersEta_(0),
            isoAreaNrTowersPhi_(0),
            isoVetoNrTowersPhi_(0) { /* no-op */
      }

      COND_SERIALIZABLE;
    };

    class JetParams {
    public:
      // Jet LSB
      double lsb_;

      // Et threshold on jet seed tower/region
      double seedThreshold_;

      // Et threshold on neighbouring towers/regions
      double neighbourThreshold_;

      JetParams() : lsb_(0), seedThreshold_(0), neighbourThreshold_(0) { /* no-op */
      }

      COND_SERIALIZABLE;
    };

    CaloParams() : pnode_(0) { version_ = Version; }
    ~CaloParams() {}

  protected:
    unsigned version_;

    std::vector<Node> pnode_;

    TowerParams towerp_;

    // Region LSB
    double regionLsb_;

    EgParams egp_;
    TauParams taup_;
    JetParams jetp_;

    /* Sums */

    // EtSum LSB
    double etSumLsb_;

    // minimum eta for EtSums (index is particular EtSum.  ETT=1, HTT=2, MET=3, MHT=4, other values reserved).
    std::vector<int> etSumEtaMin_;

    // maximum eta for EtSums (index is particular EtSum.  ETT=1, HTT=2, MET=3, MHT=4, other values reserved).
    std::vector<int> etSumEtaMax_;

    // minimum eta for EtSums (index is particular EtSum.  ETT=1, HTT=2, MET=3, MHT=4, other values reserved).
    std::vector<double> etSumEtThreshold_;

    COND_SERIALIZABLE;
  };

}  // namespace l1t
#endif
