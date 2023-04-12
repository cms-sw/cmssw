//------------------------------------
// Helper functions for Phase2L1CaloEGammaEmulator.h
//------------------------------------

#include <ap_int.h>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "L1Trigger/L1CaloTrigger/interface/ParametricCalibration.h"

// Output collections
#include "DataFormats/L1TCalorimeterPhase2/interface/CaloCrystalCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/CaloTower.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/DigitizedClusterCorrelator.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/DigitizedTowerCorrelator.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/DigitizedClusterGT.h"

#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"

#ifndef PHASE_2_L1_CALO_EGAMMA_EMULATOR_H
#define PHASE_2_L1_CALO_EGAMMA_EMULATOR_H

namespace p2eg {

  static constexpr int n_towers_Eta = 34;
  static constexpr int n_towers_Phi = 72;
  static constexpr int n_towers_halfPhi = 36;
  static constexpr int n_towers_cardEta = 17;  // new: equivalent to n_towers_per_link
  static constexpr int n_towers_cardPhi = 4;
  static constexpr int n_crystals_cardEta = (n_towers_Eta * n_towers_cardEta);
  static constexpr int n_crystals_cardPhi = (n_towers_Phi * n_towers_cardPhi);

  // outputs
  static constexpr int n_links_card = 4;      // 4 links per card
  static constexpr int n_clusters_link = 2;   // 2 clusters sent out in each link
  static constexpr int n_clusters_4link = 8;  // 8 clusters sent out in 4 links
  static constexpr int n_towers_per_link = 17;

  static constexpr int CRYSTALS_IN_TOWER_ETA = 5;
  static constexpr int CRYSTALS_IN_TOWER_PHI = 5;

  static constexpr int TOWER_IN_ETA = 3;  // number of towers in eta, in one 3x4 region (barrel)
  static constexpr int TOWER_IN_PHI = 4;  // number of towers in phi, in one 3x4 region (barrel)

  static constexpr int CRYSTAL_IN_ETA = 15;  // number of crystals in eta, in one 3x4 region (barrel)
  static constexpr int CRYSTAL_IN_PHI = 20;  // number of crystals in phi, in one 3x4 region (barrel)

  static constexpr float ECAL_eta_range = 1.4841;
  static constexpr float half_crystal_size = 0.00873;

  static constexpr float slideIsoPtThreshold = 80;
  static constexpr float a0_80 = 0.85, a1_80 = 0.0080, a0 = 0.21;                        // passes_iso
  static constexpr float b0 = 0.38, b1 = 1.9, b2 = 0.05;                                 // passes_looseTkiso
  static constexpr float c0_ss = 0.94, c1_ss = 0.052, c2_ss = 0.044;                     // passes_ss
  static constexpr float d0 = 0.96, d1 = 0.0003;                                         // passes_photon
  static constexpr float e0_looseTkss = 0.944, e1_looseTkss = 0.65, e2_looseTkss = 0.4;  // passes_looseTkss
  static constexpr float cut_500_MeV = 0.5;

  static constexpr float ECAL_LSB = 0.125;  // to convert from int to float (GeV) multiply by LSB
  static constexpr float HCAL_LSB = 0.5;

  static constexpr int N_CLUSTERS_PER_REGION = 4;  // number of clusters per ECAL region
  static constexpr int N_REGIONS_PER_CARD = 6;     // number of ECAL regions per card

  // GCT constants
  static constexpr int N_RCTCARDS_PHI = 8;
  static constexpr int N_RCTGCT_FIBERS = 4;
  static constexpr int N_RCTTOWERS_FIBER = 17;
  static constexpr int N_RCTCLUSTERS_FIBER = 2;

  static constexpr int N_GCTCARDS = 3;
  static constexpr int N_GCTCORR_FIBERS = 48;
  static constexpr int N_GCTTOWERS_FIBER = 17;
  static constexpr int N_GCTCLUSTERS_FIBER = 2;

  static constexpr int N_GCTINTERNAL_FIBERS = 64;
  static constexpr int N_GCTPOSITIVE_FIBERS = 32;
  static constexpr int N_GCTETA = 34;
  static constexpr int N_GCTPHI = 32;

  static constexpr int GCTCARD_0_TOWER_IPHI_OFFSET =
      20;  // for emulator: "top" of the GCT card in phi is tower idx 20, for GCT card #0
  static constexpr int GCTCARD_1_TOWER_IPHI_OFFSET =
      44;  // same but for GCT card #1 (card #1 wraps around phi = 180 degrees)
  static constexpr int GCTCARD_2_TOWER_IPHI_OFFSET =
      68;  // same for GCT card #2 (card #2 also wraps around phi = 180 degrees)

  static constexpr int N_GCTTOWERS_CLUSTER_ISO_ONESIDE = 5;  // window size of isolation sum (5x5 in towers)

  /*
    * Convert HCAL ET to ECAL ET convention 
    */
  inline ap_uint<12> convertHcalETtoEcalET(ap_uint<12> HCAL) {
    float hcalEtAsFloat = HCAL * HCAL_LSB;
    return (ap_uint<12>(hcalEtAsFloat / ECAL_LSB));
  }

  //////////////////////////////////////////////////////////////////////////
  // RCT: indexing helper functions
  //////////////////////////////////////////////////////////////////////////

  // Assert that the card index is within bounds. (Valid cc: 0 to 35, since there are 36 RCT cards)
  inline bool isValidCard(int cc) { return ((cc > -1) && (cc < 36)); }

  // RCT Cards: need to know their min/max crystal boundaries.

  // For a card (ranging from 0 to 35, since there are 36 cards), return the iEta of the crystal with max iEta.
  // This represents the card boundaries in eta (identical to getEtaMax_card in the original emulator)
  inline int getCard_iEtaMax(int cc) {
    assert(isValidCard(cc));

    int etamax = 0;
    if (cc % 2 == 0)                                            // Even card: negative eta
      etamax = (n_towers_cardEta * CRYSTALS_IN_TOWER_ETA - 1);  // First eta half. 5 crystals in eta in 1 tower.
    else                                                        // Odd card: positive eta
      etamax = (n_towers_Eta * CRYSTALS_IN_TOWER_ETA - 1);
    return etamax;
  }

  // Same as above but for minimum iEta.
  inline int getCard_iEtaMin(int cc) {
    int etamin = 0;
    if (cc % 2 == 0)  // Even card: negative eta
      etamin = (0);
    else  // Odd card: positive eta
      etamin = (n_towers_cardEta * CRYSTALS_IN_TOWER_ETA);
    return etamin;
  }

  // Same as above but for maximum iPhi.
  inline int getCard_iPhiMax(int cc) {
    int phimax = ((cc / 2) + 1) * 4 * CRYSTALS_IN_TOWER_PHI - 1;
    return phimax;
  }

  // Same as above but for minimum iPhi.
  inline int getCard_iPhiMin(int cc) {
    int phimin = (cc / 2) * 4 * CRYSTALS_IN_TOWER_PHI;
    return phimin;
  }

  // Given the RCT card number (0-35), get the crystal iEta of the "bottom left" corner
  inline int getCard_refCrystal_iEta(int cc) {
    if ((cc % 2) == 1) {  // if cc is odd (positive eta)
      return (17 * CRYSTALS_IN_TOWER_ETA);
    } else {  // if cc is even (negative eta) the bottom left corner is further in eta, hence +4
      return ((16 * CRYSTALS_IN_TOWER_ETA) + 4);
    }
  }

  // Given the RCT card number (0-35), get the global crystal iPhi of the "bottom left" corner (0- 71*5)
  inline int getCard_refCrystal_iPhi(int cc) {
    if ((cc % 2) == 1) {
      // if cc is odd: positive eta
      return int(cc / 2) * TOWER_IN_PHI * CRYSTALS_IN_TOWER_PHI;
    } else {
      // if cc is even, the bottom left corner is further in phi, hence the +4 and -1
      return (((int(cc / 2) * TOWER_IN_PHI) + 4) * CRYSTALS_IN_TOWER_PHI) - 1;
    }
  }

  // Towers: Go from real (eta, phi) to tower absolute ID

  /* 
  * For a real eta, get the tower absolute Eta index (possible values are 0-33, since there
  * are 34 towers in eta. (Adapted from getTower_absoluteEtaID)
  */
  inline int getTower_absEtaID(float eta) {
    float size_cell = 2 * ECAL_eta_range / n_towers_Eta;
    int etaID = int((eta + ECAL_eta_range) / size_cell);
    return etaID;
  }

  /* 
  * Same as above, but for phi.
  * Possible values range from 0-71 (Adapted from getTower_absolutePhiID)
  */
  inline int getTower_absPhiID(float phi) {
    float size_cell = 2 * M_PI / n_towers_Phi;
    int phiID = int((phi + M_PI) / size_cell);
    return phiID;
  }

  // Towers: Go from firmware specifics (RCT card, tower number in link, and link number (all firmware convention))
  //         to tower absolute ID.

  /* 
  * Get the global tower iEta (0-31) from the firmware card, tower number (0-16), and link (0-3). Respects the fact that
  * in the firmware, negative eta cards are "rotated" (link 0, tower 0) starts in the "top right" corner if we
  * look at a diagram of the barrel region.
  */
  inline int getAbsID_iEta_fromFirmwareCardTowerLink(int nCard, int nTower, int nLink) {
    // iEta only depends on the tower position in the link
    (void)nCard;
    (void)nLink;
    if ((nCard % 2) == 1) {  // if cc is odd (positive eta), e.g. nTower = 0 will correspond to absolute iEta ID 17.
      return n_towers_per_link + nTower;
    } else {  // if cc is even (negative eta): e.g. nTower = 0 will correspond to absolute iEta ID 16.
      return (16 - nTower);
    }
  }

  /*
  * Get the global tower iPhi (0-71) from the firmware card, tower number (0-16), and link (0-3).
  */
  inline int getAbsID_iPhi_fromFirmwareCardTowerLink(int nCard, int nTower, int nLink) {
    // iPhi only depends on the card and link number
    (void)nTower;
    if ((nCard % 2) == 1) {  // if cc is odd (positive eta),
      // e.g. cc=3, link #2, global iPhi = int(3/2) * 4 + 2 = 6
      return (int(nCard / 2) * TOWER_IN_PHI) + nLink;
    } else {  // if cc is even (negative eta)
      // e.g. cc=4, link #2, global iPhi = int(4/2) * 4 + (4 - 2 - 1)
      //                                 = 2*4 + 1
      //                                 = 9
      // minus one is because TOWER_IN_PHI is 4
      return (int(nCard / 2) * TOWER_IN_PHI) + (TOWER_IN_PHI - nLink - 1);
    }
  }

  // Towers: Go from absolute ID, back to real eta and phi.

  /*
  * From the tower absolute ID in eta (0-33), get the real eta of the tower center
  * Same as getTowerEta_fromAbsoluteID in previous CMSSW emulator
  */
  inline float getTowerEta_fromAbsID(int id) {
    float size_cell = 2 * ECAL_eta_range / n_towers_Eta;
    float eta = (id * size_cell) - ECAL_eta_range + 0.5 * size_cell;
    return eta;
  }

  /*
  * From the tower absolute ID in phi (0-71), get the real phi of the tower center
  * Same as getTowerPhi_fromAbsoluteID in previous CMSSW emulator
  */
  inline float getTowerPhi_fromAbsID(int id) {
    float size_cell = 2 * M_PI / n_towers_Phi;
    float phi = (id * size_cell) - M_PI + 0.5 * size_cell;
    return phi;
  }

  /* 
  * Get the RCT card region that a crystal is in, given the "local" iEta of the crystal 
  * 0 is region closest to eta = 0. Regions 0, 1, 2, 3, 4 are in the barrel, Region 5 is in overlap
  */
  inline int getRegionNumber(const int local_iEta) {
    int no = int(local_iEta / (TOWER_IN_ETA * CRYSTALS_IN_TOWER_ETA));
    assert(no < 6);
    return no;
  }

  /*******************************************************************/
  /* RCT classes and structs                                         */
  /*******************************************************************/

  /* 
   * Represents one input HCAL or ECAL hit.
   */
  class SimpleCaloHit {
  private:
    float pt_ = 0;
    float energy_ = 0.;
    ap_uint<10> et_uint_;
    GlobalVector position_;  // As opposed to GlobalPoint, so we can add them (for weighted average)
    HcalDetId id_hcal_;
    EBDetId id_;

  public:
    // tool functions
    inline void setPt() { pt_ = (position_.mag2() > 0) ? energy_ * sin(position_.theta()) : 0; };
    inline void setEnergy(float et) { energy_ = et / sin(position_.theta()); };
    inline void setEt_uint(ap_uint<10> et_uint) { et_uint_ = et_uint; }
    inline void setPosition(const GlobalVector& pos) { position_ = pos; };
    inline void setIdHcal(const HcalDetId& idhcal) { id_hcal_ = idhcal; };
    inline void setId(const EBDetId& id) { id_ = id; };

    inline float pt() const { return pt_; };
    inline float energy() const { return energy_; };
    inline ap_uint<10> et_uint() const { return et_uint_; };
    inline const GlobalVector& position() const { return position_; };
    inline const EBDetId& id() const { return id_; };

    /* 
       * Get crystal's iEta from real eta. (identical to getCrystal_etaID in L1EGammaCrystalsEmulatorProducer.cc)
       * This "global" iEta ranges from 0 to (33*5) since there are 34 towers in eta in the full detector, 
       * each with five crystals in eta.
       */
    int crystaliEta(void) const {
      float size_cell = 2 * ECAL_eta_range / (CRYSTALS_IN_TOWER_ETA * n_towers_Eta);
      int iEta = int((position().eta() + ECAL_eta_range) / size_cell);
      return iEta;
    }

    /* 
       * Get crystal's iPhi from real phi. (identical to getCrystal_phiID in L1EGammaCrystalsEmulatorProducer.cc)
       * This "global" iPhi ranges from 0 to (71*5) since there are 72 towers in phi in the full detector, each with five crystals in eta.
       */
    int crystaliPhi(void) const {
      float phi = position().phi();
      float size_cell = 2 * M_PI / (CRYSTALS_IN_TOWER_PHI * n_towers_Phi);
      int iPhi = int((phi + M_PI) / size_cell);
      return iPhi;
    }

    /*
       * Check if it falls within the boundary of a card.
       */
    bool isInCard(int cc) const {
      return (crystaliPhi() <= getCard_iPhiMax(cc) && crystaliPhi() >= getCard_iPhiMin(cc) &&
              crystaliEta() <= getCard_iEtaMax(cc) && crystaliEta() >= getCard_iEtaMin(cc));
    };

    /*
      * For a crystal with real eta, and falling in card cc, get its local iEta 
      * relative to the bottom left corner of the card (possible local iEta ranges from 0 to 17 * 5,
      * since in one card, there are 17 towers in eta, each with 5 crystals in eta.
      */
    int crystalLocaliEta(int cc) const { return abs(getCard_refCrystal_iEta(cc) - crystaliEta()); }

    /*
      * Same as above, but for iPhi (possible local iPhi ranges from 0 to (3*5), since in one card,
      * there are 4 towers in phi, each with 5 crystals in phi.
      */
    int crystalLocaliPhi(int cc) const { return abs(getCard_refCrystal_iPhi(cc) - crystaliPhi()); }

    /*
       * Print hit info
       */
    void printHitInfo(std::string description = "") const {
      std::cout << "[printHitInfo]: [" << description << "]"
                << " hit with energy " << pt() << " at eta " << position().eta() << ", phi " << position().phi()
                << std::endl;
    }
  };

  /*******************************************************************/

  /*
  * linkECAL class: represents one ECAL link (one tower: 5x5 crystals)
  */

  class linkECAL {
  private:
    ap_uint<10> crystalE[CRYSTALS_IN_TOWER_ETA][CRYSTALS_IN_TOWER_PHI];

  public:
    // constructor
    linkECAL() {}

    // Set members
    inline void zeroOut() {  // zero out the crystalE array
      for (int i = 0; i < CRYSTALS_IN_TOWER_ETA; i++) {
        for (int j = 0; j < CRYSTALS_IN_TOWER_PHI; j++) {
          crystalE[i][j] = 0;
        }
      }
    };
    inline void setCrystalE(int iEta, int iPhi, ap_uint<10> energy) {
      assert(iEta < CRYSTALS_IN_TOWER_ETA);
      assert(iPhi < CRYSTALS_IN_TOWER_PHI);
      crystalE[iEta][iPhi] = energy;
    };
    inline void addCrystalE(int iEta, int iPhi, ap_uint<10> energy) {
      assert(iEta < CRYSTALS_IN_TOWER_ETA);
      assert(iPhi < CRYSTALS_IN_TOWER_PHI);
      crystalE[iEta][iPhi] += energy;
    };

    // Access members
    inline ap_uint<10> getCrystalE(int iEta, int iPhi) {
      assert(iEta < 5);
      assert(iPhi < 5);
      return crystalE[iEta][iPhi];
    };
  };

  /*******************************************************************/

  /*
  * region3x4 class: represents one 3x4 ECAL region. The region stores no
  *                  information about which card it is located in.
  *                  idx: 0-4. Region 0 is the one closest to eta = 0, counting outwards in eta  
  */

  class region3x4 {
  private:
    int idx_ = -1;
    linkECAL linksECAL[TOWER_IN_ETA][TOWER_IN_PHI];  // 3x4 in towers

  public:
    // constructor
    region3x4() { idx_ = -1; }

    // copy constructor
    region3x4(const region3x4& other) {
      idx_ = other.idx_;
      for (int i = 0; i < TOWER_IN_ETA; i++) {
        for (int j = 0; j < TOWER_IN_PHI; j++) {
          linksECAL[i][j] = other.linksECAL[i][j];
        }
      }
    }

    // overload operator= to use copy constructor
    region3x4 operator=(const region3x4& other) {
      const region3x4& newRegion(other);
      return newRegion;
    };

    // set members
    inline void zeroOut() {
      for (int i = 0; i < TOWER_IN_ETA; i++) {
        for (int j = 0; j < TOWER_IN_PHI; j++) {
          linksECAL[i][j].zeroOut();
        }
      }
    };
    inline void setIdx(int idx) { idx_ = idx; };

    // get members
    inline float getIdx() const { return idx_; };
    inline linkECAL& getLinkECAL(int iEta, int iPhi) { return linksECAL[iEta][iPhi]; };
  };

  /*******************************************************************/

  /*
  * towerHCAL class: represents one HCAL tower
  */

  class towerHCAL {
  private:
    ap_uint<10> et;
    ap_uint<6> fb;

  public:
    // constructor
    towerHCAL() {
      et = 0;
      fb = 0;
    };

    // copy constructor
    towerHCAL(const towerHCAL& other) {
      et = other.et;
      fb = other.fb;
    };

    // set members
    inline void zeroOut() {
      et = 0;
      fb = 0;
    };
    inline void addEt(ap_uint<10> newEt) { et += newEt; };

    // get members
    inline ap_uint<10> getEt() { return et; };
  };

  /*******************************************************************/

  /*
  * towers3x4 class: represents 3x4 array of HCAL towers. idx = 0, 1, ... 4 are the barrel gion
  */

  class towers3x4 {
  private:
    int idx_ = -1;
    towerHCAL towersHCAL[TOWER_IN_ETA][TOWER_IN_PHI];  // 3x4 in towers

  public:
    // constructor
    towers3x4() { idx_ = -1; };

    // copy constructor
    towers3x4(const towers3x4& other) {
      idx_ = other.idx_;
      for (int i = 0; i < TOWER_IN_ETA; i++) {
        for (int j = 0; j < TOWER_IN_PHI; j++) {
          towersHCAL[i][j] = other.towersHCAL[i][j];
        }
      };
    };

    // set members
    inline void zeroOut() {
      for (int i = 0; i < TOWER_IN_ETA; i++) {
        for (int j = 0; j < TOWER_IN_PHI; j++) {
          towersHCAL[i][j].zeroOut();
        }
      }
    };
    inline void setIdx(int idx) { idx_ = idx; };

    // get members
    inline float getIdx() const { return idx_; };
    inline towerHCAL& getTowerHCAL(int iEta, int iPhi) { return towersHCAL[iEta][iPhi]; };
  };

  /*******************************************************************/

  /* 
   * card class: represents one RCT card. Each card has five 3x4 regions and one 2x4 region,
   *             which is represented by a 3x4 region with its third row zero'd out.
   *             idx 0-35: odd values of cardIdx span eta = 0 to eta = 1.41 
   *                       even values of cardIdx span eta = -1.41 to eta = 0
   *             The realEta and realPhi arrays store the (eta, phi) of the center of the towers.
   */

  class card {
  private:
    int idx_ = -1;
    region3x4 card3x4Regions[N_REGIONS_PER_CARD];
    towers3x4 card3x4Towers[N_REGIONS_PER_CARD];

  public:
    // constructor
    card() {
      idx_ = -1;
      for (int i = 0; i < N_REGIONS_PER_CARD; i++) {
        card3x4Regions[i].setIdx(i);
        card3x4Regions[i].zeroOut();
        card3x4Towers[i].setIdx(i);
        card3x4Towers[i].zeroOut();
      }
    };

    // copy constructor
    card(const card& other) {
      idx_ = other.idx_;
      for (int i = 0; i < N_REGIONS_PER_CARD; i++) {
        card3x4Regions[i] = other.card3x4Regions[i];
        card3x4Towers[i] = other.card3x4Towers[i];
      }
    };

    // overload operator= to use copy constructor
    card operator=(const card& other) {
      const card& newCard(other);
      return newCard;
    };

    // set members
    inline void setIdx(int idx) { idx_ = idx; };
    inline void zeroOut() {
      for (int i = 0; i < N_REGIONS_PER_CARD; i++) {
        card3x4Regions[i].zeroOut();
        card3x4Towers[i].zeroOut();
      };
    };

    // get members
    inline float getIdx() const { return idx_; };
    inline region3x4& getRegion3x4(int idx) {
      assert(idx < N_REGIONS_PER_CARD);
      return card3x4Regions[idx];
    }
    inline towers3x4& getTowers3x4(int idx) {
      assert(idx < N_REGIONS_PER_CARD);
      return card3x4Towers[idx];
    }
  };

  /*******************************************************************/

  /*
   *  Crystal class for RCT
   */

  class crystal {
  public:
    ap_uint<10> energy;

    crystal() {
      energy = 0;
      //    timing = 0;
    }

    crystal(ap_uint<10> energy) {  // To-do: add timing information
      this->energy = energy;
      //    this->timing = 0;
    }

    crystal& operator=(const crystal& rhs) {
      energy = rhs.energy;
      //    timing = rhs.timing;
      return *this;
    }
  };

  /*
  * crystalMax class for RCT
  */
  class crystalMax {
  public:
    ap_uint<10> energy;
    uint8_t phiMax;
    uint8_t etaMax;

    crystalMax() {
      energy = 0;
      phiMax = 0;
      etaMax = 0;
    }

    crystalMax& operator=(const crystalMax& rhs) {
      energy = rhs.energy;
      phiMax = rhs.phiMax;
      etaMax = rhs.etaMax;
      return *this;
    }
  };

  class ecaltp_t {
  public:
    ap_uint<10> energy;
    ap_uint<5> phi;
    ap_uint<5> eta;
  };

  class etaStrip_t {
  public:
    ecaltp_t cr0;
    ecaltp_t cr1;
    ecaltp_t cr2;
    ecaltp_t cr3;
    ecaltp_t cr4;
    ecaltp_t cr5;
    ecaltp_t cr6;
    ecaltp_t cr7;
    ecaltp_t cr8;
    ecaltp_t cr9;
    ecaltp_t cr10;
    ecaltp_t cr11;
    ecaltp_t cr12;
    ecaltp_t cr13;
    ecaltp_t cr14;
    ecaltp_t cr15;
    ecaltp_t cr16;
    ecaltp_t cr17;
    ecaltp_t cr18;
    ecaltp_t cr19;
  };

  class ecalRegion_t {
  public:
    etaStrip_t etaStrip0;
    etaStrip_t etaStrip1;
    etaStrip_t etaStrip2;
    etaStrip_t etaStrip3;
    etaStrip_t etaStrip4;
    etaStrip_t etaStrip5;
    etaStrip_t etaStrip6;
    etaStrip_t etaStrip7;
    etaStrip_t etaStrip8;
    etaStrip_t etaStrip9;
    etaStrip_t etaStrip10;
    etaStrip_t etaStrip11;
    etaStrip_t etaStrip12;
    etaStrip_t etaStrip13;
    etaStrip_t etaStrip14;
  };

  class etaStripPeak_t {
  public:
    ecaltp_t pk0;
    ecaltp_t pk1;
    ecaltp_t pk2;
    ecaltp_t pk3;
    ecaltp_t pk4;
    ecaltp_t pk5;
    ecaltp_t pk6;
    ecaltp_t pk7;
    ecaltp_t pk8;
    ecaltp_t pk9;
    ecaltp_t pk10;
    ecaltp_t pk11;
    ecaltp_t pk12;
    ecaltp_t pk13;
    ecaltp_t pk14;
  };

  class tower_t {
  public:
    ap_uint<16> data;

    tower_t() { data = 0; }
    tower_t& operator=(const tower_t& rhs) {
      data = rhs.data;
      return *this;
    }

    tower_t(ap_uint<12> et, ap_uint<4> hoe) { data = (et) | (((ap_uint<16>)hoe) << 12); }

    ap_uint<12> et() { return (data & 0xFFF); }
    ap_uint<4> hoe() { return ((data >> 12) & 0xF); }

    float getEt() { return (float)et() * ECAL_LSB; }

    operator uint16_t() { return (uint16_t)data; }

    // Only for ECAL towers! Apply calibration and modify the et() value.
    void applyCalibration(float factor) {
      // Get the new pT as a float
      float newEt = getEt() * factor;

      // Convert the new pT to an unsigned int (16 bits so we can take the logical OR with the bit mask later)
      ap_uint<16> newEt_uint = (ap_uint<16>)(int)(newEt * 8.0);
      // Make sure the first four bits are zero
      newEt_uint = (newEt_uint & 0x0FFF);

      // Modify 'data'
      ap_uint<16> bitMask = 0xF000;  // last twelve digits are zero
      data = (data & bitMask);       // zero out the last twelve digits
      data = (data | newEt_uint);    // write in the new ET
    }
    /*
     * For towers: Calculate H/E ratio given the ECAL and HCAL energies and modify the hoe() value.
     */
    void getHoverE(ap_uint<12> ECAL, ap_uint<12> HCAL_inHcalConvention) {
      // Convert HCAL ET to ECAL ET convention
      ap_uint<12> HCAL = convertHcalETtoEcalET(HCAL_inHcalConvention);
      ap_uint<4> hoeOut;
      ap_uint<1> hoeLSB = 0;
      ap_uint<4> hoe = 0;
      ap_uint<12> A;
      ap_uint<12> B;

      A = (ECAL > HCAL) ? ECAL : HCAL;
      B = (ECAL > HCAL) ? HCAL : ECAL;

      if (ECAL == 0 || HCAL == 0 || HCAL >= ECAL)
        hoeLSB = 0;
      else
        hoeLSB = 1;
      if (A > B) {
        if (A > 2 * B)
          hoe = 0x1;
        if (A > 4 * B)
          hoe = 0x2;
        if (A > 8 * B)
          hoe = 0x3;
        if (A > 16 * B)
          hoe = 0x4;
        if (A > 32 * B)
          hoe = 0x5;
        if (A > 64 * B)
          hoe = 0x6;
        if (A > 128 * B)
          hoe = 0x7;
      }
      hoeOut = hoeLSB | (hoe << 1);
      ap_uint<16> hoeOutLong =
          ((((ap_uint<16>)hoeOut) << 12) | 0x0000);  // e.g. 0b ____ 0000 0000 0000 where ___ are the hoe digits
      // Take the logical OR to preserve the saturation and tower ET bits
      ap_uint<16> bitMask = 0x0FFF;  // 0000 1111 1111 1111 : zero-out the HoE bits
      data = (data & bitMask);       // zero-out the HoE bits
      data = (data | hoeOutLong);    // write in the new HoE bits
    }
  };

  class clusterInfo {
  public:
    ap_uint<10> seedEnergy;
    ap_uint<15> energy;
    ap_uint<15> et5x5;
    ap_uint<15> et2x5;
    ap_uint<5> phiMax;
    ap_uint<5> etaMax;
    ap_uint<2> brems;

    clusterInfo() {
      seedEnergy = 0;
      energy = 0;
      et5x5 = 0;
      et2x5 = 0;
      phiMax = 0;
      etaMax = 0;
      brems = 0;
    }

    clusterInfo& operator=(const clusterInfo& rhs) {
      seedEnergy = rhs.seedEnergy;
      energy = rhs.energy;
      et5x5 = rhs.et5x5;
      et2x5 = rhs.et2x5;
      phiMax = rhs.phiMax;
      etaMax = rhs.etaMax;
      brems = rhs.brems;
      return *this;
    }
  };

  //--------------------------------------------------------//

  /*
  * Cluster class for RCT.
  */
  class Cluster {
  public:
    ap_uint<28> data;
    int regionIdx;       // region is 0 through 4 in barrel, 5 for overlap in barrel+endcap
    float calib;         //  ECAL energy calibration factor
    ap_uint<2> brems;    // brems flag
    ap_uint<15> et5x5;   // et5x5 sum in towers around cluster
    ap_uint<15> et2x5;   // et2x5 sum in towers around cluster
    bool is_ss;          // shower shape flag
    bool is_looseTkss;   // loose Tk shower shape flag
    bool is_iso;         // isolation flag (not computed until GCT)
    bool is_looseTkiso;  // isolation flag (not computed until GCT)

    Cluster() {
      data = 0;
      regionIdx = -1;
      calib = 1.0;
      brems = 0;
      et5x5 = 0;
      et2x5 = 0;
      is_ss = false;
      is_looseTkss = false;
      is_iso = false;
      is_looseTkiso = false;
    }

    Cluster(ap_uint<12> clusterEnergy,
            ap_uint<5> towerEta,
            ap_uint<2> towerPhi,
            ap_uint<3> clusterEta,
            ap_uint<3> clusterPhi,
            ap_uint<3> satur,
            ap_uint<15> clusterEt5x5 = 0,
            ap_uint<15> clusterEt2x5 = 0,
            ap_uint<2> clusterBrems = 0,
            bool cluster_is_ss = false,
            bool cluster_is_looseTkss = false,
            bool cluster_is_iso = false,
            bool cluster_is_looseTkiso = false,
            int clusterRegionIdx = 0) {
      data = (clusterEnergy) | (((ap_uint<32>)towerEta) << 12) | (((ap_uint<32>)towerPhi) << 17) |
             (((ap_uint<32>)clusterEta) << 19) | (((ap_uint<32>)clusterPhi) << 22) | (((ap_uint<32>)satur) << 25);
      regionIdx = clusterRegionIdx, et5x5 = clusterEt5x5;
      et2x5 = clusterEt2x5;
      brems = clusterBrems;
      is_ss = cluster_is_ss;
      is_looseTkss = cluster_is_looseTkss;
      is_iso = cluster_is_iso;
      is_looseTkiso = cluster_is_looseTkiso;
    }

    Cluster& operator=(const Cluster& rhs) {
      data = rhs.data;
      regionIdx = rhs.regionIdx;
      calib = rhs.calib;
      brems = rhs.brems;
      et5x5 = rhs.et5x5;
      et2x5 = rhs.et2x5;
      is_ss = rhs.is_ss;
      is_looseTkss = rhs.is_looseTkss;
      is_iso = rhs.is_iso;
      is_looseTkiso = rhs.is_looseTkiso;
      return *this;
    }

    void setRegionIdx(int regIdx) { regionIdx = regIdx; }  // Newly added

    ap_uint<12> clusterEnergy() const { return (data & 0xFFF); }
    ap_uint<5> towerEta() const { return ((data >> 12) & 0x1F); }  // goes from 0 to 3 (need region for full info)
    ap_uint<2> towerPhi() const { return ((data >> 17) & 0x3); }
    ap_uint<3> clusterEta() const { return ((data >> 19) & 0x7); }
    ap_uint<3> clusterPhi() const { return ((data >> 22) & 0x7); }
    ap_uint<3> satur() const { return ((data >> 25) & 0x7); }
    ap_uint<15> uint_et2x5() const { return et2x5; }
    ap_uint<15> uint_et5x5() const { return et5x5; }

    operator uint32_t() const { return (ap_uint<28>)data; }
    int region() const { return regionIdx; }  // Newly added
    int getBrems() const { return (int)brems; }
    float getCalib() const { return (float)calib; }
    float getPt() const { return ((float)clusterEnergy() * ECAL_LSB); }  // Return pT as a float
    float getEt5x5() const { return ((float)et5x5 * ECAL_LSB); }         // Return ET5x5 as a float
    float getEt2x5() const { return ((float)et2x5 * ECAL_LSB); }         // Return ET2x5 as a float

    int towerEtaInCard() { return ((int)(region() * TOWER_IN_ETA) + towerEta()); }

    bool getIsSS() { return is_ss; }
    bool getIsLooseTkss() { return is_looseTkss; }
    bool getIsIso() { return is_iso; }
    bool getIsLooseTkIso() { return is_looseTkiso; }

    /*
      * Apply calibration (float) to the pT in-place.
      */
    void applyCalibration(float factor) {
      float newPt = getPt() * factor;
      // Convert the new pT to an unsigned int, 28 bits to take the logical OR with the mask
      ap_uint<28> newPt_uint = (ap_uint<28>)(int)(newPt / ECAL_LSB);
      // Make sure that the new pT only takes up the last twelve bits
      newPt_uint = (newPt_uint & 0x0000FFF);

      // Modify 'data'
      ap_uint<28> bitMask = 0xFFFF000;  // last twelve digits are zero
      data = (data & bitMask);          // zero out the last twelve digits
      data = (data | newPt_uint);       // write in the new ET
    }

    // Get Cluster crystal iEta from card number, region, tower eta, and cluster eta indices
    const int crystaliEtaFromCardRegionInfo(int cc) {
      int crystalEta_in_card =
          ((region() * TOWER_IN_ETA * CRYSTALS_IN_TOWER_ETA) + (towerEta() * CRYSTALS_IN_TOWER_ETA) + clusterEta());
      if ((cc % 2) == 1) {
        return (getCard_refCrystal_iEta(cc) + crystalEta_in_card);
      } else {
        // if card is even (negative eta)
        return (getCard_refCrystal_iEta(cc) - crystalEta_in_card);
      }
    }

    // Get Cluster crystal iPhi from card number, region, tower eta, and cluster phi indices
    const int crystaliPhiFromCardRegionInfo(int cc) {
      int crystalPhi_in_card = (towerPhi() * CRYSTALS_IN_TOWER_PHI) + clusterPhi();
      if ((cc % 2) == 1) {
        // if card is odd (positive eta)
        return (getCard_refCrystal_iPhi(cc) + crystalPhi_in_card);
      } else {
        // if card is even (negative eta)
        return (getCard_refCrystal_iPhi(cc) - crystalPhi_in_card);
      }
    }

    // Get real eta
    const float realEta(int cc) {
      float size_cell = 2 * ECAL_eta_range / (CRYSTALS_IN_TOWER_ETA * n_towers_Eta);
      return crystaliEtaFromCardRegionInfo(cc) * size_cell - ECAL_eta_range + half_crystal_size;
    }

    // Get real phi
    const float realPhi(int cc) {
      float size_cell = 2 * M_PI / (CRYSTALS_IN_TOWER_PHI * n_towers_Phi);
      return crystaliPhiFromCardRegionInfo(cc) * size_cell - M_PI + half_crystal_size;
    }

    // Print info
    void printClusterInfo(int cc, std::string description = "") {
      std::cout << "[Print Cluster class info:] [" << description << "]: "
                << "card " << cc << ", "
                << "et (float): " << getPt() << ", "
                << "eta: " << realEta(cc) << ", "
                << "phi: " << realPhi(cc) << std::endl;
    }
  };

  /*
  * Compare the ET of two clusters (pass this to std::sort to get clusters sorted in decreasing ET).
  */
  inline bool compareClusterET(const Cluster& lhs, const Cluster& rhs) {
    return (lhs.clusterEnergy() > rhs.clusterEnergy());
  }

  /*******************************************************************/
  /* RCT helper functions                                            */
  /*******************************************************************/
  ecalRegion_t initStructure(crystal temporary[CRYSTAL_IN_ETA][CRYSTAL_IN_PHI]);
  ecaltp_t bestOf2(const ecaltp_t ecaltp0, const ecaltp_t ecaltp1);
  ecaltp_t getPeakBin20N(const etaStrip_t etaStrip);
  crystalMax getPeakBin15N(const etaStripPeak_t etaStrip);
  void getECALTowersEt(crystal tempX[CRYSTAL_IN_ETA][CRYSTAL_IN_PHI], ap_uint<12> towerEt[12]);
  clusterInfo getClusterPosition(const ecalRegion_t ecalRegion);
  Cluster packCluster(ap_uint<15>& clusterEt, ap_uint<5>& etaMax_t, ap_uint<5>& phiMax_t);
  void removeClusterFromCrystal(crystal temp[CRYSTAL_IN_ETA][CRYSTAL_IN_PHI],
                                ap_uint<5> seed_eta,
                                ap_uint<5> seed_phi,
                                ap_uint<2> brems);
  clusterInfo getBremsValuesPos(crystal tempX[CRYSTAL_IN_ETA][CRYSTAL_IN_PHI],
                                ap_uint<5> seed_eta,
                                ap_uint<5> seed_phi);
  clusterInfo getBremsValuesNeg(crystal tempX[CRYSTAL_IN_ETA][CRYSTAL_IN_PHI],
                                ap_uint<5> seed_eta,
                                ap_uint<5> seed_phi);
  clusterInfo getClusterValues(crystal tempX[CRYSTAL_IN_ETA][CRYSTAL_IN_PHI], ap_uint<5> seed_eta, ap_uint<5> seed_phi);
  Cluster getClusterFromRegion3x4(crystal temp[CRYSTAL_IN_ETA][CRYSTAL_IN_PHI]);
  void stitchClusterOverRegionBoundary(std::vector<Cluster>& cluster_list, int towerEtaUpper, int towerEtaLower, int cc);

  /*******************************************************************/
  /* Cluster flags                                                   */
  /*******************************************************************/
  inline bool passes_iso(float pt, float iso) {
    bool is_iso = true;
    if (pt < slideIsoPtThreshold) {
      if (!((a0_80 - a1_80 * pt) > iso))
        is_iso = false;
    } else {
      if (iso > a0)
        is_iso = false;
    }
    if (pt > 130)
      is_iso = true;
    return is_iso;
  }

  inline bool passes_looseTkiso(float pt, float iso) {
    bool is_iso = (b0 + b1 * std::exp(-b2 * pt) > iso);
    if (pt > 130)
      is_iso = true;
    return is_iso;
  }

  inline bool passes_ss(float pt, float ss) {
    bool is_ss = ((c0_ss + c1_ss * std::exp(-c2_ss * pt)) <= ss);
    if (pt > 130)
      is_ss = true;
    return is_ss;
  }

  inline bool passes_looseTkss(float pt, float ss) {
    bool is_ss = ((e0_looseTkss - e1_looseTkss * std::exp(-e2_looseTkss * pt)) <= ss);
    if (pt > 130)
      is_ss = true;
    return is_ss;
  }

  /*******************************************************************/
  /* GCT classes.                                                    */
  /*******************************************************************/

  class RCTcluster_t {
  public:
    ap_uint<12> et;
    ap_uint<5> towEta;  // goes from 0 to 17 (vs. class Cluster in the RCT emulator)
    ap_uint<2> towPhi;
    ap_uint<3> crEta;
    ap_uint<3> crPhi;

    ap_uint<12> iso;
    ap_uint<15> et2x5;
    ap_uint<15> et5x5;
    bool is_ss;
    bool is_looseTkss;
    bool is_iso;
    bool is_looseTkiso;
    ap_uint<2> brems;

    int nGCTCard;
  };

  class RCTtower_t {
  public:
    ap_uint<12> et;
    ap_uint<4> hoe;
    // For CMSSW outputs, not firmware
    ap_uint<12> ecalEt;
    ap_uint<12> hcalEt;
  };

  class RCTtoGCTfiber_t {
  public:
    RCTtower_t RCTtowers[N_RCTTOWERS_FIBER];
    RCTcluster_t RCTclusters[N_RCTCLUSTERS_FIBER];
  };

  class RCTcard_t {
  public:
    RCTtoGCTfiber_t RCTtoGCTfiber[N_RCTGCT_FIBERS];
  };

  class GCTcard_t {
  public:
    RCTcard_t RCTcardEtaPos[N_RCTCARDS_PHI];
    RCTcard_t RCTcardEtaNeg[N_RCTCARDS_PHI];
  };

  class GCTcluster_t {
  public:
    bool isPositiveEta;  // is cluster in positive eta side or not
    ap_uint<12> et;
    ap_uint<6> towEta;
    ap_uint<7> towPhi;
    ap_uint<3> crEta;
    ap_uint<3> crPhi;
    ap_uint<12> iso;

    ap_uint<15> et2x5;
    ap_uint<15> et5x5;
    bool is_ss;
    bool is_looseTkss;
    bool is_iso;
    bool is_looseTkiso;

    unsigned int hoe;     // not defined
    unsigned int fb;      // not defined
    unsigned int timing;  // not defined
    ap_uint<2>
        brems;  // 0 if no brems applied, 1 or 2 if brems applied (one for + direction, one for - direction: check firmware)

    float relIso;  // for analyzer only, not firmware
    int nGCTCard;  // for analyzer only, not firmware

    inline float etFloat() const { return ((float)et * ECAL_LSB); }        // Return energy as a float
    inline float isoFloat() const { return ((float)iso * ECAL_LSB); }      // Return energy as a float
    inline float et2x5Float() const { return ((float)et2x5 * ECAL_LSB); }  // Return energy as a float
    inline float et5x5Float() const { return ((float)et5x5 * ECAL_LSB); }  // Return energy as a float
    inline float relIsoFloat() const { return relIso; }                    // relIso is a float already

    inline int etInt() const { return et; }
    inline int isoInt() const { return iso; }

    inline int standaloneWP() const { return (is_iso && is_ss); }
    inline int looseL1TkMatchWP() const { return (is_looseTkiso && is_looseTkss); }
    inline int photonWP() const { return 1; }  // NOTE: NO PHOTON WP

    inline int passesShowerShape() const { return is_ss; }

    /*
       * Initialize from RCTcluster_t.
       */
    void initFromRCTCluster(int iRCTcardIndex, bool isPosEta, const RCTcluster_t& rctCluster) {
      isPositiveEta = isPosEta;

      et = rctCluster.et;
      towEta = rctCluster.towEta;
      if (isPositiveEta) {
        towPhi = rctCluster.towPhi + (iRCTcardIndex * 4);
      } else {
        towPhi = (3 - rctCluster.towPhi) + (iRCTcardIndex * 4);
      }
      crEta = rctCluster.crEta;
      if (isPositiveEta) {
        crPhi = rctCluster.crPhi;
      } else {
        crPhi = (4 - rctCluster.crPhi);
      }
      et2x5 = rctCluster.et2x5;
      et5x5 = rctCluster.et5x5;
      is_ss = rctCluster.is_ss;
      is_looseTkss = rctCluster.is_looseTkss;
      iso = 0;                // initialize: no info from RCT, so set it to null
      relIso = 0;             // initialize: no info from RCT, so set it to null
      is_iso = false;         // initialize: no info from RCT, so set it to false
      is_looseTkiso = false;  // initialize: no info from RCT, so set it to false
      hoe = 0;                // initialize: no info from RCT, so set it to null
      fb = 0;                 // initialize: no info from RCT, so set it to null
      timing = 0;             // initialize: no info from RCT, so set it to null
      brems = rctCluster.brems;
      nGCTCard = rctCluster.nGCTCard;
    }

    /*
       * Get GCT cluster c's iEta (global iEta convention), (0-33*5). Called in realEta().
       */
    int globalClusteriEta(void) const {
      // First get the "iEta/iPhi" in the GCT card. i.e. in the diagram where the barrel is split up into three GCT cards,
      // (iEta, iPhi) = (0, 0) is the top left corner of the GCT card.
      int iEta_in_gctCard;
      if (!isPositiveEta) {
        // Negative eta: c.towEta and c.crEta count outwards from the real eta = 0 center line, so to convert to the barrel diagram global iEta
        // (global iEta = 0 from LHS of page), do (17*5 - 1) minus the GCT value.
        // e.g. If in GCT, a negative card's cluster had iEta = 84, this would be global iEta = 0.
        iEta_in_gctCard =
            ((N_GCTTOWERS_FIBER * CRYSTALS_IN_TOWER_ETA - 1) - ((towEta * CRYSTALS_IN_TOWER_ETA) + crEta));
      } else {
        // c.towEta and c.crEta count outwards from the real eta = 0 center line, so for positive
        // eta we need to add the 17*5 offset so that positive eta 0+epsilon starts at 17*5.
        // e.g. If in GCT, a positive card's cluster had iEta = 0, this would be global iEta = 85.
        // e.g. If in GCT, a positive card's cluster had iEta = 84, this would be global iEta = 169.
        iEta_in_gctCard = ((N_GCTTOWERS_FIBER * CRYSTALS_IN_TOWER_ETA) + ((towEta * CRYSTALS_IN_TOWER_ETA) + crEta));
      }
      // Last, convert to the global iEta/iPhi in the barrel region. For eta these two indices are the same (but this is not true for phi).
      int iEta_in_barrel = iEta_in_gctCard;
      return iEta_in_barrel;
    }

    /* 
       * Get GCT cluster iPhi (global convention, (0-71*5)). Called in realPhi().
       * Use with getPhi_fromCrystaliPhi from Phase2L1RCT.h to convert from GCT cluster to real phi.
       * If returnGlobalGCTiPhi is true (Default value) then return the iPhi in the entire GCT barrel. Otherwise
       * just return the iPhi in the current GCT card.
       */
    int globalClusteriPhi(bool returnGlobalGCTiPhi = true) const {
      int iPhi_in_gctCard = ((towPhi * CRYSTALS_IN_TOWER_PHI) + crPhi);
      // If we should return the global GCT iPhi, get the iPhi offset due to the number of the GCT card
      int iPhi_card_offset = 0;
      if (returnGlobalGCTiPhi) {
        if (nGCTCard == 0)
          iPhi_card_offset = GCTCARD_0_TOWER_IPHI_OFFSET * CRYSTALS_IN_TOWER_PHI;
        else if (nGCTCard == 1)
          iPhi_card_offset = GCTCARD_1_TOWER_IPHI_OFFSET * CRYSTALS_IN_TOWER_PHI;
        else if (nGCTCard == 2)
          iPhi_card_offset = GCTCARD_2_TOWER_IPHI_OFFSET * CRYSTALS_IN_TOWER_PHI;
      }
      // Detector wraps around in phi: modulo number of crystals in phi (n_towers_Phi = 72)
      int iPhi_in_barrel = (iPhi_card_offset + iPhi_in_gctCard) % (n_towers_Phi * CRYSTALS_IN_TOWER_PHI);
      return iPhi_in_barrel;
    }

    /*
       * Each cluster falls in a tower: get this tower iEta in the GCT card (same as global) given the cluster's info. 
       */
    int globalToweriEta(void) const { return (int)(globalClusteriEta() / 5); }

    /*
      * Each cluster falls in a tower: get this tower iPhi in global given the cluster's info. 
      */
    int globalToweriPhi(void) const {
      bool getGlobalIndex = true;
      return (int)(globalClusteriPhi(getGlobalIndex) / 5);
    }

    /*
       * Get tower iPhi IN GCT CARD
       */
    int inCardToweriPhi(void) const {
      bool getGlobalIndex = false;
      return (int)(globalClusteriPhi(getGlobalIndex) / 5);
    }

    /*
       * Get tower iEta IN GCT CARD (conveniently the same as global eta)
       */
    int inCardToweriEta(void) const { return (int)(globalClusteriEta() / 5); }

    /*
       * Get GCT cluster's real eta from global iEta (0-33*5).
       */
    float realEta(void) const {
      float size_cell = 2 * ECAL_eta_range / (CRYSTALS_IN_TOWER_ETA * n_towers_Eta);
      return globalClusteriEta() * size_cell - ECAL_eta_range + half_crystal_size;
    }

    /* 
       * Get GCT cluster's real eta from global iPhi (0-71*5).
       */
    float realPhi(void) const {
      float size_cell = 2 * M_PI / (CRYSTALS_IN_TOWER_PHI * n_towers_Phi);
      return globalClusteriPhi() * size_cell - M_PI + half_crystal_size;
    }

    /* 
       * Return the 4-vector.
       */
    reco::Candidate::PolarLorentzVector p4(void) const {
      return reco::Candidate::PolarLorentzVector(etFloat(), realEta(), realPhi(), 0.);
    }

    /*
       *  Compute relative isolation and set its flags in-place, assuming that the isolation is already computed.
       */
    void setRelIsoAndFlags(void) {
      float relativeIsolationAsFloat = 0;
      if (et > 0) {
        relativeIsolationAsFloat = (isoFloat() / etFloat());
      } else {
        relativeIsolationAsFloat = 0;
      }
      relIso = relativeIsolationAsFloat;
      is_iso = passes_iso(etFloat(), relIso);
      is_looseTkiso = passes_looseTkiso(isoFloat(), relIso);
    }

    /*
       * Create a l1tp2::CaloCrystalCluster object.
       */
    l1tp2::CaloCrystalCluster createCaloCrystalCluster(void) const {
      l1tp2::CaloCrystalCluster caloCrystalCluster(
          p4(),
          etFloat(),      // convert to float
          0,              // supposed to be H over E in the constructor but we do not store/use this information
          relIsoFloat(),  // for consistency with the old emulator, in this field save (iso energy sum)/(cluster energy)
          0,              // DetId seedCrystal
          0,              // puCorrPt
          0,              // brems: not propagated to output (0, 1, or 2 as computed in firmware)
          0,              // et2x2 (not calculated)
          et2x5Float(),   // et2x5 (save float)
          0,              // et3x5 (not calculated)
          et5x5Float(),   // et5x5 (save float)
          standaloneWP(),  // standalone WP
          false,           // electronWP98: not computed
          false,           // photonWP80: not computed
          false,           // electronWP90: not computed
          false,           // looseL1TkMatchWP: not computed
          false            // stage2effMatch: not computed
      );

      // Flags
      std::map<std::string, float> params;
      params["standaloneWP_showerShape"] = is_ss;
      params["standaloneWP_isolation"] = is_iso;
      params["trkMatchWP_showerShape"] = is_looseTkss;
      params["trkMatchWP_isolation"] = is_looseTkiso;
      caloCrystalCluster.setExperimentalParams(params);

      return caloCrystalCluster;
    }

    /*
      * Create a l1t::EGamma object.
      */
    l1t::EGamma createL1TEGamma(void) const {
      // n.b. No photon WP, photonWP() always returns true
      int quality =
          (standaloneWP() * std::pow(2, 0)) + (looseL1TkMatchWP() * std::pow(2, 1)) + (photonWP() * std::pow(2, 2));

      // The constructor zeros out everyhing except the p4()
      l1t::EGamma eg = l1t::EGamma(p4(), etInt(), globalClusteriEta(), globalClusteriPhi(), quality, isoInt());

      // Write in fields that were zerod out
      eg.setRawEt(etInt());                // et as int
      eg.setTowerIEta(globalToweriEta());  // 0-33 in barrel
      eg.setTowerIPhi(globalToweriPhi());  // 0-71 in barrel
      eg.setIsoEt(isoInt());               // raw isolation sum as int
      eg.setShape(passesShowerShape());    // write shower shape flag to this field
      return eg;
    }

    /*
     * Create a l1tp2::DigitizedClusterCorrelator object, with corrTowPhiOffset specifying the offset necessary to correct the tower phi to the region
     * unique to each GCT card.
     */
    l1tp2::DigitizedClusterCorrelator createDigitizedClusterCorrelator(const int corrTowPhiOffset) const {
      return l1tp2::DigitizedClusterCorrelator(
          etFloat(),  // technically we are just multiplying and then dividing again by the LSB
          towEta,
          towPhi - corrTowPhiOffset,
          crEta,
          crPhi,
          hoe,
          is_iso,
          fb,
          timing,
          is_ss,
          brems,
          nGCTCard);
    }

    /*
     * Create a l1tp2::DigitizedClusterGT object
     */
    l1tp2::DigitizedClusterGT createDigitizedClusterGT(bool isValid) const {
      // Constructor arguments take phi, then eta
      return l1tp2::DigitizedClusterGT(isValid, etFloat(), realPhi(), realEta());
    }

    /*
      * Print GCT cluster information.
      */
    void printGCTClusterInfo(std::string description = "") {
      std::cout << "[PrintGCTClusterInfo:] [" << description << "]: "
                << "et (float): " << etFloat() << ", "
                << "eta: " << realEta() << ", "
                << "phi: " << realPhi() << ", "
                << "isPositiveEta " << isPositiveEta << ", "
                << "towEta: " << towEta << ", "
                << "towPhi: " << towPhi << ", "
                << "crEta: " << crEta << ", "
                << "crPhi: " << crPhi << ", "
                << "iso (GeV): " << isoFloat() << ", "
                << "rel iso (unitless float): " << relIsoFloat() << ", "
                << "et2x5 (GeV): " << et2x5Float() << ", "
                << "et5x5 (GeV): " << et5x5Float() << ", "
                << "is_ss: " << is_ss << ", "
                << "is_looseTkss" << is_looseTkss << ", "
                << "is_iso: " << is_iso << ", "
                << "is_looseTkiso: " << is_looseTkiso << ", "
                << "brems: " << brems << std::endl;
    }
  };

  class GCTtower_t {
  public:
    ap_uint<12> et;
    ap_uint<4> hoe;
    ap_uint<2> fb;  // not defined yet in emulator
    // For CMSSW outputs, not firmware
    ap_uint<12> ecalEt;
    ap_uint<12> hcalEt;

    inline float totalEtFloat() const {
      return ((float)et * ECAL_LSB);
    }  // Return total energy as a float (assuming the energy uses the ECAL LSB convention)
    inline float ecalEtFloat() const { return ((float)ecalEt * ECAL_LSB); }  // Return ECAL energy as a float
    inline float hcalEtFloat() const {
      return ((float)hcalEt * HCAL_LSB);
    }  // Return HCAL energy as a float, use HCAL LSB

    /*
       * Initialize from RCTtower_t.
       */
    void initFromRCTTower(const RCTtower_t& rctTower) {
      et = rctTower.et;
      hoe = rctTower.hoe;
      ecalEt = rctTower.ecalEt;
      hcalEt = rctTower.hcalEt;
    }

    /*
      * Correlator fiber convention -> Global GCT convention
      * Get tower's global (iEta) from the GCTCorrFiber index [0, 64) and the tower's postion in the fiber [0, 17).
      * Recall that GCTCorrFiber is [0, 32) for negative eta and [32, 64) for positive eta. The tower's position in the fiber [0, 17)
      * always counts outwards from real eta = 0.
      * Use in conjunction with (float) getTowerEta_fromAbsID(int id) from Phase2L1RCT.h to get a tower's real eta.
      */
    int globalToweriEta(unsigned int nGCTCard, unsigned int gctCorrFiberIdx, unsigned int posInFiber) {
      (void)nGCTCard;                                                        // not needed
      bool isTowerInPositiveEta = (gctCorrFiberIdx < N_GCTPOSITIVE_FIBERS);  // N_GCTPOSITIVE_FIBERS = 32
      int global_toweriEta;
      if (isTowerInPositiveEta) {
        global_toweriEta = (N_GCTTOWERS_FIBER + posInFiber);  // N_GCTTOWERS_FIBER = 17
      } else {
        // e.g. For negative eta, posInFiber = 0 is at real eta = 0, and global tower iEta is 17 - 1 - 0 = 16
        // posInFiber = 16 is at real eta = -1.4841, and global tower iEta is 17 - 1 - 16 = 0.
        global_toweriEta = (N_GCTTOWERS_FIBER - 1 - posInFiber);
      }
      return global_toweriEta;
    }

    /*
       * Correlator fiber convention -> Global GCT convention
      * Get tower's global (iPhi) from the GCT card number (0, 1, 2), and the GCTCorrFiber index [0, 64).
      * GCTCorrFiber is [0, 32) for negative eta and [32, 64) for positive eta. In the phi direction, fiber index #0 has the same phi 
      * as fiber index #32, so only the (fiber index modulo 32) matters for the phi direction. 
      * The tower's position in the fiber doesn't matter; in each fiber the phi is the same. 
      * Use in conjunction with (float) getTowerPhi_fromAbsID(int id) from Phase2L1RCT.h to get a tower's real phi.
      */
    int globalToweriPhi(unsigned int nGCTCard, unsigned int gctCorrFiberIdx, unsigned int posInFiber) {
      (void)posInFiber;                                                           // not needed
      unsigned int effectiveFiberIdx = (gctCorrFiberIdx % N_GCTPOSITIVE_FIBERS);  // N_GCTPOSITIVE_FIBERS = 32
      int toweriPhi_card_offset = 0;
      if (nGCTCard == 0)
        toweriPhi_card_offset = GCTCARD_0_TOWER_IPHI_OFFSET;
      else if (nGCTCard == 1)
        toweriPhi_card_offset = GCTCARD_1_TOWER_IPHI_OFFSET;
      else if (nGCTCard == 2)
        toweriPhi_card_offset = GCTCARD_2_TOWER_IPHI_OFFSET;

      int global_tower_iPhi = (toweriPhi_card_offset + effectiveFiberIdx) %
                              (n_towers_Phi);  //  as explained above, effectiveFiberIdx is [0, 32). n_towers_Phi = 72
      return global_tower_iPhi;
    }

    /*
     * For fulltowers that are indexed by GCT local index: eta
     */
    int globalToweriEtaFromGCTcardiEta(int gctCard_tower_iEta) {
      int global_iEta = gctCard_tower_iEta;
      return global_iEta;
    }

    /*
     * For fulltowers that are indexed by GCT local index: phi. Very similar to globalToweriPhi function but keep them separate for clarity.
     */
    int globalToweriPhiFromGCTcardiPhi(unsigned int nGCTCard, int gctCard_tower_iPhi) {
      assert(nGCTCard <= 2);  // Make sure the card number is valid
      int toweriPhi_card_offset = 0;
      if (nGCTCard == 0)
        toweriPhi_card_offset = GCTCARD_0_TOWER_IPHI_OFFSET;
      else if (nGCTCard == 1)
        toweriPhi_card_offset = GCTCARD_1_TOWER_IPHI_OFFSET;
      else if (nGCTCard == 2)
        toweriPhi_card_offset = GCTCARD_2_TOWER_IPHI_OFFSET;

      int global_iPhi = (toweriPhi_card_offset + gctCard_tower_iPhi) % (n_towers_Phi);  //   n_towers_Phi = 72
      return global_iPhi;
    }

    /* 
       * Method to create a l1tp2::CaloTower object from the fiber and tower-in-fiber indices. 
       * nGCTCard (0, 1, 2) is needed to determine the absolute eta/phi.
       * iFiber and iTowerInFiber are the indices of the tower in the card, e.g. GCTinternal.GCTCorrFiber[iFiber].GCTtowers[iTowerInFiber]
       */
    l1tp2::CaloTower createCaloTowerFromFiberIdx(int nGCTCard, int iFiber, int iTowerInFiber) {
      l1tp2::CaloTower l1CaloTower;
      l1CaloTower.setEcalTowerEt(ecalEtFloat());  // float: ECAL divide by 8.0
      l1CaloTower.setHcalTowerEt(hcalEtFloat());  // float: HCAL multiply by LSB
      int global_tower_iEta = globalToweriEta(nGCTCard, iFiber, iTowerInFiber);
      int global_tower_iPhi = globalToweriPhi(nGCTCard, iFiber, iTowerInFiber);
      l1CaloTower.setTowerIEta(global_tower_iEta);
      l1CaloTower.setTowerIPhi(global_tower_iPhi);
      l1CaloTower.setTowerEta(getTowerEta_fromAbsID(global_tower_iEta));
      l1CaloTower.setTowerPhi(getTowerPhi_fromAbsID(global_tower_iPhi));
      return l1CaloTower;
    }

    /*
     * Method to create a l1tp2::CaloTower object from the global tower ieta and iphi.
     */
    l1tp2::CaloTower createFullTowerFromCardIdx(int nGCTCard, int gctCard_tower_iEta, int gctCard_tower_iPhi) {
      l1tp2::CaloTower l1CaloTower;
      // Store total Et (HCAL+ECAL) in the ECAL Et member
      l1CaloTower.setEcalTowerEt(totalEtFloat());
      int global_tower_iEta = globalToweriEtaFromGCTcardiEta(gctCard_tower_iEta);
      int global_tower_iPhi = globalToweriPhiFromGCTcardiPhi(nGCTCard, gctCard_tower_iPhi);
      l1CaloTower.setTowerIEta(global_tower_iEta);
      l1CaloTower.setTowerIPhi(global_tower_iPhi);
      l1CaloTower.setTowerEta(getTowerEta_fromAbsID(global_tower_iEta));
      l1CaloTower.setTowerPhi(getTowerPhi_fromAbsID(global_tower_iPhi));
      return l1CaloTower;
    }

    /*
     * Method to create a l1tp2::DigitizedTowerCorrelator, from the GCT card number, the fiber index *inside the GCT card* (excluding overlap region),
     * and the index of the tower inside the fiber.
     */
    l1tp2::DigitizedTowerCorrelator createDigitizedTowerCorrelator(unsigned int indexCard,
                                                                   unsigned int indexFiber,
                                                                   unsigned int indexTower) {
      return l1tp2::DigitizedTowerCorrelator(totalEtFloat(), hoe, fb, indexCard, indexFiber, indexTower);
    }

    /*
     * Print GCTtower_t tower information.
     */
    void printGCTTowerInfoFromGlobalIdx(int global_tower_iEta, int global_tower_iPhi, std::string description = "") {
      std::cout << "[Print GCTtower_t class info from global idx:] [" << description << "]: "
                << "total et (float): " << totalEtFloat() << ", "
                << "ecal et (float): " << ecalEtFloat() << ", "
                << "hcal et (float): " << hcalEtFloat() << ", "
                << "fb: " << fb << ", "
                << "global tower ieta: " << global_tower_iEta << ", "
                << "global tower iphi: " << global_tower_iPhi << ", "
                << "eta: " << getTowerEta_fromAbsID(global_tower_iEta) << ", "
                << "phi: " << getTowerPhi_fromAbsID(global_tower_iPhi) << std::endl;
    }
  };

  class GCTCorrfiber_t {
  public:
    GCTtower_t GCTtowers[N_GCTTOWERS_FIBER];
    GCTcluster_t GCTclusters[N_GCTCLUSTERS_FIBER];
  };

  class GCTtoCorr_t {
  public:
    GCTCorrfiber_t GCTCorrfiber[N_GCTCORR_FIBERS];
  };

  class GCTinternal_t {
  public:
    GCTCorrfiber_t GCTCorrfiber[N_GCTINTERNAL_FIBERS];

    void computeClusterIsolationInPlace(int nGCTCard) {
      for (unsigned int iFiber = 0; iFiber < N_GCTINTERNAL_FIBERS; iFiber++) {
        for (unsigned int iCluster = 0; iCluster < N_GCTCLUSTERS_FIBER; iCluster++) {
          // We will only save clusters with > 0 GeV, so only need to do this for clusters with >0 energy
          if (GCTCorrfiber[iFiber].GCTclusters[iCluster].et == 0) {
            GCTCorrfiber[iFiber].GCTclusters[iCluster].iso = 0;
            continue;
          }

          ap_uint<12> uint_isolation = 0;

          // do not add the GCT card off-set, so we remain in the gct local card iEta/iPhi
          int toweriEta_in_GCT_card = GCTCorrfiber[iFiber].GCTclusters[iCluster].inCardToweriEta();
          int toweriPhi_in_GCT_card = GCTCorrfiber[iFiber].GCTclusters[iCluster].inCardToweriPhi();

          // If cluster is in the overlap region, do not compute isolation
          bool inOverlapWithAnotherGCTCard = (((toweriPhi_in_GCT_card >= 0) && (toweriPhi_in_GCT_card < 4)) ||
                                              ((toweriPhi_in_GCT_card >= 28) && (toweriPhi_in_GCT_card < 32)));
          if (inOverlapWithAnotherGCTCard) {
            GCTCorrfiber[iFiber].GCTclusters[iCluster].iso = 0;
            continue;
          }

          // Size 5x5 in towers: include the overlap-region-between-GCT-cards-if-applicable. In eta direction, the min and max towers (inclusive!) are:
          int isoWindow_toweriEta_in_GCT_card_min = std::max(0, toweriEta_in_GCT_card - 2);
          int isoWindow_toweriEta_in_GCT_card_max = std::min(toweriEta_in_GCT_card + 2, N_GCTETA - 1);  // N_GCTETA = 34
          // e.g. if our window is centered at tower_iEta = 5, we want to sum towers_iEta 3, 4, (5), 6, 7, inclusive
          // e.g. if our window is near the boundary, tower_iEta = 32, we want to sum towers_iEta 30, 31, (32), 33
          // inclusive (but there are only N_GCTETA = 34 towers, so we stop at tower_iEta = 33)

          // in phi direction, the min and max towers (inclusive!) are:
          int isoWindow_toweriPhi_in_GCT_card_min = std::max(0, toweriPhi_in_GCT_card - 2);
          int isoWindow_toweriPhi_in_GCT_card_max = std::min(toweriPhi_in_GCT_card + 2, N_GCTPHI - 1);

          // Keep track of the number of towers we summed over
          int nTowersSummed = 0;

          // First add any nearby clusters to the isolation
          for (unsigned int candFiber = 0; candFiber < N_GCTINTERNAL_FIBERS; candFiber++) {
            for (unsigned int candCluster = 0; candCluster < N_GCTCLUSTERS_FIBER; candCluster++) {
              // Do not double-count the cluster we are calculating the isolation for
              if (!((candFiber == iFiber) && (candCluster == iCluster))) {
                // Only consider clusters with et > 0 for isolation sum
                if (GCTCorrfiber[candFiber].GCTclusters[candCluster].et > 0) {
                  // Get the candidate cluster's tower iEta and iPhi in GCT card
                  int candidate_toweriEta = GCTCorrfiber[candFiber].GCTclusters[candCluster].inCardToweriEta();
                  int candidate_toweriPhi = GCTCorrfiber[candFiber].GCTclusters[candCluster].inCardToweriPhi();

                  // If the tower that the candidate cluster is in, is within a 5x5 window, add the candidate cluster energy's to the isolation as a proxy for the ECAL energy
                  if (((candidate_toweriEta >= isoWindow_toweriEta_in_GCT_card_min) &&
                       (candidate_toweriEta <= isoWindow_toweriEta_in_GCT_card_max)) &&
                      ((candidate_toweriPhi >= isoWindow_toweriPhi_in_GCT_card_min) &&
                       (candidate_toweriPhi <= isoWindow_toweriPhi_in_GCT_card_max))) {
                    uint_isolation += GCTCorrfiber[candFiber].GCTclusters[candCluster].et;
                  }
                }
              }
            }
          }

          //  From "tower index in GCT card", get which fiber it is in (out of 64 fibers), and which tower it is inside the fiber (out of 17 towers)
          for (int iEta = isoWindow_toweriEta_in_GCT_card_min; iEta <= isoWindow_toweriEta_in_GCT_card_max; iEta++) {
            for (int iPhi = isoWindow_toweriPhi_in_GCT_card_min; iPhi <= isoWindow_toweriPhi_in_GCT_card_max; iPhi++) {
              nTowersSummed += 1;

              int indexInto64Fibers;
              int indexInto17TowersInFiber;

              bool isTowerInPositiveEta = (iEta >= N_GCTTOWERS_FIBER);
              if (isTowerInPositiveEta) {
                // phi index is simple (e.g. if real phi = +80 degrees, iPhi in GCT = 31)
                indexInto64Fibers = iPhi;
                // if real eta = 1.47, iEta in GCT card = 33. If real eta = 0.0, iEta in GCT = 17, so iEta in fiber = 17%17 = 0.
                indexInto17TowersInFiber = (iEta % 17);
              } else {
                // add offset (e.g. if real phi = +80 degrees, iPhi in GCT = 31, and my index into GCT fibers 31 + 32 = 63)
                indexInto64Fibers = (iPhi + N_GCTPOSITIVE_FIBERS);
                // e.g.  if real eta = 0, iEta innew GCT card = 16, i.e. our index into the GCT fiber is 16-16 = 0
                indexInto17TowersInFiber = (16 - iEta);
              }

              ap_uint<12> ecalEt = GCTCorrfiber[indexInto64Fibers].GCTtowers[indexInto17TowersInFiber].ecalEt;
              uint_isolation += ecalEt;
            }
          }

          // Scale the isolation sum up if we summed over fewer than (5x5) = 25 towers
          float scaleFactor =
              ((float)(N_GCTTOWERS_CLUSTER_ISO_ONESIDE * N_GCTTOWERS_CLUSTER_ISO_ONESIDE) / (float)nTowersSummed);

          uint_isolation = (ap_uint<12>)(((float)uint_isolation) * scaleFactor);

          // Set the iso in the cluster
          GCTCorrfiber[iFiber].GCTclusters[iCluster].iso = uint_isolation;
        }
      }
    }

    void setIsolationInfo(void) {
      for (unsigned int iFiber = 0; iFiber < N_GCTINTERNAL_FIBERS; iFiber++) {
        for (unsigned int iCluster = 0; iCluster < N_GCTCLUSTERS_FIBER; iCluster++) {
          // update the cluster's isolation information
          GCTCorrfiber[iFiber].GCTclusters[iCluster].setRelIsoAndFlags();
        }
      }
    }
  };

  class GCTintTowers_t {
  public:
    GCTtower_t GCTtower[N_GCTETA][N_GCTPHI];

    // Write contents to output CMSSW collection. Note the use of the GCTtower_t method that creates the
    // l1tp2::CaloTower object from the global eta/phi.
    void writeToPFOutput(int nGCTCard, std::unique_ptr<l1tp2::CaloTowerCollection> const& gctFullTowers) {
      for (unsigned int iEta = 0; iEta < N_GCTETA; iEta++) {
        for (unsigned int iPhi = 0; iPhi < N_GCTPHI; iPhi++) {
          GCTtower_t thisFullTower = GCTtower[iEta][iPhi];
          gctFullTowers->push_back(thisFullTower.createFullTowerFromCardIdx(nGCTCard, iEta, iPhi));
        }
      }
    }
  };

  /* For each GCT card (3 of them in total, for barrel + endcap), list the sixteen                
    * RCT cards that fall in them. The first eight are in positive eta, the next                   
    * eight are in negative eta (see figure of one GCT card). The RCT cards themselves             
    * run from 0 to 35 (see RCT figure).                                                          
    * Hard-coded because the GCT cards wrap around the barrel region.                            
    * Used only to convert the RCT emulator outputs to the GCT emulator inputs.                   
    */
  static const unsigned int GCTcardtoRCTcardnumber[N_GCTCARDS][N_RCTCARDS_PHI * 2] = {
      // GCT Card 0
      {11, 13, 15, 17, 19, 21, 23, 25, 10, 12, 14, 16, 18, 20, 22, 24},

      // GCT Card 1
      {23, 25, 27, 29, 31, 33, 35, 1, 22, 24, 26, 28, 30, 32, 34, 0},

      // GCT Card 2
      {35, 1, 3, 5, 7, 9, 11, 13, 34, 0, 2, 4, 6, 8, 10, 12}};

  /*
   * Helper function to monitor l1tp2::CaloTower members.
   */
  inline void printl1tp2TowerInfo(l1tp2::CaloTower thisTower, std::string description = "") {
    std::cout << "[Print l1tp2::CaloTower info:] [" << description << "]: "
              << ".ecalTowerEta() (float): " << thisTower.ecalTowerEt() << ", "
              << ".hcalTowerEta() (float): " << thisTower.hcalTowerEt() << ", "
              << ".towerIEta(): " << thisTower.towerIEta() << ", "
              << ".towerIPhi(): " << thisTower.towerIPhi() << ", "
              << ".towerEta() " << thisTower.towerEta() << ", "
              << ".towerPhi() " << thisTower.towerPhi() << std::endl;
  }

  void algo_top(const GCTcard_t& GCTcard,
                GCTtoCorr_t& GCTtoCorr,
                unsigned int nGCTCard,
                std::unique_ptr<l1tp2::CaloCrystalClusterCollection> const& gctClusters,
                std::unique_ptr<l1tp2::CaloTowerCollection> const& gctTowers,
                std::unique_ptr<l1tp2::CaloTowerCollection> const& gctFullTowers,
                std::unique_ptr<l1t::EGammaBxCollection> const& gctEGammas,
                std::unique_ptr<l1tp2::DigitizedClusterCorrelatorCollection> const& gctDigitizedClustersCorrelator,
                std::unique_ptr<l1tp2::DigitizedTowerCorrelatorCollection> const& gctDigitizedTowersCorrelator,
                std::unique_ptr<l1tp2::DigitizedClusterGTCollection> const& gctDigitizedClustersGT,
                l1tp2::ParametricCalibration calib_);

  GCTinternal_t getClustersTowers(const GCTcard_t& GCTcard, unsigned int nGCTCard);

  void doProximityAndBremsStitching(const RCTcard_t (&inputCards)[N_RCTCARDS_PHI],
                                    RCTcard_t (&outputCards)[N_RCTCARDS_PHI],
                                    int iStartingCard,
                                    bool isPositiveEta);

  GCTcard_t getClustersCombined(const GCTcard_t& GCTcard, unsigned int nGCTCard);

  GCTintTowers_t getFullTowers(const GCTinternal_t& GCTinternal);

  void writeToCorrelatorAndGTOutputs(
      const GCTinternal_t& GCTinternal,
      GCTtoCorr_t& GCTtoCorrOutput,
      std::unique_ptr<l1tp2::CaloCrystalClusterCollection> const& gctClustersOutput,
      std::unique_ptr<l1tp2::CaloTowerCollection> const& gctTowersOutput,
      std::unique_ptr<l1t::EGammaBxCollection> const& gctEGammas,
      std::unique_ptr<l1tp2::DigitizedClusterCorrelatorCollection> const& gctDigitizedClustersCorrelator,
      std::unique_ptr<l1tp2::DigitizedTowerCorrelatorCollection> const& gctDigitizedTowersCorrelator,
      std::unique_ptr<l1tp2::DigitizedClusterGTCollection> const& gctDigitizedClustersGT,
      int nGCTCard,
      int fiberStart,
      int fiberEnd,
      int corrFiberIndexOffset,
      int corrTowPhiOffset);

}  // namespace p2eg

#endif
