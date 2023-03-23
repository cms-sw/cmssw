#ifndef CONDCORE_ALIGNMENTPLUGINS_ALIGNMENTPAYLOADINSPECTORHELPER_H
#define CONDCORE_ALIGNMENTPLUGINS_ALIGNMENTPAYLOADINSPECTORHELPER_H

#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <string>
#include "TH1.h"
#include "TCanvas.h"
#include "TPaveStats.h"
#include "TStyle.h"
#include "TList.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/deltaPhi.h"  // for deltaPhi
#include "DataFormats/Math/interface/Rounding.h"  // for rounding
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define MMDEBUG // uncomment for debugging at compile time
#ifdef MMDEBUG
#include <iostream>
#define COUT std::cout << "MM "
#else
#define COUT edm::LogVerbatim("")
#endif

namespace AlignmentPI {

  // size of the phase-I Tracker APE payload (including both SS + DS modules)
  static const unsigned int phase0size = 19876;
  static const unsigned int phase1size = 20292;
  static const float cmToUm = 10000.f;
  static const float tomRad = 1000.f;

  // method to zero all elements whose difference from 2Pi
  // is less than the tolerance (2*10e-7)
  inline double returnZeroIfNear2PI(const double phi) {
    const double tol = 2.e-7;  // default tolerance 1.e-7 doesn't account for possible variations
    if (cms_rounding::roundIfNear0(std::abs(phi) - 2 * M_PI, tol) == 0.f) {
      return 0.f;
    } else {
      return phi;
    }
  }

  // method to bring back around 0 all elements whose  difference
  // frm 2Pi is is less than the tolerance (in micro-rad)
  inline double trim2PIs(const double phi, const double tolerance = 1.f) {
    if (std::abs((std::abs(phi) - 2 * M_PI) * tomRad) < tolerance) {
      return (std::abs(phi) - 2 * M_PI);
    } else {
      return phi;
    }
  }

  enum coordinate {
    t_x = 1,
    t_y = 2,
    t_z = 3,
    rot_alpha = 4,
    rot_beta = 5,
    rot_gamma = 6,
  };

  // M.M. 2017/09/12
  // As the matrix is symmetric, we map only 6/9 terms
  // More terms for the extended APE can be added to the following methods

  enum index { XX = 1, XY = 2, XZ = 3, YZ = 4, YY = 5, ZZ = 6 };

  enum partitions { INVALID = 0, BPix = 1, FPix = 2, TIB = 3, TID = 4, TOB = 5, TEC = 6 };

  enum class PARTITION {
    BPIX,   // 0 Barrel Pixel
    FPIXp,  // 1 Forward Pixel Plus
    FPIXm,  // 2 Forward Pixel Minus
    TIB,    // 3 Tracker Inner Barrel
    TIDp,   // 4 Tracker Inner Disks Plus
    TIDm,   // 5 Tracker Inner Disks Minus
    TOB,    // 6 Tracker Outer Barrel
    TECp,   // 7 Tracker Endcaps Plus
    TECm,   // 8 Tracker Endcaps Minus
    LAST = TECm
  };

  extern const PARTITION PARTITIONS[(int)PARTITION::LAST + 1];
  const PARTITION PARTITIONS[] = {PARTITION::BPIX,
                                  PARTITION::FPIXp,
                                  PARTITION::FPIXm,
                                  PARTITION::TIB,
                                  PARTITION::TIDp,
                                  PARTITION::TIDm,
                                  PARTITION::TOB,
                                  PARTITION::TECp,
                                  PARTITION::TECm};

  inline std::ostream& operator<<(std::ostream& o, PARTITION x) {
    return o << std::underlying_type<PARTITION>::type(x);
  }

  enum regions {
    BPixL1o,          //0  Barrel Pixel Layer 1 outer
    BPixL1i,          //1  Barrel Pixel Layer 1 inner
    BPixL2o,          //2  Barrel Pixel Layer 2 outer
    BPixL2i,          //3  Barrel Pixel Layer 2 inner
    BPixL3o,          //4  Barrel Pixel Layer 3 outer
    BPixL3i,          //5  Barrel Pixel Layer 3 inner
    BPixL4o,          //6  Barrel Pixel Layer 4 outer
    BPixL4i,          //7  Barrel Pixel Layer 4 inner
    FPixmL1,          //8  Forward Pixel Minus side Disk 1
    FPixmL2,          //9 Forward Pixel Minus side Disk 2
    FPixmL3,          //10 Forward Pixel Minus side Disk 3
    FPixpL1,          //11 Forward Pixel Plus side Disk 1
    FPixpL2,          //12 Forward Pixel Plus side Disk 2
    FPixpL3,          //13 Forward Pixel Plus side Disk 3
    TIBL1Ro,          //14 Inner Barrel Layer 1 Rphi outer
    TIBL1Ri,          //15 Inner Barrel Layer 1 Rphi inner
    TIBL1So,          //16 Inner Barrel Layer 1 Stereo outer
    TIBL1Si,          //17 Inner Barrel Layer 1 Stereo inner
    TIBL2Ro,          //18 Inner Barrel Layer 2 Rphi outer
    TIBL2Ri,          //19 Inner Barrel Layer 2 Rphi inner
    TIBL2So,          //20 Inner Barrel Layer 2 Stereo outer
    TIBL2Si,          //21 Inner Barrel Layer 2 Stereo inner
    TIBL3o,           //22 Inner Barrel Layer 3 outer
    TIBL3i,           //23 Inner Barrel Layer 3 inner
    TIBL4o,           //24 Inner Barrel Layer 4 outer
    TIBL4i,           //25 Inner Barrel Layer 4 inner
    TOBL1Ro,          //26 Outer Barrel Layer 1 Rphi outer
    TOBL1Ri,          //27 Outer Barrel Layer 1 Rphi inner
    TOBL1So,          //28 Outer Barrel Layer 1 Stereo outer
    TOBL1Si,          //29 Outer Barrel Layer 1 Stereo inner
    TOBL2Ro,          //30 Outer Barrel Layer 2 Rphi outer
    TOBL2Ri,          //31 Outer Barrel Layer 2 Rphi inner
    TOBL2So,          //32 Outer Barrel Layer 2 Stereo outer
    TOBL2Si,          //33 Outer Barrel Layer 2 Stereo inner
    TOBL3o,           //34 Outer Barrel Layer 3 outer
    TOBL3i,           //35 Outer Barrel Layer 3 inner
    TOBL4o,           //36 Outer Barrel Layer 4 outer
    TOBL4i,           //37 Outer Barrel Layer 4 inner
    TOBL5o,           //38 Outer Barrel Layer 5 outer
    TOBL5i,           //39 Outer Barrel Layer 5 inner
    TOBL6o,           //40 Outer Barrel Layer 6 outer
    TOBL6i,           //41 Outer Barrel Layer 6 inner
    TIDmR1R,          //42 Inner Disk Minus side Ring 1 Rphi
    TIDmR1S,          //43 Inner Disk Minus side Ring 1 Stereo
    TIDmR2R,          //44 Inner Disk Minus side Ring 2 Rphi
    TIDmR2S,          //45 Inner Disk Minus side Ring 2 Stereo
    TIDmR3,           //46 Inner Disk Minus side Ring 3
    TIDpR1R,          //47 Inner Disk Plus side Ring 1 Rphi
    TIDpR1S,          //48 Inner Disk Plus side Ring 1 Stereo
    TIDpR2R,          //49 Inner Disk Plus side Ring 2 Rphi
    TIDpR2S,          //50 Inner Disk Plus side Ring 2 Stereo
    TIDpR3,           //51 Inner Disk Plus side Ring 3
    TECmR1R,          //52 Endcaps Minus side Ring 1 Rphi
    TECmR1S,          //53 Endcaps Minus side Ring 1 Stereo
    TECmR2R,          //54 Encdaps Minus side Ring 2 Rphi
    TECmR2S,          //55 Endcaps Minus side Ring 2 Stereo
    TECmR3,           //56 Endcaps Minus side Ring 3
    TECmR4,           //57 Endcaps Minus side Ring 4
    TECmR5,           //58 Endcaps Minus side Ring 5
    TECmR6,           //59 Endcaps Minus side Ring 6
    TECmR7,           //60 Endcaps Minus side Ring 7
    TECpR1R,          //61 Endcaps Plus side Ring 1 Rphi
    TECpR1S,          //62 Endcaps Plus side Ring 1 Stereo
    TECpR2R,          //63 Encdaps Plus side Ring 2 Rphi
    TECpR2S,          //64 Endcaps Plus side Ring 2 Stereo
    TECpR3,           //65 Endcaps Plus side Ring 3
    TECpR4,           //66 Endcaps Plus side Ring 4
    TECpR5,           //67 Endcaps Plus side Ring 5
    TECpR6,           //68 Endcaps Plus side Ring 6
    TECpR7,           //67 Endcaps Plus side Ring 7
    StripDoubleSide,  // 70 -- not to be considered
    NUM_OF_REGIONS    // 71 -- default
  };

  /*--------------------------------------------------------------------*/
  inline std::string getStringFromRegionEnum(AlignmentPI::regions e)
  /*--------------------------------------------------------------------*/
  {
    switch (e) {
      case AlignmentPI::BPixL1o:
        return "BPixL1o";
      case AlignmentPI::BPixL1i:
        return "BPixL1i";
      case AlignmentPI::BPixL2o:
        return "BPixL2o";
      case AlignmentPI::BPixL2i:
        return "BPixL2i";
      case AlignmentPI::BPixL3o:
        return "BPixL3o";
      case AlignmentPI::BPixL3i:
        return "BPixL3i";
      case AlignmentPI::BPixL4o:
        return "BPixL4o";
      case AlignmentPI::BPixL4i:
        return "BPixL4i";
      case AlignmentPI::FPixmL1:
        return "FPixmL1";
      case AlignmentPI::FPixmL2:
        return "FPixmL2";
      case AlignmentPI::FPixmL3:
        return "FPixmL3";
      case AlignmentPI::FPixpL1:
        return "FPixpL1";
      case AlignmentPI::FPixpL2:
        return "FPixpL2";
      case AlignmentPI::FPixpL3:
        return "FPixpL3";
      case AlignmentPI::TIBL1Ro:
        return "TIBL1Ro";
      case AlignmentPI::TIBL1Ri:
        return "TIBL1Ri";
      case AlignmentPI::TIBL1So:
        return "TIBL1So";
      case AlignmentPI::TIBL1Si:
        return "TIBL1Si";
      case AlignmentPI::TIBL2Ro:
        return "TIBL2Ro";
      case AlignmentPI::TIBL2Ri:
        return "TIBL2Ri";
      case AlignmentPI::TIBL2So:
        return "TIBL2So";
      case AlignmentPI::TIBL2Si:
        return "TIBL2Si";
      case AlignmentPI::TIBL3o:
        return "TIBL3o";
      case AlignmentPI::TIBL3i:
        return "TIBL3i";
      case AlignmentPI::TIBL4o:
        return "TIBL4o";
      case AlignmentPI::TIBL4i:
        return "TIBL4i";
      case AlignmentPI::TOBL1Ro:
        return "TOBL1Ro";
      case AlignmentPI::TOBL1Ri:
        return "TOBL1Ri";
      case AlignmentPI::TOBL1So:
        return "TOBL1So";
      case AlignmentPI::TOBL1Si:
        return "TOBL1Si";
      case AlignmentPI::TOBL2Ro:
        return "TOBL2Ro";
      case AlignmentPI::TOBL2Ri:
        return "TOBL2Ri";
      case AlignmentPI::TOBL2So:
        return "TOBL2So";
      case AlignmentPI::TOBL2Si:
        return "TOBL2Si";
      case AlignmentPI::TOBL3o:
        return "TOBL3o";
      case AlignmentPI::TOBL3i:
        return "TOBL3i";
      case AlignmentPI::TOBL4o:
        return "TOBL4o";
      case AlignmentPI::TOBL4i:
        return "TOBL4i";
      case AlignmentPI::TOBL5o:
        return "TOBL5o";
      case AlignmentPI::TOBL5i:
        return "TOBL5i";
      case AlignmentPI::TOBL6o:
        return "TOBL6o";
      case AlignmentPI::TOBL6i:
        return "TOBL6i";
      case AlignmentPI::TIDmR1R:
        return "TIDmR1R";
      case AlignmentPI::TIDmR1S:
        return "TIDmR1S";
      case AlignmentPI::TIDmR2R:
        return "TIDmR2R";
      case AlignmentPI::TIDmR2S:
        return "TIDmR2S";
      case AlignmentPI::TIDmR3:
        return "TIDmR3";
      case AlignmentPI::TIDpR1R:
        return "TIDpR1R";
      case AlignmentPI::TIDpR1S:
        return "TIDpR1S";
      case AlignmentPI::TIDpR2R:
        return "TIDpR2R";
      case AlignmentPI::TIDpR2S:
        return "TIDpR2S";
      case AlignmentPI::TIDpR3:
        return "TIDpR3";
      case AlignmentPI::TECmR1R:
        return "TECmR1R";
      case AlignmentPI::TECmR1S:
        return "TECmR1S";
      case AlignmentPI::TECmR2R:
        return "TECmR2R";
      case AlignmentPI::TECmR2S:
        return "TECmR2S";
      case AlignmentPI::TECmR3:
        return "TECmR3";
      case AlignmentPI::TECmR4:
        return "TECmR4";
      case AlignmentPI::TECmR5:
        return "TECmR5";
      case AlignmentPI::TECmR6:
        return "TECmR6";
      case AlignmentPI::TECmR7:
        return "TECmR7";
      case AlignmentPI::TECpR1R:
        return "TECpR1R";
      case AlignmentPI::TECpR1S:
        return "TECpR1S";
      case AlignmentPI::TECpR2R:
        return "TECpR2R";
      case AlignmentPI::TECpR2S:
        return "TECpR2S";
      case AlignmentPI::TECpR3:
        return "TECpR3";
      case AlignmentPI::TECpR4:
        return "TECpR4";
      case AlignmentPI::TECpR5:
        return "TECpR5";
      case AlignmentPI::TECpR6:
        return "TECpR6";
      case AlignmentPI::TECpR7:
        return "TECpR7";
      default:
        edm::LogWarning("LogicError") << "Unknown partition: " << e;
        return "";
    }
  }

  /*--------------------------------------------------------------------*/
  inline bool isBPixOuterLadder(const DetId& detid, const TrackerTopology& tTopo, bool isPhase0)
  /*--------------------------------------------------------------------*/
  {
    // Using TrackerTopology
    // Ladders have a staggered structure
    // Non-flipped ladders are on the outer radius
    // Phase 0: Outer ladders are odd for layer 1,3 and even for layer 2
    // Phase 1: Outer ladders are odd for layer 1,2,3 and even for layer 4
    bool isOuter = false;
    int layer = tTopo.pxbLayer(detid.rawId());
    bool odd_ladder = tTopo.pxbLadder(detid.rawId()) % 2;
    if (isPhase0) {
      if (layer == 2)
        isOuter = !odd_ladder;
      else
        isOuter = odd_ladder;
    } else {
      if (layer == 4)
        isOuter = !odd_ladder;
      else
        isOuter = odd_ladder;
    }
    return isOuter;
  }

  // ancillary struct to manage the topology
  // info in a more compact way

  struct topolInfo {
  private:
    uint32_t m_rawid;
    int m_subdetid;
    int m_layer;
    int m_side;
    int m_ring;
    bool m_isRphi;
    bool m_isDoubleSide;
    bool m_isInternal;

  public:
    void init();
    void fillGeometryInfo(const DetId& detId, const TrackerTopology& tTopo, bool isPhase0);
    AlignmentPI::regions filterThePartition();
    bool sanityCheck();
    void printAll();
    virtual ~topolInfo() {}
  };

  /*--------------------------------------------------------------------*/
  inline void topolInfo::printAll()
  /*--------------------------------------------------------------------*/
  {
    std::cout << " detId:" << m_rawid << " subdetid: " << m_subdetid << " layer: " << m_layer << " side: " << m_side
              << " ring: " << m_ring << " isRphi:" << m_isRphi << " isDoubleSide:" << m_isDoubleSide
              << " isInternal:" << m_isInternal << std::endl;
  }

  /*--------------------------------------------------------------------*/
  inline void topolInfo::init()
  /*--------------------------------------------------------------------*/
  {
    m_rawid = 0;
    m_subdetid = -1;
    m_layer = -1;
    m_side = -1;
    m_ring = -1;
    m_isRphi = false;
    m_isDoubleSide = false;
    m_isInternal = false;
  };

  /*--------------------------------------------------------------------*/
  inline bool topolInfo::sanityCheck()
  /*--------------------------------------------------------------------*/
  {
    if (m_layer == 0 || (m_subdetid == 1 && m_layer > 4) || (m_subdetid == 2 && m_layer > 3)) {
      return false;
    } else {
      return true;
    }
  }
  /*--------------------------------------------------------------------*/
  inline void topolInfo::fillGeometryInfo(const DetId& detId, const TrackerTopology& tTopo, bool isPhase0)
  /*--------------------------------------------------------------------*/
  {
    unsigned int subdetId = static_cast<unsigned int>(detId.subdetId());

    m_rawid = detId.rawId();
    m_subdetid = subdetId;

    if (subdetId == StripSubdetector::TIB) {
      m_layer = tTopo.tibLayer(detId.rawId());
      m_side = tTopo.tibSide(detId.rawId());
      m_isRphi = tTopo.isRPhi(detId.rawId());
      m_isDoubleSide = tTopo.tibIsDoubleSide(detId.rawId());
      m_isInternal = tTopo.tibIsInternalString(detId.rawId());
    } else if (subdetId == StripSubdetector::TOB) {
      m_layer = tTopo.tobLayer(detId.rawId());
      m_side = tTopo.tobSide(detId.rawId());
      m_isRphi = tTopo.isRPhi(detId.rawId());
      m_isDoubleSide = tTopo.tobIsDoubleSide(detId.rawId());
      m_isInternal = tTopo.tobModule(detId.rawId()) % 2;
    } else if (subdetId == StripSubdetector::TID) {
      m_layer = tTopo.tidWheel(detId.rawId());
      m_side = tTopo.tidSide(detId.rawId());
      m_isRphi = tTopo.isRPhi(detId.rawId());
      m_ring = tTopo.tidRing(detId.rawId());
      m_isDoubleSide = tTopo.tidIsDoubleSide(detId.rawId());
      m_isInternal = tTopo.tidModuleInfo(detId.rawId())[0];
    } else if (subdetId == StripSubdetector::TEC) {
      m_layer = tTopo.tecWheel(detId.rawId());
      m_side = tTopo.tecSide(detId.rawId());
      m_isRphi = tTopo.isRPhi(detId.rawId());
      m_ring = tTopo.tecRing(detId.rawId());
      m_isDoubleSide = tTopo.tecIsDoubleSide(detId.rawId());
      m_isInternal = tTopo.tecPetalInfo(detId.rawId())[0];
    } else if (subdetId == PixelSubdetector::PixelBarrel) {
      m_layer = tTopo.pxbLayer(detId.rawId());
      m_isInternal = !AlignmentPI::isBPixOuterLadder(detId, tTopo, isPhase0);
    } else if (subdetId == PixelSubdetector::PixelEndcap) {
      m_layer = tTopo.pxfDisk(detId.rawId());
      m_side = tTopo.pxfSide(detId.rawId());
    } else
      edm::LogWarning("LogicError") << "Unknown subdetid: " << subdetId;
  }

  // ------------ method to assign a partition based on the topology struct info ---------------

  /*--------------------------------------------------------------------*/
  inline AlignmentPI::regions topolInfo::filterThePartition()
  /*--------------------------------------------------------------------*/
  {
    AlignmentPI::regions ret = AlignmentPI::NUM_OF_REGIONS;

    if (m_isDoubleSide) {
      return AlignmentPI::StripDoubleSide;
    }

    // BPix
    if (m_subdetid == 1) {
      switch (m_layer) {
        case 1:
          m_isInternal > 0 ? ret = AlignmentPI::BPixL1o : ret = AlignmentPI::BPixL1i;
          break;
        case 2:
          m_isInternal > 0 ? ret = AlignmentPI::BPixL2o : ret = AlignmentPI::BPixL2i;
          break;
        case 3:
          m_isInternal > 0 ? ret = AlignmentPI::BPixL3o : ret = AlignmentPI::BPixL3i;
          break;
        case 4:
          m_isInternal > 0 ? ret = AlignmentPI::BPixL4o : ret = AlignmentPI::BPixL4i;
          break;
        default:
          edm::LogWarning("LogicError") << "Unknow BPix layer: " << m_layer;
          break;
      }
      // FPix
    } else if (m_subdetid == 2) {
      switch (m_layer) {
        case 1:
          m_side > 1 ? ret = AlignmentPI::FPixpL1 : ret = AlignmentPI::FPixmL1;
          break;
        case 2:
          m_side > 1 ? ret = AlignmentPI::FPixpL2 : ret = AlignmentPI::FPixmL2;
          break;
        case 3:
          m_side > 1 ? ret = AlignmentPI::FPixpL3 : ret = AlignmentPI::FPixmL3;
          break;
        default:
          edm::LogWarning("LogicError") << "Unknow FPix disk: " << m_layer;
          break;
      }
      // TIB
    } else if (m_subdetid == 3) {
      switch (m_layer) {
        case 1:
          if (m_isRphi) {
            m_isInternal > 0 ? ret = AlignmentPI::TIBL1Ro : ret = AlignmentPI::TIBL1Ri;
          } else {
            m_isInternal > 0 ? ret = AlignmentPI::TIBL1So : ret = AlignmentPI::TIBL1Si;
          }
          break;
        case 2:
          if (m_isRphi) {
            m_isInternal > 0 ? ret = AlignmentPI::TIBL2Ro : ret = AlignmentPI::TIBL2Ri;
          } else {
            m_isInternal > 0 ? ret = AlignmentPI::TIBL2So : ret = AlignmentPI::TIBL2Si;
          }
          break;
        case 3:
          m_isInternal > 0 ? ret = AlignmentPI::TIBL3o : ret = AlignmentPI::TIBL3i;
          break;
        case 4:
          m_isInternal > 0 ? ret = AlignmentPI::TIBL4o : ret = AlignmentPI::TIBL4i;
          break;
        default:
          edm::LogWarning("LogicError") << "Unknow TIB layer: " << m_layer;
          break;
      }
      // TID
    } else if (m_subdetid == 4) {
      switch (m_ring) {
        case 1:
          if (m_isRphi) {
            m_side > 1 ? ret = AlignmentPI::TIDpR1R : ret = AlignmentPI::TIDmR1R;
          } else {
            m_side > 1 ? ret = AlignmentPI::TIDpR1S : ret = AlignmentPI::TIDmR1S;
          }
          break;
        case 2:
          if (m_isRphi) {
            m_side > 1 ? ret = AlignmentPI::TIDpR2R : ret = AlignmentPI::TIDmR2R;
          } else {
            m_side > 1 ? ret = AlignmentPI::TIDpR2S : ret = AlignmentPI::TIDmR2S;
          }
          break;
        case 3:
          m_side > 1 ? ret = AlignmentPI::TIDpR3 : ret = AlignmentPI::TIDmR3;
          break;
        default:
          edm::LogWarning("LogicError") << "Unknow TID wheel: " << m_layer;
          break;
      }
      // TOB
    } else if (m_subdetid == 5) {
      switch (m_layer) {
        case 1:
          if (m_isRphi) {
            m_isInternal > 0 ? ret = AlignmentPI::TOBL1Ro : ret = AlignmentPI::TOBL1Ri;
          } else {
            m_isInternal > 0 ? ret = AlignmentPI::TOBL1So : ret = AlignmentPI::TOBL1Si;
          }
          break;
        case 2:
          if (m_isRphi) {
            m_isInternal > 0 ? ret = AlignmentPI::TOBL2Ro : ret = AlignmentPI::TOBL2Ri;
          } else {
            m_isInternal > 0 ? ret = AlignmentPI::TOBL2So : ret = AlignmentPI::TOBL2Si;
          }
          break;
        case 3:
          m_isInternal > 0 ? ret = AlignmentPI::TOBL3o : ret = AlignmentPI::TOBL3i;
          break;
        case 4:
          m_isInternal > 0 ? ret = AlignmentPI::TOBL4o : ret = AlignmentPI::TOBL4i;
          break;
        case 5:
          m_isInternal > 0 ? ret = AlignmentPI::TOBL5o : ret = AlignmentPI::TOBL5i;
          break;
        case 6:
          m_isInternal > 0 ? ret = AlignmentPI::TOBL6o : ret = AlignmentPI::TOBL6i;
          break;
        default:
          edm::LogWarning("LogicError") << "Unknow TOB layer: " << m_layer;
          break;
      }
      // TEC
    } else if (m_subdetid == 6) {
      switch (m_ring) {
        case 1:
          if (m_isRphi) {
            m_side > 1 ? ret = AlignmentPI::TECpR1R : ret = AlignmentPI::TECmR1R;
          } else {
            m_side > 1 ? ret = AlignmentPI::TECpR1S : ret = AlignmentPI::TECmR1S;
          }
          break;
        case 2:
          if (m_isRphi) {
            m_side > 1 ? ret = AlignmentPI::TECpR2R : ret = AlignmentPI::TECmR2R;
          } else {
            m_side > 1 ? ret = AlignmentPI::TECpR2S : ret = AlignmentPI::TECmR2S;
          }
          break;
        case 3:
          m_side > 1 ? ret = AlignmentPI::TECpR3 : ret = AlignmentPI::TECmR3;
          break;
        case 4:
          m_side > 1 ? ret = AlignmentPI::TECpR4 : ret = AlignmentPI::TECmR4;
          break;
        case 5:
          m_side > 1 ? ret = AlignmentPI::TECpR5 : ret = AlignmentPI::TECmR5;
          break;
        case 6:
          m_side > 1 ? ret = AlignmentPI::TECpR6 : ret = AlignmentPI::TECmR6;
          break;
        case 7:
          m_side > 1 ? ret = AlignmentPI::TECpR7 : ret = AlignmentPI::TECmR7;
          break;
        default:
          edm::LogWarning("LogicError") << "Unknow TEC ring: " << m_ring;
          break;
      }
    }

    return ret;
  }

  /*--------------------------------------------------------------------*/
  inline std::string getStringFromCoordinate(AlignmentPI::coordinate coord)
  /*--------------------------------------------------------------------*/
  {
    switch (coord) {
      case t_x:
        return "x-translation";
      case t_y:
        return "y-translation";
      case t_z:
        return "z-translation";
      case rot_alpha:
        return "#alpha angle rotation";
      case rot_beta:
        return "#beta angle rotation";
      case rot_gamma:
        return "#gamma angle rotation";
      default:
        return "should never be here!";
    }
  }

  /*--------------------------------------------------------------------*/
  inline std::string getStringFromIndex(AlignmentPI::index i)
  /*--------------------------------------------------------------------*/
  {
    switch (i) {
      case XX:
        return "XX";
      case XY:
        return "XY";
      case XZ:
        return "XZ";
      case YZ:
        return "YX";
      case YY:
        return "YY";
      case ZZ:
        return "ZZ";
      default:
        return "should never be here!";
    }
  }

  /*--------------------------------------------------------------------*/
  inline std::string getStringFromPart(AlignmentPI::partitions i, bool isPhase2 = false)
  /*--------------------------------------------------------------------*/
  {
    switch (i) {
      case BPix:
        return "BPix";
      case FPix:
        return "FPix";
      case TIB:
        return (isPhase2 ? "TIB-invalid" : "TIB");
      case TID:
        return (isPhase2 ? "P2OTEC" : "TID");
      case TOB:
        return (isPhase2 ? "P2OTB" : "TOB");
      case TEC:
        return (isPhase2 ? "TEC-invalid" : "TEC");
      default:
        return "should never be here!";
    }
  }

  /*--------------------------------------------------------------------*/
  inline std::pair<int, int> getIndices(AlignmentPI::index i)
  /*--------------------------------------------------------------------*/
  {
    switch (i) {
      case XX:
        return std::make_pair(0, 0);
      case XY:
        return std::make_pair(0, 1);
      case XZ:
        return std::make_pair(0, 2);
      case YZ:
        return std::make_pair(1, 0);
      case YY:
        return std::make_pair(1, 1);
      case ZZ:
        return std::make_pair(2, 2);
      default:
        return std::make_pair(-1, -1);
    }
  }

  /*--------------------------------------------------------------------*/
  inline void makeNicePlotStyle(TH1* hist, int color)
  /*--------------------------------------------------------------------*/
  {
    hist->SetStats(kFALSE);

    hist->GetXaxis()->SetTitleColor(color);
    hist->SetLineColor(color);
    hist->SetTitleSize(0.08);
    hist->SetLineWidth(2);
    hist->GetXaxis()->CenterTitle(true);
    hist->GetYaxis()->CenterTitle(true);
    hist->GetXaxis()->SetTitleFont(42);
    hist->GetYaxis()->SetTitleFont(42);
    hist->GetXaxis()->SetNdivisions(505);
    hist->GetXaxis()->SetTitleSize(0.06);
    hist->GetYaxis()->SetTitleSize(0.06);
    hist->GetXaxis()->SetTitleOffset(1.0);
    hist->GetYaxis()->SetTitleOffset(1.3);
    hist->GetXaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetLabelSize(.05);
    hist->GetXaxis()->SetLabelSize(.05);
  }

  /*--------------------------------------------------------------------*/
  inline void makeNiceStats(TH1F* hist, AlignmentPI::partitions part, int color)
  /*--------------------------------------------------------------------*/
  {
    char buffer[255];
    TPaveText* stat = new TPaveText(0.60, 0.75, 0.95, 0.95, "NDC");
    sprintf(buffer, "%s \n", AlignmentPI::getStringFromPart(part).c_str());
    stat->AddText(buffer);

    sprintf(buffer, "Entries : %i\n", (int)hist->GetEntries());
    stat->AddText(buffer);

    if (std::abs(hist->GetMean()) > 0.01) {
      sprintf(buffer, "Mean    : %6.2f\n", hist->GetMean());
    } else {
      sprintf(buffer, "Mean    : %6.2f e-2\n", 100 * hist->GetMean());
    }
    stat->AddText(buffer);

    if (std::abs(hist->GetRMS()) > 0.01) {
      sprintf(buffer, "RMS     : %6.2f\n", hist->GetRMS());
    } else {
      sprintf(buffer, "RMS     : %6.2f e-2\n", 100 * hist->GetRMS());
    }
    stat->AddText(buffer);

    stat->SetLineColor(color);
    stat->SetTextColor(color);
    stat->SetFillColor(10);
    stat->SetShadowColor(10);
    stat->Draw();
  }

  /*--------------------------------------------------------------------*/
  inline std::pair<float, float> getTheRange(std::map<uint32_t, float> values, const float nsigma)
  /*--------------------------------------------------------------------*/
  {
    float sum = std::accumulate(
        std::begin(values), std::end(values), 0.0, [](float value, const std::map<uint32_t, float>::value_type& p) {
          return value + p.second;
        });

    float m = sum / values.size();

    float accum = 0.0;
    std::for_each(std::begin(values), std::end(values), [&](const std::map<uint32_t, float>::value_type& p) {
      accum += (p.second - m) * (p.second - m);
    });

    float stdev = sqrt(accum / (values.size() - 1));

    if (stdev != 0.) {
      return std::make_pair(m - nsigma * stdev, m + nsigma * stdev);
    } else {
      return std::make_pair(m > 0. ? 0.95 * m : 1.05 * m, m > 0 ? 1.05 * m : 0.95 * m);
    }
  }

  /*--------------------------------------------------------------------*/
  inline std::pair<double, double> calculatePosition(TVirtualPad* myPad, int boundary)
  /*--------------------------------------------------------------------*/
  {
    int ix1;
    int ix2;
    int iw = myPad->GetWw();
    int ih = myPad->GetWh();
    double x1p, y1p, x2p, y2p;
    myPad->GetPadPar(x1p, y1p, x2p, y2p);
    ix1 = (int)(iw * x1p);
    ix2 = (int)(iw * x2p);
    double wndc = std::min(1., (double)iw / (double)ih);
    double rw = wndc / (double)iw;
    double x1ndc = (double)ix1 * rw;
    double x2ndc = (double)ix2 * rw;
    double rx1, ry1, rx2, ry2;
    myPad->GetRange(rx1, ry1, rx2, ry2);
    double rx = (x2ndc - x1ndc) / (rx2 - rx1);
    double _sx;
    _sx = rx * (boundary - rx1) + x1ndc;
    double _dx = _sx + 0.05;

    return std::make_pair(_sx, _dx);
  }

  // ancillary struct to manage the barycenters
  // info in a more compact way

  struct TkAlBarycenters {
    std::map<AlignmentPI::PARTITION, double> Xbarycenters;
    std::map<AlignmentPI::PARTITION, double> Ybarycenters;
    std::map<AlignmentPI::PARTITION, double> Zbarycenters;
    std::map<AlignmentPI::PARTITION, double> nmodules;

  public:
    void init();
    GlobalPoint getPartitionAvg(AlignmentPI::PARTITION p);
    void computeBarycenters(const std::vector<AlignTransform>& input,
                            const TrackerTopology& tTopo,
                            const std::map<AlignmentPI::coordinate, float>& GPR);
    const double getNModules(AlignmentPI::PARTITION p) { return nmodules[p]; };

    // M.M. 2020/01/09
    // introduce methods for entire partitions, summing up the two sides of the
    // endcap detectors

    /*--------------------------------------------------------------------*/
    const std::array<double, 6> getX()
    /*--------------------------------------------------------------------*/
    {
      return {{Xbarycenters[PARTITION::BPIX],
               (Xbarycenters[PARTITION::FPIXm] + Xbarycenters[PARTITION::FPIXp]) / 2.,
               Xbarycenters[PARTITION::TIB],
               (Xbarycenters[PARTITION::TIDm] + Xbarycenters[PARTITION::TIDp]) / 2,
               Xbarycenters[PARTITION::TOB],
               (Xbarycenters[PARTITION::TECm] + Xbarycenters[PARTITION::TECp]) / 2}};
    };

    /*--------------------------------------------------------------------*/
    const std::array<double, 6> getY()
    /*--------------------------------------------------------------------*/
    {
      return {{Ybarycenters[PARTITION::BPIX],
               (Ybarycenters[PARTITION::FPIXm] + Ybarycenters[PARTITION::FPIXp]) / 2.,
               Ybarycenters[PARTITION::TIB],
               (Ybarycenters[PARTITION::TIDm] + Ybarycenters[PARTITION::TIDp]) / 2,
               Ybarycenters[PARTITION::TOB],
               (Ybarycenters[PARTITION::TECm] + Ybarycenters[PARTITION::TECp]) / 2}};
    };

    /*--------------------------------------------------------------------*/
    const std::array<double, 6> getZ()
    /*--------------------------------------------------------------------*/
    {
      return {{Zbarycenters[PARTITION::BPIX],
               (Zbarycenters[PARTITION::FPIXm] + Zbarycenters[PARTITION::FPIXp]) / 2.,
               Zbarycenters[PARTITION::TIB],
               (Zbarycenters[PARTITION::TIDm] + Zbarycenters[PARTITION::TIDp]) / 2,
               Zbarycenters[PARTITION::TOB],
               (Zbarycenters[PARTITION::TECm] + Zbarycenters[PARTITION::TECp]) / 2}};
    };
    virtual ~TkAlBarycenters() {}
  };

  /*--------------------------------------------------------------------*/
  inline GlobalPoint TkAlBarycenters::getPartitionAvg(AlignmentPI::PARTITION p)
  /*--------------------------------------------------------------------*/
  {
    return GlobalPoint(Xbarycenters[p], Ybarycenters[p], Zbarycenters[p]);
  }

  /*--------------------------------------------------------------------*/
  inline void TkAlBarycenters::computeBarycenters(const std::vector<AlignTransform>& input,
                                                  const TrackerTopology& tTopo,
                                                  const std::map<AlignmentPI::coordinate, float>& GPR)
  /*--------------------------------------------------------------------*/
  {
    // zero in the n. modules per partition...
    for (const auto& p : PARTITIONS) {
      nmodules[p] = 0.;
    }

    for (const auto& ali : input) {
      if (DetId(ali.rawId()).det() != DetId::Tracker) {
        edm::LogWarning("TkAlBarycenters::computeBarycenters")
            << "Encountered invalid Tracker DetId:" << ali.rawId() << " " << DetId(ali.rawId()).det()
            << " is different from " << DetId::Tracker << "  - terminating ";
        assert(DetId(ali.rawId()).det() != DetId::Tracker);
      }

      int subid = DetId(ali.rawId()).subdetId();
      switch (subid) {
        case PixelSubdetector::PixelBarrel:
          Xbarycenters[PARTITION::BPIX] += (ali.translation().x());
          Ybarycenters[PARTITION::BPIX] += (ali.translation().y());
          Zbarycenters[PARTITION::BPIX] += (ali.translation().z());
          nmodules[PARTITION::BPIX]++;
          break;
        case PixelSubdetector::PixelEndcap:

          // minus side
          if (tTopo.pxfSide(DetId(ali.rawId())) == 1) {
            Xbarycenters[PARTITION::FPIXm] += (ali.translation().x());
            Ybarycenters[PARTITION::FPIXm] += (ali.translation().y());
            Zbarycenters[PARTITION::FPIXm] += (ali.translation().z());
            nmodules[PARTITION::FPIXm]++;
          }  // plus side
          else {
            Xbarycenters[PARTITION::FPIXp] += (ali.translation().x());
            Ybarycenters[PARTITION::FPIXp] += (ali.translation().y());
            Zbarycenters[PARTITION::FPIXp] += (ali.translation().z());
            nmodules[PARTITION::FPIXp]++;
          }
          break;
        case StripSubdetector::TIB:
          Xbarycenters[PARTITION::TIB] += (ali.translation().x());
          Ybarycenters[PARTITION::TIB] += (ali.translation().y());
          Zbarycenters[PARTITION::TIB] += (ali.translation().z());
          nmodules[PARTITION::TIB]++;
          break;
        case StripSubdetector::TID:
          // minus side
          if (tTopo.tidSide(DetId(ali.rawId())) == 1) {
            Xbarycenters[PARTITION::TIDm] += (ali.translation().x());
            Ybarycenters[PARTITION::TIDm] += (ali.translation().y());
            Zbarycenters[PARTITION::TIDm] += (ali.translation().z());
            nmodules[PARTITION::TIDm]++;
          }  // plus side
          else {
            Xbarycenters[PARTITION::TIDp] += (ali.translation().x());
            Ybarycenters[PARTITION::TIDp] += (ali.translation().y());
            Zbarycenters[PARTITION::TIDp] += (ali.translation().z());
            nmodules[PARTITION::TIDp]++;
          }
          break;
        case StripSubdetector::TOB:
          Xbarycenters[PARTITION::TOB] += (ali.translation().x());
          Ybarycenters[PARTITION::TOB] += (ali.translation().y());
          Zbarycenters[PARTITION::TOB] += (ali.translation().z());
          nmodules[PARTITION::TOB]++;
          break;
        case StripSubdetector::TEC:
          // minus side
          if (tTopo.tecSide(DetId(ali.rawId())) == 1) {
            Xbarycenters[PARTITION::TECm] += (ali.translation().x());
            Ybarycenters[PARTITION::TECm] += (ali.translation().y());
            Zbarycenters[PARTITION::TECm] += (ali.translation().z());
            nmodules[PARTITION::TECm]++;
          }  // plus side
          else {
            Xbarycenters[PARTITION::TECp] += (ali.translation().x());
            Ybarycenters[PARTITION::TECp] += (ali.translation().y());
            Zbarycenters[PARTITION::TECp] += (ali.translation().z());
            nmodules[PARTITION::TECp]++;
          }
          break;
        default:
          edm::LogError("TrackerAlignment_PayloadInspector") << "Unrecognized partition " << subid << std::endl;
          break;
      }
    }

    for (const auto& p : PARTITIONS) {
      // take the arithmetic mean
      Xbarycenters[p] /= nmodules[p];
      Ybarycenters[p] /= nmodules[p];
      Zbarycenters[p] /= nmodules[p];

      // add the Tracker Global Position Record
      Xbarycenters[p] += GPR.at(AlignmentPI::t_x);
      Ybarycenters[p] += GPR.at(AlignmentPI::t_y);
      Zbarycenters[p] += GPR.at(AlignmentPI::t_z);

      COUT << "Partition: " << p << " n. modules: " << nmodules[p] << "|"
           << " X: " << std::right << std::setw(12) << Xbarycenters[p] << " Y: " << std::right << std::setw(12)
           << Ybarycenters[p] << " Z: " << std::right << std::setw(12) << Zbarycenters[p] << std::endl;
    }
  }

  /*--------------------------------------------------------------------*/
  inline void fillComparisonHistogram(const AlignmentPI::coordinate& coord,
                                      std::map<int, AlignmentPI::partitions>& boundaries,
                                      const std::vector<AlignTransform>& ref_ali,
                                      const std::vector<AlignTransform>& target_ali,
                                      std::unique_ptr<TH1F>& compare)
  /*--------------------------------------------------------------------*/
  {
    int counter = 0; /* start the counter */
    AlignmentPI::partitions currentPart = AlignmentPI::BPix;
    for (unsigned int i = 0; i < ref_ali.size(); i++) {
      if (ref_ali[i].rawId() == target_ali[i].rawId()) {
        counter++;
        int subid = DetId(ref_ali[i].rawId()).subdetId();

        auto thePart = static_cast<AlignmentPI::partitions>(subid);
        if (thePart != currentPart) {
          currentPart = thePart;
          boundaries.insert({counter, thePart});
        }

        CLHEP::HepRotation target_rot(target_ali[i].rotation());
        CLHEP::HepRotation ref_rot(ref_ali[i].rotation());

        align::RotationType target_ROT(target_rot.xx(),
                                       target_rot.xy(),
                                       target_rot.xz(),
                                       target_rot.yx(),
                                       target_rot.yy(),
                                       target_rot.yz(),
                                       target_rot.zx(),
                                       target_rot.zy(),
                                       target_rot.zz());

        align::RotationType ref_ROT(ref_rot.xx(),
                                    ref_rot.xy(),
                                    ref_rot.xz(),
                                    ref_rot.yx(),
                                    ref_rot.yy(),
                                    ref_rot.yz(),
                                    ref_rot.zx(),
                                    ref_rot.zy(),
                                    ref_rot.zz());

        const std::vector<double> deltaRot = {::deltaPhi(align::toAngles(target_ROT)[0], align::toAngles(ref_ROT)[0]),
                                              ::deltaPhi(align::toAngles(target_ROT)[1], align::toAngles(ref_ROT)[1]),
                                              ::deltaPhi(align::toAngles(target_ROT)[2], align::toAngles(ref_ROT)[2])};

        const auto& deltaTrans = target_ali[i].translation() - ref_ali[i].translation();

        switch (coord) {
          case AlignmentPI::t_x:
            compare->SetBinContent(i + 1, deltaTrans.x() * AlignmentPI::cmToUm);
            break;
          case AlignmentPI::t_y:
            compare->SetBinContent(i + 1, deltaTrans.y() * AlignmentPI::cmToUm);
            break;
          case AlignmentPI::t_z:
            compare->SetBinContent(i + 1, deltaTrans.z() * AlignmentPI::cmToUm);
            break;
          case AlignmentPI::rot_alpha:
            compare->SetBinContent(i + 1, deltaRot[0] * AlignmentPI::tomRad);
            break;
          case AlignmentPI::rot_beta:
            compare->SetBinContent(i + 1, deltaRot[1] * AlignmentPI::tomRad);
            break;
          case AlignmentPI::rot_gamma:
            compare->SetBinContent(i + 1, deltaRot[2] * AlignmentPI::tomRad);
            break;
          default:
            edm::LogError("TrackerAlignment_PayloadInspector") << "Unrecognized coordinate " << coord << std::endl;
            break;
        }  // switch on the coordinate
      }    // check on the same detID
    }      // loop on the components
  }

  /*--------------------------------------------------------------------*/
  inline void fillComparisonHistograms(std::map<int, AlignmentPI::partitions>& boundaries,
                                       const std::vector<AlignTransform>& ref_ali,
                                       const std::vector<AlignTransform>& target_ali,
                                       std::unordered_map<AlignmentPI::coordinate, std::unique_ptr<TH1F> >& compare,
                                       bool diff = false,
                                       AlignmentPI::partitions checkPart = AlignmentPI::INVALID)
  /*--------------------------------------------------------------------*/
  {
    int counter = 0; /* start the counter */
    AlignmentPI::partitions currentPart = AlignmentPI::BPix;
    for (unsigned int i = 0; i < ref_ali.size(); i++) {
      if (ref_ali[i].rawId() == target_ali[i].rawId()) {
        counter++;
        int subid = DetId(ref_ali[i].rawId()).subdetId();

        auto thePart = static_cast<AlignmentPI::partitions>(subid);

        // in case it has to be filtered
        if (checkPart > 0 && thePart != checkPart) {
          continue;
        }

        if (thePart != currentPart) {
          currentPart = thePart;
          boundaries.insert({counter, thePart});
        }

        CLHEP::HepRotation target_rot(target_ali[i].rotation());
        CLHEP::HepRotation ref_rot(ref_ali[i].rotation());

        align::RotationType target_ROT(target_rot.xx(),
                                       target_rot.xy(),
                                       target_rot.xz(),
                                       target_rot.yx(),
                                       target_rot.yy(),
                                       target_rot.yz(),
                                       target_rot.zx(),
                                       target_rot.zy(),
                                       target_rot.zz());

        align::RotationType ref_ROT(ref_rot.xx(),
                                    ref_rot.xy(),
                                    ref_rot.xz(),
                                    ref_rot.yx(),
                                    ref_rot.yy(),
                                    ref_rot.yz(),
                                    ref_rot.zx(),
                                    ref_rot.zy(),
                                    ref_rot.zz());

        const std::vector<double> deltaRot = {::deltaPhi(align::toAngles(target_ROT)[0], align::toAngles(ref_ROT)[0]),
                                              ::deltaPhi(align::toAngles(target_ROT)[1], align::toAngles(ref_ROT)[1]),
                                              ::deltaPhi(align::toAngles(target_ROT)[2], align::toAngles(ref_ROT)[2])};

        const auto& deltaTrans = target_ali[i].translation() - ref_ali[i].translation();

        // fill the histograms
        if (diff) {
          compare[AlignmentPI::t_x]->Fill(deltaTrans.x() * AlignmentPI::cmToUm);
          compare[AlignmentPI::t_y]->Fill(deltaTrans.y() * AlignmentPI::cmToUm);
          compare[AlignmentPI::t_z]->Fill(deltaTrans.z() * AlignmentPI::cmToUm);

          compare[AlignmentPI::rot_alpha]->Fill(deltaRot[0] * AlignmentPI::tomRad);
          compare[AlignmentPI::rot_beta]->Fill(deltaRot[1] * AlignmentPI::tomRad);
          compare[AlignmentPI::rot_gamma]->Fill(deltaRot[2] * AlignmentPI::tomRad);
        } else {
          compare[AlignmentPI::t_x]->SetBinContent(i + 1, deltaTrans.x() * AlignmentPI::cmToUm);
          compare[AlignmentPI::t_y]->SetBinContent(i + 1, deltaTrans.y() * AlignmentPI::cmToUm);
          compare[AlignmentPI::t_z]->SetBinContent(i + 1, deltaTrans.z() * AlignmentPI::cmToUm);

          compare[AlignmentPI::rot_alpha]->SetBinContent(i + 1, deltaRot[0] * AlignmentPI::tomRad);
          compare[AlignmentPI::rot_beta]->SetBinContent(i + 1, deltaRot[1] * AlignmentPI::tomRad);
          compare[AlignmentPI::rot_gamma]->SetBinContent(i + 1, deltaRot[2] * AlignmentPI::tomRad);
        }

      }  // if it's the same detid
    }    // loop on detids
  }

}  // namespace AlignmentPI

#endif
