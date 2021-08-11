#ifndef DQM_SiPixelPhase1Summary_SiPixelBarycenterHelper_h
#define DQM_SiPixelPhase1Summary_SiPixelBarycenterHelper_h

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "Alignment/TrackerAlignment/interface/TrackerNameSpace.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"

// Helper mainly based on https://github.com/cms-sw/cmssw/blob/master/CondCore/AlignmentPlugins/interface/AlignmentPayloadInspectorHelper.h

namespace DQMBarycenter {

  enum coordinate {
    t_x = 1,
    t_y = 2,
    t_z = 3,
    rot_alpha = 4,
    rot_beta = 5,
    rot_gamma = 6,
  };

  enum partitions {
    BPix = 1,
    FPix_zp = 2,
    FPix_zm = 3,
    BPix_xp = 4,
    BPix_xm = 5,
    FPix_zp_xp = 6,
    FPix_zm_xp = 7,
    FPix_zp_xm = 8,
    FPix_zm_xm = 9,
  };

  enum class PARTITION {
    BPIX,        // 0 Barrel Pixel
    FPIX_zp,     // 1 Forward Pixel Z-Plus
    FPIX_zm,     // 2 Forward Pixel Z-Minus
    BPIX_xp,     // 3 Barrel Pixel X-Plus
    BPIX_xm,     // 4 Barrel Pixel X-Minus
    FPIX_zp_xp,  // 5 Forward Pixel Z-Plus X-Plus
    FPIX_zm_xp,  // 6 Forward Pixel Z-Minus X-Plus
    FPIX_zp_xm,  // 7 Forward Pixel Z-Plus X-Minus
    FPIX_zm_xm,  // 8 Forward Pixel Z-Minus X-Minus
    LAST = FPIX_zm_xm
  };

  extern const PARTITION PARTITIONS[(int)PARTITION::LAST + 1];
  const PARTITION PARTITIONS[] = {
      PARTITION::BPIX,
      PARTITION::FPIX_zp,
      PARTITION::FPIX_zm,
      PARTITION::BPIX_xp,
      PARTITION::BPIX_xm,
      PARTITION::FPIX_zp_xp,
      PARTITION::FPIX_zm_xp,
      PARTITION::FPIX_zp_xm,
      PARTITION::FPIX_zm_xm,
  };

  class TkAlBarycenters {
    std::map<DQMBarycenter::PARTITION, double> Xbarycenters;
    std::map<DQMBarycenter::PARTITION, double> Ybarycenters;
    std::map<DQMBarycenter::PARTITION, double> Zbarycenters;
    std::map<DQMBarycenter::PARTITION, double> nmodules;

  public:
    inline void init();

    /*--------------------------------------------------------------------*/
    inline const std::array<double, 9> getX()
    /*--------------------------------------------------------------------*/
    {
      return {{Xbarycenters[PARTITION::BPIX],
               Xbarycenters[PARTITION::FPIX_zm],
               Xbarycenters[PARTITION::FPIX_zp],
               Xbarycenters[PARTITION::BPIX_xp],
               Xbarycenters[PARTITION::BPIX_xm],
               Xbarycenters[PARTITION::FPIX_zp_xp],
               Xbarycenters[PARTITION::FPIX_zm_xp],
               Xbarycenters[PARTITION::FPIX_zp_xm],
               Xbarycenters[PARTITION::FPIX_zm_xm]}};
    };

    /*--------------------------------------------------------------------*/
    inline const std::array<double, 9> getY()
    /*--------------------------------------------------------------------*/
    {
      return {{Ybarycenters[PARTITION::BPIX],
               Ybarycenters[PARTITION::FPIX_zm],
               Ybarycenters[PARTITION::FPIX_zp],
               Ybarycenters[PARTITION::BPIX_xp],
               Ybarycenters[PARTITION::BPIX_xm],
               Ybarycenters[PARTITION::FPIX_zp_xp],
               Ybarycenters[PARTITION::FPIX_zm_xp],
               Ybarycenters[PARTITION::FPIX_zp_xm],
               Ybarycenters[PARTITION::FPIX_zm_xm]}};
    };

    /*--------------------------------------------------------------------*/
    inline const std::array<double, 9> getZ()
    /*--------------------------------------------------------------------*/
    {
      return {{Zbarycenters[PARTITION::BPIX],
               Zbarycenters[PARTITION::FPIX_zm],
               Zbarycenters[PARTITION::FPIX_zp],
               Zbarycenters[PARTITION::BPIX_xp],
               Zbarycenters[PARTITION::BPIX_xm],
               Zbarycenters[PARTITION::FPIX_zp_xp],
               Zbarycenters[PARTITION::FPIX_zm_xp],
               Zbarycenters[PARTITION::FPIX_zp_xm],
               Zbarycenters[PARTITION::FPIX_zm_xm]}};
    };
    inline virtual ~TkAlBarycenters() {}

    /*--------------------------------------------------------------------*/
    inline void computeBarycenters(const std::vector<AlignTransform>& input,
                                   const TrackerTopology& tTopo,
                                   const std::map<DQMBarycenter::coordinate, float>& GPR)
    /*--------------------------------------------------------------------*/
    {
      for (const auto& ali : input) {
        if (DetId(ali.rawId()).det() != DetId::Tracker) {
          edm::LogWarning("SiPixelBarycenters::computeBarycenters")
              << "Encountered invalid Tracker DetId:" << ali.rawId() << " " << DetId(ali.rawId()).det()
              << " is different from " << DetId::Tracker << "  - terminating ";
          assert(DetId(ali.rawId()).det() != DetId::Tracker);
        }

        const auto& tns = align::TrackerNameSpace(&tTopo);
        int subid = DetId(ali.rawId()).subdetId();
        if (subid == PixelSubdetector::PixelBarrel || subid == PixelSubdetector::PixelEndcap) {  // use only pixel
          switch (subid) {  // Separate BPIX, FPIX_zp and FPIX_zm
            case PixelSubdetector::PixelBarrel:
              Xbarycenters[PARTITION::BPIX] += (ali.translation().x());
              Ybarycenters[PARTITION::BPIX] += (ali.translation().y());
              Zbarycenters[PARTITION::BPIX] += (ali.translation().z());
              nmodules[PARTITION::BPIX]++;
              break;
            case PixelSubdetector::PixelEndcap:
              // minus side
              if (tns.tpe().endcapNumber(ali.rawId()) == 1) {
                Xbarycenters[PARTITION::FPIX_zm] += (ali.translation().x());
                Ybarycenters[PARTITION::FPIX_zm] += (ali.translation().y());
                Zbarycenters[PARTITION::FPIX_zm] += (ali.translation().z());
                nmodules[PARTITION::FPIX_zm]++;
              }  // plus side
              else {
                Xbarycenters[PARTITION::FPIX_zp] += (ali.translation().x());
                Ybarycenters[PARTITION::FPIX_zp] += (ali.translation().y());
                Zbarycenters[PARTITION::FPIX_zp] += (ali.translation().z());
                nmodules[PARTITION::FPIX_zp]++;
              }
              break;
            default:
              edm::LogError("PixelDQM") << "Unrecognized partition for barycenter computation " << subid << std::endl;
              break;
          }

          switch (subid) {  // Separate following the PCL HLS
            case PixelSubdetector::PixelBarrel:
              if ((PixelBarrelName(DetId(ali.rawId()), true).shell() == PixelBarrelName::mO) ||
                  (PixelBarrelName(DetId(ali.rawId()), true).shell() == PixelBarrelName::pO)) {  // BPIX x-
                Xbarycenters[PARTITION::BPIX_xm] += (ali.translation().x());
                Ybarycenters[PARTITION::BPIX_xm] += (ali.translation().y());
                Zbarycenters[PARTITION::BPIX_xm] += (ali.translation().z());
                nmodules[PARTITION::BPIX_xm]++;
              } else {  // BPIX x+
                Xbarycenters[PARTITION::BPIX_xp] += (ali.translation().x());
                Ybarycenters[PARTITION::BPIX_xp] += (ali.translation().y());
                Zbarycenters[PARTITION::BPIX_xp] += (ali.translation().z());
                nmodules[PARTITION::BPIX_xp]++;
              }
              break;
            case PixelSubdetector::PixelEndcap:
              if (PixelEndcapName(DetId(ali.rawId()), true).halfCylinder() == PixelEndcapName::mO) {  //FPIX z- x-
                Xbarycenters[PARTITION::FPIX_zm_xm] += (ali.translation().x());
                Ybarycenters[PARTITION::FPIX_zm_xm] += (ali.translation().y());
                Zbarycenters[PARTITION::FPIX_zm_xm] += (ali.translation().z());
                nmodules[PARTITION::FPIX_zm_xm]++;
              } else if (PixelEndcapName(DetId(ali.rawId()), true).halfCylinder() ==
                         PixelEndcapName::mI) {  //FPIX z- x+
                Xbarycenters[PARTITION::FPIX_zm_xp] += (ali.translation().x());
                Ybarycenters[PARTITION::FPIX_zm_xp] += (ali.translation().y());
                Zbarycenters[PARTITION::FPIX_zm_xp] += (ali.translation().z());
                nmodules[PARTITION::FPIX_zm_xp]++;
              } else if (PixelEndcapName(DetId(ali.rawId()), true).halfCylinder() ==
                         PixelEndcapName::pO) {  //FPIX z+ x-
                Xbarycenters[PARTITION::FPIX_zp_xm] += (ali.translation().x());
                Ybarycenters[PARTITION::FPIX_zp_xm] += (ali.translation().y());
                Zbarycenters[PARTITION::FPIX_zp_xm] += (ali.translation().z());
                nmodules[PARTITION::FPIX_zp_xm]++;
              } else if (PixelEndcapName(DetId(ali.rawId()), true).halfCylinder() ==
                         PixelEndcapName::pI) {  //FPIX z+ x+
                Xbarycenters[PARTITION::FPIX_zp_xp] += (ali.translation().x());
                Ybarycenters[PARTITION::FPIX_zp_xp] += (ali.translation().y());
                Zbarycenters[PARTITION::FPIX_zp_xp] += (ali.translation().z());
                nmodules[PARTITION::FPIX_zp_xp]++;
              } else {
                edm::LogError("PixelDQM") << "Unrecognized partition for barycenter computation " << subid << std::endl;
              }
              break;
            default:
              edm::LogError("PixelDQM") << "Unrecognized partition for barycenter computation " << subid << std::endl;
              break;
          }
        }
      }

      for (const auto& p : PARTITIONS) {
        Xbarycenters[p] /= nmodules[p];
        Ybarycenters[p] /= nmodules[p];
        Zbarycenters[p] /= nmodules[p];

        Xbarycenters[p] += GPR.at(DQMBarycenter::t_x);
        Ybarycenters[p] += GPR.at(DQMBarycenter::t_y);
        Zbarycenters[p] += GPR.at(DQMBarycenter::t_z);
      }
    }
  };
}  // namespace DQMBarycenter

#endif
