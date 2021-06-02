#ifndef DQM_SiPixelPhase1Summary_SiPixelBarycenterHelper_h
#define DQM_SiPixelPhase1Summary_SiPixelBarycenterHelper_h

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/Alignment/interface/Alignments.h"

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

  enum partitions { BPix = 1, FPix_p = 2, FPix_m = 3,  };

  enum class PARTITION {
    BPIX,   // 0 Barrel Pixel
    FPIXp,  // 1 Forward Pixel Plus
    FPIXm,  // 2 Forward Pixel Minus
    LAST = FPIXm
  };

  extern const PARTITION PARTITIONS[(int)PARTITION::LAST + 1];
  const PARTITION PARTITIONS[] = {PARTITION::BPIX,
                                  PARTITION::FPIXp,
                                  PARTITION::FPIXm,
                                  };


  struct TkAlBarycenters {
    std::map<DQMBarycenter::PARTITION, double> Xbarycenters;
    std::map<DQMBarycenter::PARTITION, double> Ybarycenters;
    std::map<DQMBarycenter::PARTITION, double> Zbarycenters;
    std::map<DQMBarycenter::PARTITION, double> nmodules;

  public:
    void init();
    void computeBarycenters(const std::vector<AlignTransform>& input,
                            const TrackerTopology& tTopo,
                            const std::map<DQMBarycenter::coordinate, float>& GPR);
    

    /*--------------------------------------------------------------------*/
    const std::array<double, 3> getX()
    /*--------------------------------------------------------------------*/
    {
      return {{Xbarycenters[PARTITION::BPIX],
               Xbarycenters[PARTITION::FPIXm],
               Xbarycenters[PARTITION::FPIXp]
               }};
    };

    /*--------------------------------------------------------------------*/
    const std::array<double, 3> getY()
    /*--------------------------------------------------------------------*/
    {
      return {{Ybarycenters[PARTITION::BPIX],
               Ybarycenters[PARTITION::FPIXm],
               Ybarycenters[PARTITION::FPIXp]
               }};
    };

    /*--------------------------------------------------------------------*/
    const std::array<double, 3> getZ()
    /*--------------------------------------------------------------------*/
    {
      return {{Zbarycenters[PARTITION::BPIX],
               Zbarycenters[PARTITION::FPIXm],
               Zbarycenters[PARTITION::FPIXp]
               }};
    };
    virtual ~TkAlBarycenters() {}
  };

  /*--------------------------------------------------------------------*/
  void TkAlBarycenters::computeBarycenters(const std::vector<AlignTransform>& input,
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

      int subid = DetId(ali.rawId()).subdetId();
      if (subid==PixelSubdetector::PixelBarrel || subid==PixelSubdetector::PixelEndcap) {   // use only pixel
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
}  // namespace DQMBarycenter

#endif
