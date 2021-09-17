#ifndef CONDCORE_SIPIXELPLUGINS_PIXELREGIONCONTAINERS_H
#define CONDCORE_SIPIXELPLUGINS_PIXELREGIONCONTAINERS_H

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "CondCore/SiPixelPlugins/interface/SiPixelPayloadInspectorHelper.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <boost/range/adaptor/indexed.hpp>
#include <cstdlib>
#include "TH1.h"
#include "TLatex.h"

namespace PixelRegions {

  inline std::string itoa(int i) {
    char temp[20];
    sprintf(temp, "%d", i);
    return ((std::string)temp);
  }

  // "PixelId"
  // BPix: 1000*(subdetId=1) + 100*(layer=1,2,3,4)
  // FPix: 1000*(subdetId=2) + 100*(side=1,2) + 10*(disk=1,2,3) + 1*(ring=1,2)
  // Phase-2 FPix: 1000*(subdetId=3) + 10*(disk=1...12)

  // clang-format off
  enum PixelId {
    // BPix
    L1 = 1100, L2 = 1200, L3 = 1300, L4 = 1400, // common to phase1 and phase2
    // FPix minus
    Rm1l = 2111, Rm1u = 2112, Rm2l = 2121, Rm2u = 2122, Rm3l = 2131, Rm3u = 2132,   // phase1-only
    // FPix plus
    Rp1l = 2211, Rp1u = 2212, Rp2l = 2221, Rp2u = 2222, Rp3l = 2231, Rp3u = 2232,   // phase1-only
    // Phase-2 endcaps
    Ph2EmR1 = 3101, Ph2EmR2 = 3102, Ph2EmR3 = 3103, Ph2EmR4 = 3104,  // phase-2 EPix
    Ph2EpR1 = 3201, Ph2EpR2 = 3202, Ph2EpR3 = 3203, Ph2EpR4 = 3204,
    Ph2FmR1 = 4101, Ph2FmR2 = 4102, Ph2FmR3 = 4103, Ph2FmR4 = 4104, Ph2FmR5 = 4105, //phase-2 FPix
    Ph2FpR1 = 4201, Ph2FpR2 = 4202, Ph2FpR3 = 4203, Ph2FpR4 = 4204, Ph2FpR5 = 4205,
    End = 99999
  };

  const std::vector<PixelId> PixelIDs = {
    // BPix
    PixelId::L1, PixelId::L2, PixelId::L3, PixelId::L4,
    // Phase-1 FPix minus
    PixelId::Rm1l, PixelId::Rm1u, PixelId::Rm2l, PixelId::Rm2u, PixelId::Rm3l, PixelId::Rm3u, // phase-1 only
    // Phase-1 FPix plus
    PixelId::Rp1l, PixelId::Rp1u, PixelId::Rp2l, PixelId::Rp2u, PixelId::Rp3l, PixelId::Rp3u, // phase-1 only
    // Phase-2 endcaps
    PixelId::Ph2EmR1, PixelId::Ph2EmR2, PixelId::Ph2EmR3, PixelId::Ph2EmR4, // phase-2 EPix
    PixelId::Ph2EpR1, PixelId::Ph2EpR2, PixelId::Ph2EpR3, PixelId::Ph2EpR4,
    PixelId::Ph2FmR1, PixelId::Ph2FmR2, PixelId::Ph2FmR3, PixelId::Ph2FmR4, PixelId::Ph2FmR5, // phase-2 FPix
    PixelId::Ph2FpR1, PixelId::Ph2FpR2, PixelId::Ph2FpR3, PixelId::Ph2FpR4, PixelId::Ph2FpR5,
    PixelId::End
  };

  // clang-format on

  const std::vector<std::string> IDlabels = {"Barrel Pixel L1",            //1
                                             "Barrel Pixel L2",            //2
                                             "Barrel Pixel L3",            //3
                                             "Barrel Pixel L4",            //4
                                             "FPIX(-) Disk 1 inner ring",  //5
                                             "FPIX(-) Disk 1 outer ring",  //6
                                             "FPIX(-) Disk 2 inner ring",  //7
                                             "FPIX(-) Disk 2 outer ring",  //8
                                             "FPIX(-) Disk 3 inner ring",  //9
                                             "FPIX(-) Disk 3 outer ring",  //10
                                             "FPIX(+) Disk 1 inner ring",  //11
                                             "FPIX(+) Disk 1 outer ring",  //12
                                             "FPIX(+) Disk 2 inner ring",  //13
                                             "FPIX(+) Disk 2 outer ring",  //14
                                             "FPIX(+) Disk 3 inner ring",  //15
                                             "FPIX(+) Disk 3 outer ring",  //16
                                             "EPix(-) Ring 1",             //17
                                             "EPix(-) Ring 2",             //18
                                             "EPix(-) Ring 3",             //19
                                             "EPix(-) Ring 4",             //20
                                             "EPix(+) Ring 1",             //21
                                             "EPix(+) Ring 2",             //22
                                             "EPix(+) Ring 3",             //23
                                             "EPix(+) Ring 4",             //24
                                             "FPix(-) Ring 1",             //25
                                             "FPix(-) Ring 2",             //26
                                             "FPix(-) Ring 3",             //27
                                             "FPix(-) Ring 4",             //28
                                             "FPix(-) Ring 5",             //29
                                             "FPix(+) Ring 1",             //30
                                             "FPix(+) Ring 2",             //31
                                             "FPix(+) Ring 3",             //32
                                             "FPix(+) Ring 4",             //33
                                             "FPix(+) Ring 5",             //34
                                             "END"};                       //35

  //============================================================================
  [[maybe_unused]] static const std::vector<std::string> getIDLabels(const SiPixelPI::phase& ph, bool isBarrel) {
    std::vector<std::string> out;
    for (const auto& label : IDlabels) {
      if (isBarrel) {
        if (label.find("Barrel") != std::string::npos) {
          out.push_back(label);
        }
      } else {
        if (ph == SiPixelPI::phase::two) {
          if (label.find("Ring") != std::string::npos) {
            out.push_back(label);
          }
        } else {
          if (label.find("ring") != std::string::npos) {
            out.push_back(label);
          }
        }
      }
    }
    return out;
  }

  //============================================================================
  static const PixelId calculateBPixID(const unsigned int layer) {
    // BPix: 1000*(subdetId=1) + 100*(layer=1,2,3,4)
    PixelId bpixLayer = static_cast<PixelId>(1000 + 100 * layer);
    return bpixLayer;
  }

  //============================================================================
  static const PixelId calculateFPixID(const SiPixelPI::phase& ph,
                                       const unsigned int side,
                                       const unsigned int disk,
                                       const unsigned int ring) {
    // FPix: 1000*(subdetId=2) + 100*(side=1,2) + 10*(disk=1,2,3) + 1*(ring=1,2)
    using namespace SiPixelPI;
    unsigned int prefix(2000);
    unsigned int disk_(0);  // if that's phase-2 set the disk n. to zero
    if (ph > phase::one) {
      if (disk < 9) {
        prefix += 1000;  // if that's EPix id starts with 3k
      } else {
        prefix += 2000;  // if that's FPix id starts with 4k
      }
    } else {
      disk_ = disk;  // if that's not phase-2 set the disk to real disk n.
    }
    PixelId fpixRing = static_cast<PixelId>(prefix + 100 * side + 10 * disk_ + ring);
    return fpixRing;
  }

  //============================================================================
  static const PixelId detIdToPixelId(const unsigned int detid,
                                      const TrackerTopology* trackTopo,
                                      const SiPixelPI::phase& ph) {
    using namespace SiPixelPI;
    DetId detId = DetId(detid);
    unsigned int subid = detId.subdetId();
    unsigned int pixid = 0;
    if (subid == PixelSubdetector::PixelBarrel) {
      int layer = trackTopo->pxbLayer(detId);  // 1, 2, 3, 4
      pixid = calculateBPixID(layer);
    } else if (subid == PixelSubdetector::PixelEndcap) {
      int side = trackTopo->pxfSide(detId);  // 1 (-z), 2 for (+z)
      int disk = trackTopo->pxfDisk(detId);  // 1, 2, 3
      int ring(0);
      switch (ph) {
        case phase::zero:
          ring = SiPixelPI::ring(detid, *trackTopo, false);  // 1 (lower), 2 (upper)
          break;
        case phase::one:
          ring = SiPixelPI::ring(detid, *trackTopo, true);  // 1 (lower), 2 (upper)
          break;
        case phase::two:
          // I know, that's funny but go look at:
          // https://github.com/cms-sw/cmssw/blob/master/Geometry/TrackerNumberingBuilder/README.md
          ring = trackTopo->pxfBlade(detId);
          break;
        default:
          throw cms::Exception("LogicalError") << " there is not such phase as " << ph;
      }
      pixid = calculateFPixID(ph, side, disk, ring);
    }
    PixelId pixID = static_cast<PixelId>(pixid);
    return pixID;
  }

  //============================================================================
  [[maybe_unused]] static const std::vector<uint32_t> attachedDets(const PixelRegions::PixelId theId,
                                                                   const TrackerTopology* trackTopo,
                                                                   const SiPixelPI::phase& ph) {
    using namespace SiPixelPI;
    std::vector<uint32_t> out = {};
    edm::FileInPath m_fp;

    switch (ph) {
      case phase::zero:
        // phase-1 skimmed geometry from release
        m_fp = edm::FileInPath("CalibTracker/SiPixelESProducers/data/PixelSkimmedGeometry.txt");
        break;
      case phase::one:
        // phase-0 skimmed geometry from release
        m_fp = edm::FileInPath("SLHCUpgradeSimulations/Geometry/data/PhaseI/PixelSkimmedGeometry_phase1.txt");
        break;
      case phase::two:
        m_fp = edm::FileInPath("SLHCUpgradeSimulations/Geometry/data/PhaseII/Tilted/PixelSkimmedGeometryT14.txt");
        break;
      default:
        throw cms::Exception("LogicalError") << " there is not such phase as " << ph;
    }

    SiPixelDetInfoFileReader pxlreader(m_fp.fullPath());
    const std::vector<uint32_t>& pxldetids = pxlreader.getAllDetIds();
    for (const auto& d : pxldetids) {
      auto ID = detIdToPixelId(d, trackTopo, ph);
      if (ID == theId) {
        out.push_back(d);
      }
    }

    // in case no DetIds are assigned, fill with UINT32_MAX
    // in order to be able to tell a part the default case
    // in SiPixelGainCalibHelper::fillTheHitsto
    if (out.empty()) {
      out.push_back(0xFFFFFFFF);
    }

    COUT << "ID:" << theId << " ";
    for (const auto& entry : out) {
      COUT << entry << ",";
    }
    COUT << std::endl;

    return out;
  }

  /*--------------------------------------------------------------------
  / Ancillary class to build pixel phase0/phase1 plots per region
  /--------------------------------------------------------------------*/
  class PixelRegionContainers {
  public:
    PixelRegionContainers(const TrackerTopology* t_topo, const SiPixelPI::phase& ph)
        : m_trackerTopo(t_topo), m_Phase(ph) {
      // set log scale by default to false
      m_isLog = false;
      if (m_trackerTopo) {
        m_isTrackerTopologySet = true;
      } else {
        m_isTrackerTopologySet = false;
      }
    }

    ~PixelRegionContainers() {}

    //============================================================================
    void setTheTopo(const TrackerTopology* t_topo) {
      m_trackerTopo = t_topo;
      m_isTrackerTopologySet = true;
    }

    //============================================================================
    const TrackerTopology* getTheTopo() { return m_trackerTopo; }

    //============================================================================
    void bookAll(std::string title_label,
                 std::string x_label,
                 std::string y_label,
                 const int nbins,
                 const float xmin,
                 const float xmax) {
      using namespace SiPixelPI;
      for (const auto& pixelId : PixelIDs | boost::adaptors::indexed(0)) {
        if (m_Phase == phase::two &&            // if that's phase-2
            pixelId.value() > PixelId::L4 &&    // if it's end-cap
            pixelId.value() < PixelId::Ph2EmR1  // it's a phase-1 ring
        ) {
          continue;
        }

        m_theMap[pixelId.value()] = std::make_shared<TH1F>((title_label + itoa(pixelId.value())).c_str(),
                                                           Form("%s %s;%s;%s",
                                                                (IDlabels.at(pixelId.index())).c_str(),
                                                                title_label.c_str(),
                                                                x_label.c_str(),
                                                                y_label.c_str()),
                                                           nbins,
                                                           xmin,
                                                           xmax);
      }
    }

    //============================================================================
    void fill(const unsigned int detid, const float value) {
      // first check that the topology is set
      assert(m_trackerTopo);

      // convert from detid to pixelid
      PixelRegions::PixelId myId = PixelRegions::detIdToPixelId(detid, m_trackerTopo, m_Phase);
      if (m_theMap.find(myId) != m_theMap.end()) {
        m_theMap[myId]->Fill(value);
      } else {
        edm::LogError("PixelRegionContainers")
            << detid << " :=> " << myId << " is not a recongnized PixelId enumerator! \n"
            << m_trackerTopo->print(detid);
      }
    }

    //============================================================================
    void draw(TCanvas& canv, bool isBarrel, const char* option = "bar2", bool isPhase1Comparison = false) {
      using namespace SiPixelPI;
      if (isBarrel) {
        if (canv.GetListOfPrimitives()->GetSize() == 0) {
          // divide only if it was not already divided
          canv.Divide(2, 2);
        }
        for (int j = 1; j <= 4; j++) {
          if (!m_isLog) {
            canv.cd(j);
          } else {
            canv.cd(j)->SetLogy();
          }
          if ((j == 4) && (m_Phase < 1) && !isPhase1Comparison) {
            m_theMap.at(PixelIDs[j - 1])->Draw("AXIS");
            TLatex t2;
            t2.SetTextAlign(22);
            t2.SetTextSize(0.1);
            t2.SetTextAngle(45);
            t2.SetTextFont(61);
            t2.SetTextColor(kBlack);
            t2.DrawLatexNDC(0.5, 0.5, "Not in Phase-0!");
          } else {
            m_theMap.at(PixelIDs[j - 1])->Draw(option);
          }
        }
      } else {  // forward
        // pattern where to insert the plots in the canvas
        // clang-format off
        const std::array<int, 18> phase2Pattern = {{1,  2,  3,  4, /**/
                                                    6,  7,  8,  9, /**/
                                                    11, 12, 13, 14, 15,
                                                    16, 17, 18, 19, 20}};
        // clang-format on

        if (canv.GetListOfPrimitives()->GetSize() == 0) {
          canv.Divide(m_Phase == phase::two ? 5 : 4, m_Phase == phase::two ? 4 : 3);
        }
        const int maxPads = (m_Phase == phase::two) ? 18 : 12;
        for (int j = 1; j <= maxPads; j++) {
          unsigned int mapIndex = m_Phase == 2 ? j + 15 : j + 3;
          if (!m_isLog) {
            canv.cd((m_Phase == phase::two) ? phase2Pattern[j - 1] : j);
            //canv.cd(j);
          } else {
            canv.cd(j)->SetLogy();
          }
          if ((j % 6 == 5 || j % 6 == 0) && (m_Phase < 1) && !isPhase1Comparison) {
            m_theMap.at(PixelIDs[mapIndex])->Draw("AXIS");
            TLatex t2;
            t2.SetTextAlign(22);
            t2.SetTextSize(0.1);
            t2.SetTextAngle(45);
            t2.SetTextFont(61);
            t2.SetTextColor(kBlack);
            t2.DrawLatexNDC(0.5, 0.5, "Not in Phase-0!");
          } else {
            m_theMap.at(PixelIDs[mapIndex])->Draw(option);
          }
        }
      }
    }

    //============================================================================
    void beautify(const int linecolor = kBlack, const int fillcolor = kRed) {
      for (const auto& plot : m_theMap) {
        plot.second->SetTitle("");
        if (!m_isLog) {
          plot.second->GetYaxis()->SetRangeUser(0., plot.second->GetMaximum() * 1.30);
        } else {
          plot.second->GetYaxis()->SetRangeUser(0.1, plot.second->GetMaximum() * 100.);
        }
        plot.second->SetLineColor(linecolor);
        if (fillcolor > 0) {
          plot.second->SetFillColor(fillcolor);
        }
        plot.second->SetMarkerStyle(20);
        plot.second->SetMarkerSize(1);
        SiPixelPI::makeNicePlotStyle(plot.second.get());
        plot.second->SetStats(true);
      }
    }

    //============================================================================
    void setLogScale() { m_isLog = true; }

    //============================================================================
    void stats(int index = 0) {
      for (const auto& plot : m_theMap) {
        TPaveStats* st = (TPaveStats*)plot.second->FindObject("stats");
        if (st) {
          st->SetTextSize(0.03);
          st->SetLineColor(10);
          if (plot.second->GetFillColor() != 0) {
            st->SetTextColor(plot.second->GetFillColor());
          } else {
            st->SetTextColor(plot.second->GetLineColor());
          }
          SiPixelPI::adjustStats(st, 0.13, 0.85 - index * 0.08, 0.36, 0.93 - index * 0.08);
        }
      }
    }

    //============================================================================
    std::shared_ptr<TH1F>& getHistoFromMap(const PixelRegions::PixelId& theId) {
      auto it = m_theMap.find(theId);
      if (it != m_theMap.end()) {
        return it->second;
      } else {
        throw cms::Exception("LogicError")
            << "PixelRegionContainer::getHistoFromMap(): No histogram is available for PixelId: " << theId << "\n";
      }
    }

    //============================================================================
    void rescaleMax(PixelRegionContainers& the2ndContainer) {
      for (const auto& plot : m_theMap) {
        auto thePixId = plot.first;
        auto extrema = SiPixelPI::getExtrema((plot.second).get(), the2ndContainer.getHistoFromMap(thePixId).get());
        plot.second->GetYaxis()->SetRangeUser(extrema.first, extrema.second);
        the2ndContainer.getHistoFromMap(thePixId)->GetYaxis()->SetRangeUser(extrema.first, extrema.second);
      }
    }

  private:
    const TrackerTopology* m_trackerTopo;
    bool m_isTrackerTopologySet;
    SiPixelPI::phase m_Phase;
    std::map<PixelId, std::shared_ptr<TH1F>> m_theMap;
    int m_nbins;
    float m_xim, m_xmax;
    bool m_isLog;
  };
}  // namespace PixelRegions
#endif
