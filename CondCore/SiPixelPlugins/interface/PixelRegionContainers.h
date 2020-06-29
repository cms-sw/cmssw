#ifndef CONDCORE_SIPIXELPLUGINS_PIXELREGIONCONTAINERS_H
#define CONDCORE_SIPIXELPLUGINS_PIXELREGIONCONTAINERS_H

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <boost/range/adaptor/indexed.hpp>
#include "TH1.h"
#include <cstdlib>

namespace PixelRegions {

  std::string itoa(int i) {
    char temp[20];
    sprintf(temp, "%d", i);
    return ((std::string)temp);
  }

  // "PixelId"
  // BPix: 1000*(subdetId=1) + 100*(layer=1,2,3,4)
  // FPix: 1000*(subdetId=2) + 100*(side=1,2) + 10*(disk=1,2,3) + 1*(ring=1,2)

  enum PixelId {
    L1 = 1100,  // BPix
    L2 = 1200,
    L3 = 1300,
    L4 = 1400,
    Rm1l = 2111,  // FPix minus
    Rm1u = 2112,
    Rm2l = 2121,
    Rm2u = 2122,
    Rm3l = 2131,
    Rm3u = 2132,
    Rp1l = 2211,  // FPix plus
    Rp1u = 2212,
    Rp2l = 2221,
    Rp2u = 2222,
    Rp3l = 2231,
    Rp3u = 2232,
    End = 99999
  };

  const std::vector<PixelId> PixelIDs = {PixelId::L1,  // BPix
                                         PixelId::L2,
                                         PixelId::L3,
                                         PixelId::L4,
                                         PixelId::Rm1l,  // FPix minus
                                         PixelId::Rm1u,
                                         PixelId::Rm2l,
                                         PixelId::Rm2u,
                                         PixelId::Rm3l,
                                         PixelId::Rm3u,
                                         PixelId::Rp1l,  // FPix plus
                                         PixelId::Rp1u,
                                         PixelId::Rp2l,
                                         PixelId::Rp2u,
                                         PixelId::Rp3l,
                                         PixelId::Rp3u,
                                         PixelId::End};

  const std::vector<std::string> IDlabels = {"Barrel Pixel L1",
                                             "Barrel Pixel L2",
                                             "Barrel Pixel L3",
                                             "Barrel Pixel L4",
                                             "FPIX(-) Disk 1 inner ring",
                                             "FPIX(-) Disk 1 outer ring",
                                             "FPIX(-) Disk 2 inner ring",
                                             "FPIX(-) Disk 2 outer ring",
                                             "FPIX(-) Disk 3 inner ring",
                                             "FPIX(-) Disk 3 outer ring",
                                             "FPIX(+) Disk 1 inner ring",
                                             "FPIX(+) Disk 1 outer ring",
                                             "FPIX(+) Disk 2 inner ring",
                                             "FPIX(+) Disk 2 outer ring",
                                             "FPIX(+) Disk 3 inner ring",
                                             "FPIX(+) Disk 3 outer ring",
                                             "END"};

  //============================================================================
  static const PixelId calculateBPixID(const unsigned int layer) {
    // BPix: 1000*(subdetId=1) + 100*(layer=1,2,3,4)
    PixelId bpixLayer = static_cast<PixelId>(1000 + 100 * layer);
    return bpixLayer;
  }

  //============================================================================
  static const PixelId calculateFPixID(const unsigned int side, const unsigned int disk, const unsigned int ring) {
    // FPix: 1000*(subdetId=2) + 100*(side=1,2) + 10*(disk=1,2,3) + 1*(ring=1,2)
    PixelId fpixRing = static_cast<PixelId>(2000 + 100 * side + 10 * disk + ring);
    return fpixRing;
  }

  //============================================================================
  static const PixelId detIdToPixelId(const unsigned int detid, const TrackerTopology* trackTopo, const bool phase1) {
    DetId detId = DetId(detid);
    unsigned int subid = detId.subdetId();
    unsigned int pixid = 0;
    if (subid == PixelSubdetector::PixelBarrel) {
      PixelBarrelName bpix(detId, trackTopo, phase1);
      int layer = bpix.layerName();
      pixid = calculateBPixID(layer);
    } else if (subid == PixelSubdetector::PixelEndcap) {
      PixelEndcapName fpix(detId, trackTopo, phase1);
      int side = trackTopo->pxfSide(detId);  // 1 (-z), 2 for (+z)
      int disk = fpix.diskName();            //trackTopo->pxfDisk(detId); // 1, 2, 3
      // This works only in case of the Phase-1 detector
      //int ring = fpix.ringName();          // 1 (lower), 2 (upper)
      int ring = SiPixelPI::ring(detid, *trackTopo, phase1);  // 1 (lower), 2 (upper)
      pixid = calculateFPixID(side, disk, ring);
    }
    PixelId pixID = static_cast<PixelId>(pixid);
    return pixID;
  }

  //============================================================================
  const std::vector<uint32_t> attachedDets(const PixelRegions::PixelId theId,
                                           const TrackerTopology* trackTopo,
                                           const bool phase1) {
    std::vector<uint32_t> out = {};
    edm::FileInPath m_fp;
    if (phase1) {
      // phase-1 skimmed geometry from release
      m_fp = edm::FileInPath("SLHCUpgradeSimulations/Geometry/data/PhaseI/PixelSkimmedGeometry_phase1.txt");
    } else {
      // phase-0 skimmed geometry from release
      m_fp = edm::FileInPath("CalibTracker/SiPixelESProducers/data/PixelSkimmedGeometry.txt");
    }

    SiPixelDetInfoFileReader pxlreader(m_fp.fullPath());
    const std::vector<uint32_t>& pxldetids = pxlreader.getAllDetIds();
    for (const auto& d : pxldetids) {
      auto ID = detIdToPixelId(d, trackTopo, phase1);
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
    PixelRegionContainers(const TrackerTopology* t_topo, const bool isPhase1)
        : m_trackerTopo(t_topo), m_isPhase1(isPhase1) {
      // set log scale by default to false
      m_isLog = false;
    }

    ~PixelRegionContainers() {}

    //============================================================================
    void bookAll(std::string title_label,
                 std::string x_label,
                 std::string y_label,
                 const int nbins,
                 const float xmin,
                 const float xmax) {
      for (const auto& pixelId : PixelIDs | boost::adaptors::indexed(0)) {
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
      // convert from detid to pixelid
      PixelRegions::PixelId myId = PixelRegions::detIdToPixelId(detid, m_trackerTopo, m_isPhase1);
      if (m_theMap.find(myId) != m_theMap.end()) {
        m_theMap[myId]->Fill(value);
      } else {
        edm::LogError("PixelRegionContainers")
            << detid << " :=> " << myId << " is not a recongnized PixelId enumerator!" << std::endl;
      }
    }

    //============================================================================
    void draw(TCanvas& canv, bool isBarrel, const char* option = "bar2", bool isPhase1Comparison = false) {
      if (isBarrel) {
        for (int j = 1; j <= 4; j++) {
          if (!m_isLog) {
            canv.cd(j);
          } else {
            canv.cd(j)->SetLogy();
          }
          if ((j == 4) && !m_isPhase1 && !isPhase1Comparison) {
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
        for (int j = 1; j <= 12; j++) {
          if (!m_isLog) {
            canv.cd(j);
          } else {
            canv.cd(j)->SetLogy();
          }
          if ((j % 6 == 5 || j % 6 == 0) && !m_isPhase1 && !isPhase1Comparison) {
            m_theMap.at(PixelIDs[j + 3])->Draw("AXIS");
            TLatex t2;
            t2.SetTextAlign(22);
            t2.SetTextSize(0.1);
            t2.SetTextAngle(45);
            t2.SetTextFont(61);
            t2.SetTextColor(kBlack);
            t2.DrawLatexNDC(0.5, 0.5, "Not in Phase-0!");
          } else {
            m_theMap.at(PixelIDs[j + 3])->Draw(option);
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
        throw cms::Exception("PixelRegionContainer") << "No histogram is available for PixelId" << theId << "\n";
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
    bool m_isPhase1;

    std::map<PixelId, std::shared_ptr<TH1F>> m_theMap;
    int m_nbins;
    float m_xim, m_xmax;
    bool m_isLog;
  };
}  // namespace PixelRegions
#endif
