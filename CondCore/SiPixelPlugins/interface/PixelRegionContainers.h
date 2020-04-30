#ifndef CONDCORE_SIPIXELPLUGINS_PIXELREGIONCONTAINERS_H
#define CONDCORE_SIPIXELPLUGINS_PIXELREGIONCONTAINERS_H

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"

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
    L1 = 1100,
    L2 = 1200,
    L3 = 1300,
    L4 = 1400,  // BPix
    Rm1l = 2111,
    Rm1u = 2112,
    Rm2l = 2121,
    Rm2u = 2122,
    Rm3l = 2131,
    Rm3u = 2132,  // FPix minus
    Rp1l = 2211,
    Rp1u = 2212,
    Rp2l = 2221,
    Rp2u = 2222,
    Rp3l = 2231,
    Rp3u = 2232,  // FPix plus
    End = 99999
  };

  const std::vector<PixelId> PixelIDs = {PixelId::L1,
                                         PixelId::L2,
                                         PixelId::L3,
                                         PixelId::L4,  // BPix
                                         PixelId::Rm1l,
                                         PixelId::Rm1u,
                                         PixelId::Rm2l,
                                         PixelId::Rm2u,
                                         PixelId::Rm3l,
                                         PixelId::Rm3u,  // FPix minus
                                         PixelId::Rp1l,
                                         PixelId::Rp1u,
                                         PixelId::Rp2l,
                                         PixelId::Rp2u,
                                         PixelId::Rp3l,
                                         PixelId::Rp3u,  // FPix plus
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
                                             "FPIX(+) Disk 3 outer ring"};

  static const PixelId calculateBPixID(const unsigned int layer) {
    // BPix: 1000*(subdetId=1) + 100*(layer=1,2,3,4)
    PixelId bpixLayer = static_cast<PixelId>(1000 + 100 * layer);
    return bpixLayer;
  }

  static const PixelId calculateFPixID(const unsigned int side, const unsigned int disk, const unsigned int ring) {
    // FPix: 1000*(subdetId=2) + 100*(side=1,2) + 10*(disk=1,2,3) + 1*(ring=1,2)
    PixelId fpixRing = static_cast<PixelId>(2000 + 100 * side + 10 * disk + ring);
    return fpixRing;
  }

  static const int getPixelSubDetector(const unsigned int pixid) {
    // subdetId: BPix=1, FPix=2
    return (pixid / 1000) % 10;
  }

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
      int ring = fpix.ringName();            // 1 (lower), 2 (upper)
      pixid = calculateFPixID(side, disk, ring);
    }
    PixelId pixID = static_cast<PixelId>(pixid);
    return pixID;
  }

  class PixelRegionContainers {
  public:
    PixelRegionContainers(const TrackerTopology* t_topo, const bool isPhase1)
        : m_trackerTopo(t_topo), m_isPhase1(isPhase1) {}

    ~PixelRegionContainers() {}

    void bookAll(std::string title_label,
                 std::string x_label,
                 std::string y_label,
                 const int nbins,
                 const float xmin,
                 const float xmax) {
      for (int i = PixelId::L1; i < PixelId::End; i++) {
        switch (i) {
          case PixelId::L1:
            m_theMap[PixelId::L1] = std::make_shared<TH1F>(
                itoa(i).c_str(),
                Form("Barrel Pixel L1 %s;%s;%s", title_label.c_str(), x_label.c_str(), y_label.c_str()),
                nbins,
                xmin,
                xmax);
            break;
          case PixelId::L2:
            m_theMap[PixelId::L2] = std::make_shared<TH1F>(
                itoa(i).c_str(),
                Form("Barrel Pixel L2 %s;%s;%s", title_label.c_str(), x_label.c_str(), y_label.c_str()),
                nbins,
                xmin,
                xmax);
            break;
          case PixelId::L3:
            m_theMap[PixelId::L3] = std::make_shared<TH1F>(
                itoa(i).c_str(),
                Form("Barrel Pixel L3 %s;%s;%s", title_label.c_str(), x_label.c_str(), y_label.c_str()),
                nbins,
                xmin,
                xmax);
            break;
          case PixelId::L4:
            m_theMap[PixelId::L4] = std::make_shared<TH1F>(
                itoa(i).c_str(),
                Form("Barrel Pixel L4 %s;%s;%s", title_label.c_str(), x_label.c_str(), y_label.c_str()),
                nbins,
                xmin,
                xmax);
            break;
          case PixelId::Rm1l:
            m_theMap[PixelId::Rm1l] = std::make_shared<TH1F>(
                itoa(i).c_str(),
                Form("FPIX(-) Disk 1 inner ring %s;%s;%s", title_label.c_str(), x_label.c_str(), y_label.c_str()),
                nbins,
                xmin,
                xmax);
            break;
          case PixelId::Rm1u:
            m_theMap[PixelId::Rm1u] = std::make_shared<TH1F>(
                itoa(i).c_str(),
                Form("FPIX(-) Disk 1 outer ring %s;%s;%s", title_label.c_str(), x_label.c_str(), y_label.c_str()),
                nbins,
                xmin,
                xmax);
            break;
          case PixelId::Rm2l:
            m_theMap[PixelId::Rm2l] = std::make_shared<TH1F>(
                itoa(i).c_str(),
                Form("FPIX(-) Disk 2 inner ring %s;%s;%s", title_label.c_str(), x_label.c_str(), y_label.c_str()),
                nbins,
                xmin,
                xmax);
            break;
          case PixelId::Rm2u:
            m_theMap[PixelId::Rm2u] = std::make_shared<TH1F>(
                itoa(i).c_str(),
                Form("FPIX(-) Disk 2 outer ring %s;%s;%s", title_label.c_str(), x_label.c_str(), y_label.c_str()),
                nbins,
                xmin,
                xmax);
            break;
          case PixelId::Rm3l:
            m_theMap[PixelId::Rm3l] = std::make_shared<TH1F>(
                itoa(i).c_str(),
                Form("FPIX(-) Disk 3 inner ring %s;%s;%s", title_label.c_str(), x_label.c_str(), y_label.c_str()),
                nbins,
                xmin,
                xmax);
            break;
          case PixelId::Rm3u:
            m_theMap[PixelId::Rm3u] = std::make_shared<TH1F>(
                itoa(i).c_str(),
                Form("FPIX(-) Disk 3 outer ring %s;%s;%s", title_label.c_str(), x_label.c_str(), y_label.c_str()),
                nbins,
                xmin,
                xmax);
            break;
          case PixelId::Rp1l:
            m_theMap[PixelId::Rp1l] = std::make_shared<TH1F>(
                itoa(i).c_str(),
                Form("FPIX(+) Disk 1 inner ring %s;%s;%s", title_label.c_str(), x_label.c_str(), y_label.c_str()),
                nbins,
                xmin,
                xmax);
            break;
          case PixelId::Rp1u:
            m_theMap[PixelId::Rp1u] = std::make_shared<TH1F>(
                itoa(i).c_str(),
                Form("FPIX(+) Disk 1 outer ring %s;%s;%s", title_label.c_str(), x_label.c_str(), y_label.c_str()),
                nbins,
                xmin,
                xmax);
            break;
          case PixelId::Rp2l:
            m_theMap[PixelId::Rp2l] = std::make_shared<TH1F>(
                itoa(i).c_str(),
                Form("FPIX(+) Disk 2 inner ring %s;%s;%s", title_label.c_str(), x_label.c_str(), y_label.c_str()),
                nbins,
                xmin,
                xmax);
            break;
          case PixelId::Rp2u:
            m_theMap[PixelId::Rp2u] = std::make_shared<TH1F>(
                itoa(i).c_str(),
                Form("FPIX(+) Disk 2 outer ring %s;%s;%s", title_label.c_str(), x_label.c_str(), y_label.c_str()),
                nbins,
                xmin,
                xmax);
            break;
          case PixelId::Rp3l:
            m_theMap[PixelId::Rp3l] = std::make_shared<TH1F>(
                itoa(i).c_str(),
                Form("FPIX(+) Disk 3 inner ring %s;%s;%s", title_label.c_str(), x_label.c_str(), y_label.c_str()),
                nbins,
                xmin,
                xmax);
            break;
          case PixelId::Rp3u:
            m_theMap[PixelId::Rp3u] = std::make_shared<TH1F>(
                itoa(i).c_str(),
                Form("FPIX(+) Disk 3 outer ring %s;%s;%s", title_label.c_str(), x_label.c_str(), y_label.c_str()),
                nbins,
                xmin,
                xmax);
            break;
          default:
            /* Found an invalid enum entry; ignore it */
            break;
        }
      }
    }

    void fill(const unsigned int detid, const float value) {
      // convert from detid to pixelid
      PixelRegions::PixelId myId = PixelRegions::detIdToPixelId(detid, m_trackerTopo, m_isPhase1);
      m_theMap[myId]->Fill(value);
    }

    void draw(TCanvas& canv, bool isBarrel) {
      if (isBarrel) {
        for (int j = 1; j <= 4; j++) {
          canv.cd(j);
          m_theMap.at(PixelIDs[j - 1])->Draw("bar2");
        }
      } else {  // forward
        for (int j = 1; j <= 12; j++) {
          canv.cd(j);
          m_theMap.at(PixelIDs[j + 3])->Draw("bar2");
        }
      }
    }

    void beautify() {
      for (const auto& plot : m_theMap) {
        plot.second->GetYaxis()->SetRangeUser(0., plot.second->GetMaximum() * 1.30);
        plot.second->SetFillColor(kRed);
        plot.second->SetMarkerStyle(20);
        plot.second->SetMarkerSize(1);
        SiPixelPI::makeNicePlotStyle(plot.second.get());
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
