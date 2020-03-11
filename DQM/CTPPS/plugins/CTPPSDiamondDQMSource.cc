/****************************************************************************
*
* This is a part of CTPPS offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*   Nicola Minafra
*   Laurent Forthomme
*
****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "FWCore/Framework/interface/Run.h"

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/CTPPSDigi/interface/TotemVFATStatus.h"
#include "DataFormats/CTPPSDigi/interface/TotemFEDInfo.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSDiamondDigi.h"

#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"

#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondLocalTrack.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

#include <string>

//----------------------------------------------------------------------------------------------------

// Utility for efficiency computations
bool channelAlignedWithTrack(const CTPPSGeometry* geom,
                             const CTPPSDiamondDetId& detid,
                             const CTPPSDiamondLocalTrack& localTrack,
                             const float tolerance = 1) {
  const DetGeomDesc* det = geom->sensor(detid);
  const float x_pos = det->translation().x(),
              x_width = 2.0 * det->params().at(0);  // parameters stand for half the size
  return ((x_pos + 0.5 * x_width > localTrack.x0() - localTrack.x0Sigma() - tolerance &&
           x_pos + 0.5 * x_width < localTrack.x0() + localTrack.x0Sigma() + tolerance) ||
          (x_pos - 0.5 * x_width > localTrack.x0() - localTrack.x0Sigma() - tolerance &&
           x_pos - 0.5 * x_width < localTrack.x0() + localTrack.x0Sigma() + tolerance) ||
          (x_pos - 0.5 * x_width < localTrack.x0() - localTrack.x0Sigma() - tolerance &&
           x_pos + 0.5 * x_width > localTrack.x0() + localTrack.x0Sigma() + tolerance));
}

namespace dds {
  struct Cache {
    std::unordered_map<unsigned int, std::unique_ptr<TH2F>> hitDistribution2dMap;

    std::unordered_map<unsigned int, unsigned long> hitsCounterMap;
  };
}  // namespace dds

class CTPPSDiamondDQMSource : public DQMOneEDAnalyzer<edm::LuminosityBlockCache<dds::Cache>> {
public:
  CTPPSDiamondDQMSource(const edm::ParameterSet&);
  ~CTPPSDiamondDQMSource() override;

protected:
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  std::shared_ptr<dds::Cache> globalBeginLuminosityBlock(const edm::LuminosityBlock&,
                                                         const edm::EventSetup&) const override;
  void globalEndLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;

private:
  // Constants
  static const double SEC_PER_LUMI_SECTION;  // Number of seconds per lumisection: used to compute hit rates in Hz
  static const int CHANNEL_OF_VFAT_CLOCK;    // Channel ID of the VFAT that contains clock data
  static const double DISPLAY_RESOLUTION_FOR_HITS_MM;  // Bin width of histograms showing hits and tracks (in mm)
  static const double INV_DISPLAY_RESOLUTION_FOR_HITS_MM;
  static const double HPTDC_BIN_WIDTH_NS;  // ns per HPTDC bin
  static const int CTPPS_NUM_OF_ARMS;
  static const int CTPPS_DIAMOND_STATION_ID;
  static const int CTPPS_DIAMOND_RP_ID;
  static const int CTPPS_PIXEL_STATION_ID;
  static const int CTPPS_NEAR_RP_ID;
  static const int CTPPS_FAR_RP_ID;
  static const int CTPPS_DIAMOND_NUM_OF_PLANES;
  static const int CTPPS_DIAMOND_NUM_OF_CHANNELS;
  static const int CTPPS_FED_ID_45;
  static const int CTPPS_FED_ID_56;

  edm::EDGetTokenT<edm::DetSetVector<TotemVFATStatus>> tokenStatus_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelLocalTrack>> tokenPixelTrack_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSDiamondDigi>> tokenDigi_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSDiamondRecHit>> tokenDiamondHit_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSDiamondLocalTrack>> tokenDiamondTrack_;
  edm::EDGetTokenT<std::vector<TotemFEDInfo>> tokenFEDInfo_;

  edm::ESGetToken<CTPPSGeometry, VeryForwardRealGeometryRecord> ctppsGeometryRunToken_;
  edm::ESGetToken<CTPPSGeometry, VeryForwardRealGeometryRecord> ctppsGeometryEventToken_;

  bool excludeMultipleHits_;
  double horizontalShiftBwDiamondPixels_;
  double horizontalShiftOfDiamond_;
  std::vector<std::pair<edm::EventRange, int>> runParameters_;
  int centralOOT_;
  unsigned int verbosity_;

  /// plots related to the whole system
  struct GlobalPlots {
    GlobalPlots() {}
    GlobalPlots(DQMStore::IBooker& ibooker);
  };

  GlobalPlots globalPlot_;

  /// plots related to one Diamond detector package
  struct PotPlots {
    std::unordered_map<unsigned int, MonitorElement*> activity_per_bx;

    MonitorElement* hitDistribution2d = nullptr;
    MonitorElement* hitDistribution2d_lumisection = nullptr;
    MonitorElement* hitDistribution2dOOT = nullptr;
    MonitorElement* hitDistribution2dOOT_le = nullptr;
    MonitorElement *activePlanes = nullptr, *activePlanesInclusive = nullptr;

    MonitorElement* trackDistribution = nullptr;
    MonitorElement* trackDistributionOOT = nullptr;

    std::unordered_map<unsigned int, MonitorElement*> pixelTomographyAll;

    MonitorElement *leadingEdgeCumulative_both = nullptr, *leadingEdgeCumulative_all = nullptr,
                   *leadingEdgeCumulative_le = nullptr, *trailingEdgeCumulative_te = nullptr;
    MonitorElement *timeOverThresholdCumulativePot = nullptr, *leadingTrailingCorrelationPot = nullptr;
    MonitorElement* leadingWithoutTrailingCumulativePot = nullptr;

    MonitorElement* ECCheck = nullptr;

    MonitorElement* HPTDCErrorFlags_2D = nullptr;
    MonitorElement* MHComprensive = nullptr;

    // MonitorElement* clock_Digi1_le = nullptr;
    // MonitorElement* clock_Digi1_te = nullptr;
    // MonitorElement* clock_Digi3_le = nullptr;
    // MonitorElement* clock_Digi3_te = nullptr;

    unsigned int HitCounter, MHCounter, LeadingOnlyCounter, TrailingOnlyCounter, CompleteCounter;

    std::map<int, int> effTriplecountingChMap;
    std::map<int, int> effDoublecountingChMap;
    MonitorElement* EfficiencyOfChannelsInPot = nullptr;
    TH2F pixelTracksMap;

    PotPlots() {}
    PotPlots(DQMStore::IBooker& ibooker, unsigned int id);
  };

  std::unordered_map<unsigned int, PotPlots> potPlots_;
  int EC_difference_56_, EC_difference_45_;

  /// plots related to one Diamond plane
  struct PlanePlots {
    MonitorElement* digiProfileCumulativePerPlane = nullptr;
    MonitorElement* hitProfile = nullptr;
    MonitorElement* hit_multiplicity = nullptr;

    MonitorElement* pixelTomography_far = nullptr;
    MonitorElement* EfficiencyWRTPixelsInPlane = nullptr;

    TH2F pixelTracksMapWithDiamonds;

    PlanePlots() {}
    PlanePlots(DQMStore::IBooker& ibooker, unsigned int id);
  };

  std::unordered_map<unsigned int, PlanePlots> planePlots_;

  /// plots related to one Diamond channel
  struct ChannelPlots {
    std::unordered_map<unsigned int, MonitorElement*> activity_per_bx;

    MonitorElement* HPTDCErrorFlags = nullptr;
    MonitorElement *leadingEdgeCumulative_both = nullptr, *leadingEdgeCumulative_le = nullptr,
                   *trailingEdgeCumulative_te = nullptr;
    MonitorElement* TimeOverThresholdCumulativePerChannel = nullptr;
    MonitorElement* LeadingTrailingCorrelationPerChannel = nullptr;
    MonitorElement* leadingWithoutTrailing = nullptr;
    MonitorElement* pixelTomography_far = nullptr;
    MonitorElement* hit_rate = nullptr;

    unsigned int HitCounter, MHCounter, LeadingOnlyCounter, TrailingOnlyCounter, CompleteCounter;

    ChannelPlots() {}
    ChannelPlots(DQMStore::IBooker& ibooker, unsigned int id);
  };

  std::unordered_map<unsigned int, ChannelPlots> channelPlots_;
};

//----------------------------------------------------------------------------------------------------

// Values for all constants
const double CTPPSDiamondDQMSource::SEC_PER_LUMI_SECTION = 23.31;
const int CTPPSDiamondDQMSource::CHANNEL_OF_VFAT_CLOCK = 30;
const double CTPPSDiamondDQMSource::DISPLAY_RESOLUTION_FOR_HITS_MM = 0.1;
const double CTPPSDiamondDQMSource::INV_DISPLAY_RESOLUTION_FOR_HITS_MM = 1. / DISPLAY_RESOLUTION_FOR_HITS_MM;
const double CTPPSDiamondDQMSource::HPTDC_BIN_WIDTH_NS = 25. / 1024;
const int CTPPSDiamondDQMSource::CTPPS_NUM_OF_ARMS = 2;
const int CTPPSDiamondDQMSource::CTPPS_DIAMOND_STATION_ID = 1;
const int CTPPSDiamondDQMSource::CTPPS_PIXEL_STATION_ID = 2;
const int CTPPSDiamondDQMSource::CTPPS_DIAMOND_RP_ID = 6;
const int CTPPSDiamondDQMSource::CTPPS_NEAR_RP_ID = 2;
const int CTPPSDiamondDQMSource::CTPPS_FAR_RP_ID = 3;
const int CTPPSDiamondDQMSource::CTPPS_DIAMOND_NUM_OF_PLANES = 4;
const int CTPPSDiamondDQMSource::CTPPS_DIAMOND_NUM_OF_CHANNELS = 12;
const int CTPPSDiamondDQMSource::CTPPS_FED_ID_56 = 582;
const int CTPPSDiamondDQMSource::CTPPS_FED_ID_45 = 583;

//----------------------------------------------------------------------------------------------------

CTPPSDiamondDQMSource::GlobalPlots::GlobalPlots(DQMStore::IBooker& ibooker) { ibooker.setCurrentFolder("CTPPS"); }

//----------------------------------------------------------------------------------------------------

CTPPSDiamondDQMSource::PotPlots::PotPlots(DQMStore::IBooker& ibooker, unsigned int id)
    : HitCounter(0),
      MHCounter(0),
      LeadingOnlyCounter(0),
      TrailingOnlyCounter(0),
      CompleteCounter(0),
      pixelTracksMap("Pixel track maps for efficiency", "Pixel track maps for efficiency", 25, 0, 25, 12, -2, 10) {
  std::string path, title;
  CTPPSDiamondDetId(id).rpName(path, CTPPSDiamondDetId::nPath);
  ibooker.setCurrentFolder(path);

  CTPPSDiamondDetId(id).rpName(title, CTPPSDiamondDetId::nFull);

  activity_per_bx[0] =
      ibooker.book1D("activity per BX 0 25", title + " Activity per BX 0 - 25 ns;Event.BX", 3600, -1.5, 3598. + 0.5);
  activity_per_bx[1] =
      ibooker.book1D("activity per BX 25 50", title + " Activity per BX 25 - 50 ns;Event.BX", 3600, -1.5, 3598. + 0.5);
  activity_per_bx[2] =
      ibooker.book1D("activity per BX 50 75", title + " Activity per BX 50 - 75 ns;Event.BX", 3600, -1.5, 3598. + 0.5);

  hitDistribution2d = ibooker.book2D("hits in planes",
                                     title + " hits in planes;plane number;x (mm)",
                                     10,
                                     -0.5,
                                     4.5,
                                     19. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM,
                                     -0.5,
                                     18.5);
  hitDistribution2d_lumisection = ibooker.book2D("hits in planes lumisection",
                                                 title + " hits in planes in the last lumisection;plane number;x (mm)",
                                                 10,
                                                 -0.5,
                                                 4.5,
                                                 19. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM,
                                                 -0.5,
                                                 18.5);
  hitDistribution2dOOT = ibooker.book2D("hits with OOT in planes",
                                        title + " hits with OOT in planes;plane number + 0.25 OOT;x (mm)",
                                        17,
                                        -0.25,
                                        4,
                                        19. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM,
                                        -0.5,
                                        18.5);
  hitDistribution2dOOT_le = ibooker.book2D("hits with OOT in planes (le only)",
                                           title + " hits with OOT in planes (le only);plane number + 0.25 OOT;x (mm)",
                                           17,
                                           -0.25,
                                           4,
                                           19. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM,
                                           -0.5,
                                           18.5);
  activePlanes =
      ibooker.book1D("active planes", title + " active planes (per event);number of active planes", 6, -0.5, 5.5);
  activePlanesInclusive =
      ibooker.book1D("active planes inclusive",
                     title + " active planes, MH and le only included (per event);number of active planes",
                     6,
                     -0.5,
                     5.5);

  trackDistribution =
      ibooker.book1D("tracks", title + " tracks;x (mm)", 19. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM, -0.5, 18.5);
  trackDistributionOOT = ibooker.book2D("tracks with OOT",
                                        title + " tracks with OOT;plane number;x (mm)",
                                        9,
                                        -0.5,
                                        4,
                                        19. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM,
                                        -0.5,
                                        18.5);

  pixelTomographyAll[0] =
      ibooker.book2D("tomography pixel 0 25",
                     title + " tomography with pixel 0 - 25 ns (all planes);x + 25*plane(mm);y (mm)",
                     100,
                     0,
                     100,
                     8,
                     0,
                     8);
  pixelTomographyAll[1] =
      ibooker.book2D("tomography pixel 25 50",
                     title + " tomography with pixel 25 - 50 ns (all planes);x + 25*plane(mm);y (mm)",
                     100,
                     0,
                     100,
                     8,
                     0,
                     8);
  pixelTomographyAll[2] =
      ibooker.book2D("tomography pixel 50 75",
                     title + " tomography with pixel 50 - 75 ns (all planes);x + 25*plane(mm);y (mm)",
                     100,
                     0,
                     100,
                     8,
                     0,
                     8);

  leadingEdgeCumulative_both = ibooker.book1D(
      "leading edge (le and te)", title + " leading edge (le and te) (recHits); leading edge (ns)", 75, 0, 75);
  leadingEdgeCumulative_all = ibooker.book1D(
      "leading edge (all)", title + " leading edge (with or without te) (DIGIs); leading edge (ns)", 75, 0, 75);
  leadingEdgeCumulative_le =
      ibooker.book1D("leading edge (le only)", title + " leading edge (le only) (DIGIs); leading edge (ns)", 75, 0, 75);
  trailingEdgeCumulative_te = ibooker.book1D(
      "trailing edge (te only)", title + " trailing edge (te only) (DIGIs); trailing edge (ns)", 75, 0, 75);
  timeOverThresholdCumulativePot =
      ibooker.book1D("time over threshold", title + " time over threshold;time over threshold (ns)", 250, -25, 100);
  leadingTrailingCorrelationPot =
      ibooker.book2D("leading trailing correlation",
                     title + " leading trailing correlation;leading edge (ns);trailing edge (ns)",
                     75,
                     0,
                     75,
                     75,
                     0,
                     75);

  leadingWithoutTrailingCumulativePot =
      ibooker.book1D("event category", title + " leading edges without trailing;;%", 3, 0.5, 3.5);
  leadingWithoutTrailingCumulativePot->setBinLabel(1, "Leading only");
  leadingWithoutTrailingCumulativePot->setBinLabel(2, "Trailing only");
  leadingWithoutTrailingCumulativePot->setBinLabel(3, "Both");

  ECCheck = ibooker.book1D("optorxEC(8bit) - vfatEC", title + " EC Error;optorxEC-vfatEC", 50, -25, 25);

  HPTDCErrorFlags_2D = ibooker.book2D("HPTDC Errors", title + " HPTDC Errors", 16, -0.5, 16.5, 9, -0.5, 8.5);
  for (unsigned short error_index = 1; error_index < 16; ++error_index)
    HPTDCErrorFlags_2D->setBinLabel(error_index, HPTDCErrorFlags::hptdcErrorName(error_index - 1));
  HPTDCErrorFlags_2D->setBinLabel(16, "Wrong EC");

  int tmpIndex = 0;
  HPTDCErrorFlags_2D->setBinLabel(++tmpIndex, "DB 0 TDC 18", /* axis */ 2);
  HPTDCErrorFlags_2D->setBinLabel(++tmpIndex, "DB 0 TDC 17", /* axis */ 2);
  HPTDCErrorFlags_2D->setBinLabel(++tmpIndex, "DB 0 TDC 16", /* axis */ 2);
  HPTDCErrorFlags_2D->setBinLabel(++tmpIndex, "DB 0 TDC 15", /* axis */ 2);
  HPTDCErrorFlags_2D->setBinLabel(++tmpIndex, "DB 1 TDC 18", /* axis */ 2);
  HPTDCErrorFlags_2D->setBinLabel(++tmpIndex, "DB 1 TDC 17", /* axis */ 2);
  HPTDCErrorFlags_2D->setBinLabel(++tmpIndex, "DB 1 TDC 16", /* axis */ 2);
  HPTDCErrorFlags_2D->setBinLabel(++tmpIndex, "DB 1 TDC 15", /* axis */ 2);

  MHComprensive =
      ibooker.book2D("MH in channels", title + " MH (%) in channels;plane number;ch number", 10, -0.5, 4.5, 14, -1, 13);

  EfficiencyOfChannelsInPot =
      ibooker.book2D("Efficiency in channels",
                     title + " Efficiency (%) in channels (diamonds only);plane number;ch number",
                     10,
                     -0.5,
                     4.5,
                     14,
                     -1,
                     13);

  // ibooker.setCurrentFolder( path+"/clock/" );
  // clock_Digi1_le = ibooker.book1D( "clock1 leading edge", title+" clock1;leading edge (ns)", 250, 0, 25 );
  // clock_Digi1_te = ibooker.book1D( "clock1 trailing edge", title+" clock1;trailing edge (ns)", 75, 0, 75 );
  // clock_Digi3_le = ibooker.book1D( "clock3 leading edge", title+" clock3;leading edge (ns)", 250, 0, 25 );
  // clock_Digi3_te = ibooker.book1D( "clock3 trailing edge", title+" clock3;trailing edge (ns)", 75, 0, 75 );
}

//----------------------------------------------------------------------------------------------------

CTPPSDiamondDQMSource::PlanePlots::PlanePlots(DQMStore::IBooker& ibooker, unsigned int id)
    : pixelTracksMapWithDiamonds("Pixel track maps for efficiency with coincidence",
                                 "Pixel track maps for efficiency with coincidence",
                                 25,
                                 0,
                                 25,
                                 12,
                                 -2,
                                 10) {
  std::string path, title;
  CTPPSDiamondDetId(id).planeName(path, CTPPSDiamondDetId::nPath);
  ibooker.setCurrentFolder(path);

  CTPPSDiamondDetId(id).planeName(title, CTPPSDiamondDetId::nFull);

  digiProfileCumulativePerPlane = ibooker.book1D("digi profile", title + " digi profile; ch number", 12, -0.5, 11.5);
  hitProfile = ibooker.book1D(
      "hit profile", title + " hit profile;x (mm)", 19. * INV_DISPLAY_RESOLUTION_FOR_HITS_MM, -0.5, 18.5);
  hit_multiplicity = ibooker.book1D("channels per plane", title + " channels per plane; ch per plane", 13, -0.5, 12.5);

  pixelTomography_far =
      ibooker.book2D("tomography pixel", title + " tomography with pixel;x + 25 OOT (mm);y (mm)", 75, 0, 75, 8, 0, 8);
  EfficiencyWRTPixelsInPlane =
      ibooker.book2D("Efficiency wrt pixels", title + " Efficiency wrt pixels;x (mm);y (mm)", 25, 0, 25, 12, -2, 10);
}

//----------------------------------------------------------------------------------------------------

CTPPSDiamondDQMSource::ChannelPlots::ChannelPlots(DQMStore::IBooker& ibooker, unsigned int id)
    : HitCounter(0), MHCounter(0), LeadingOnlyCounter(0), TrailingOnlyCounter(0), CompleteCounter(0) {
  std::string path, title;
  CTPPSDiamondDetId(id).channelName(path, CTPPSDiamondDetId::nPath);
  ibooker.setCurrentFolder(path);

  CTPPSDiamondDetId(id).channelName(title, CTPPSDiamondDetId::nFull);

  leadingWithoutTrailing = ibooker.book1D("event category", title + " Event Category;;%", 3, 0.5, 3.5);
  leadingWithoutTrailing->setBinLabel(1, "Leading only");
  leadingWithoutTrailing->setBinLabel(2, "Trailing only");
  leadingWithoutTrailing->setBinLabel(3, "Full");

  activity_per_bx[0] =
      ibooker.book1D("activity per BX 0 25", title + " Activity per BX 0 - 25 ns;Event.BX", 500, -1.5, 498. + 0.5);
  activity_per_bx[1] =
      ibooker.book1D("activity per BX 25 50", title + " Activity per BX 25 - 50 ns;Event.BX", 500, -1.5, 498. + 0.5);
  activity_per_bx[2] =
      ibooker.book1D("activity per BX 50 75", title + " Activity per BX 50 - 75 ns;Event.BX", 500, -1.5, 498. + 0.5);

  HPTDCErrorFlags = ibooker.book1D("hptdc_Errors", title + " HPTDC Errors", 16, -0.5, 16.5);
  for (unsigned short error_index = 1; error_index < 16; ++error_index)
    HPTDCErrorFlags->setBinLabel(error_index, HPTDCErrorFlags::hptdcErrorName(error_index - 1));
  HPTDCErrorFlags->setBinLabel(16, "MH  (%)");

  leadingEdgeCumulative_both =
      ibooker.book1D("leading edge (le and te)", title + " leading edge (recHits); leading edge (ns)", 75, 0, 75);
  leadingEdgeCumulative_le =
      ibooker.book1D("leading edge (le only)", title + " leading edge (DIGIs); leading edge (ns)", 75, 0, 75);
  trailingEdgeCumulative_te = ibooker.book1D(
      "trailing edge (te only)", title + " trailing edge (te only) (DIGIs); trailing edge (ns)", 75, 0, 75);
  TimeOverThresholdCumulativePerChannel =
      ibooker.book1D("time over threshold", title + " time over threshold;time over threshold (ns)", 75, -25, 50);
  LeadingTrailingCorrelationPerChannel =
      ibooker.book2D("leading trailing correlation",
                     title + " leading trailing correlation;leading edge (ns);trailing edge (ns)",
                     75,
                     0,
                     75,
                     75,
                     0,
                     75);

  pixelTomography_far =
      ibooker.book2D("tomography pixel", "tomography with pixel;x + 25 OOT (mm);y (mm)", 75, 0, 75, 8, 0, 8);

  hit_rate = ibooker.book1D("hit rate", title + "hit rate;rate (Hz)", 40, 0, 20);
}

//----------------------------------------------------------------------------------------------------

CTPPSDiamondDQMSource::CTPPSDiamondDQMSource(const edm::ParameterSet& ps)
    : tokenStatus_(consumes<edm::DetSetVector<TotemVFATStatus>>(ps.getParameter<edm::InputTag>("tagStatus"))),
      tokenPixelTrack_(
          consumes<edm::DetSetVector<CTPPSPixelLocalTrack>>(ps.getParameter<edm::InputTag>("tagPixelLocalTracks"))),
      tokenDigi_(consumes<edm::DetSetVector<CTPPSDiamondDigi>>(ps.getParameter<edm::InputTag>("tagDigi"))),
      tokenDiamondHit_(
          consumes<edm::DetSetVector<CTPPSDiamondRecHit>>(ps.getParameter<edm::InputTag>("tagDiamondRecHits"))),
      tokenDiamondTrack_(
          consumes<edm::DetSetVector<CTPPSDiamondLocalTrack>>(ps.getParameter<edm::InputTag>("tagDiamondLocalTracks"))),
      tokenFEDInfo_(consumes<std::vector<TotemFEDInfo>>(ps.getParameter<edm::InputTag>("tagFEDInfo"))),
      ctppsGeometryRunToken_(esConsumes<CTPPSGeometry, VeryForwardRealGeometryRecord, edm::Transition::BeginRun>()),
      ctppsGeometryEventToken_(esConsumes<CTPPSGeometry, VeryForwardRealGeometryRecord>()),
      excludeMultipleHits_(ps.getParameter<bool>("excludeMultipleHits")),
      centralOOT_(-999),
      verbosity_(ps.getUntrackedParameter<unsigned int>("verbosity", 0)),
      EC_difference_56_(-500),
      EC_difference_45_(-500) {
  for (const auto& pset : ps.getParameter<std::vector<edm::ParameterSet>>("offsetsOOT")) {
    runParameters_.emplace_back(
        std::make_pair(pset.getParameter<edm::EventRange>("validityRange"), pset.getParameter<int>("centralOOT")));
  }
}

//----------------------------------------------------------------------------------------------------

CTPPSDiamondDQMSource::~CTPPSDiamondDQMSource() {}

//----------------------------------------------------------------------------------------------------

void CTPPSDiamondDQMSource::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  centralOOT_ = -999;
  for (const auto& oot : runParameters_) {
    if (edm::contains(oot.first, edm::EventID(iRun.run(), 0, 1))) {
      centralOOT_ = oot.second;
      break;
    }
  }

  // Get detector shifts from the geometry
  const CTPPSGeometry& geom = iSetup.getData(ctppsGeometryRunToken_);
  const CTPPSDiamondDetId detid(0, CTPPS_DIAMOND_STATION_ID, CTPPS_DIAMOND_RP_ID, 0, 0);
  const DetGeomDesc* det = geom.sensor(detid);
  horizontalShiftOfDiamond_ = det->translation().x() - det->params().at(0);

  // Rough alignement of pixel detector for diamond thomography
  const CTPPSPixelDetId pixid(0, CTPPS_PIXEL_STATION_ID, CTPPS_FAR_RP_ID, 0);
  if (iRun.run() > 300000) {  //Pixel installed
    det = geom.sensor(pixid);
    horizontalShiftBwDiamondPixels_ = det->translation().x() - det->params().at(0) - horizontalShiftOfDiamond_ - 1;
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSDiamondDQMSource::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {
  ibooker.cd();
  ibooker.setCurrentFolder("CTPPS");

  globalPlot_ = GlobalPlots(ibooker);

  for (unsigned short arm = 0; arm < CTPPS_NUM_OF_ARMS; ++arm) {
    const CTPPSDiamondDetId rpId(arm, CTPPS_DIAMOND_STATION_ID, CTPPS_DIAMOND_RP_ID);
    potPlots_[rpId] = PotPlots(ibooker, rpId);
    for (unsigned short pl = 0; pl < CTPPS_DIAMOND_NUM_OF_PLANES; ++pl) {
      const CTPPSDiamondDetId plId(arm, CTPPS_DIAMOND_STATION_ID, CTPPS_DIAMOND_RP_ID, pl);
      planePlots_[plId] = PlanePlots(ibooker, plId);
      for (unsigned short ch = 0; ch < CTPPS_DIAMOND_NUM_OF_CHANNELS; ++ch) {
        const CTPPSDiamondDetId chId(arm, CTPPS_DIAMOND_STATION_ID, CTPPS_DIAMOND_RP_ID, pl, ch);
        channelPlots_[chId] = ChannelPlots(ibooker, chId);
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------

std::shared_ptr<dds::Cache> CTPPSDiamondDQMSource::globalBeginLuminosityBlock(const edm::LuminosityBlock&,
                                                                              const edm::EventSetup&) const {
  auto d = std::make_shared<dds::Cache>();
  d->hitDistribution2dMap.reserve(potPlots_.size());
  for (auto& plot : potPlots_)
    d->hitDistribution2dMap[plot.first] =
        std::unique_ptr<TH2F>(static_cast<TH2F*>(plot.second.hitDistribution2d_lumisection->getTH2F()->Clone()));
  return d;
}

//----------------------------------------------------------------------------------------------------

void CTPPSDiamondDQMSource::analyze(const edm::Event& event, const edm::EventSetup& iSetup) {
  // get event data
  edm::Handle<edm::DetSetVector<TotemVFATStatus>> diamondVFATStatus;
  event.getByToken(tokenStatus_, diamondVFATStatus);

  edm::Handle<edm::DetSetVector<CTPPSPixelLocalTrack>> pixelTracks;
  event.getByToken(tokenPixelTrack_, pixelTracks);

  edm::Handle<edm::DetSetVector<CTPPSDiamondDigi>> diamondDigis;
  event.getByToken(tokenDigi_, diamondDigis);

  edm::Handle<std::vector<TotemFEDInfo>> fedInfo;
  event.getByToken(tokenFEDInfo_, fedInfo);

  edm::Handle<edm::DetSetVector<CTPPSDiamondRecHit>> diamondRecHits;
  event.getByToken(tokenDiamondHit_, diamondRecHits);

  edm::Handle<edm::DetSetVector<CTPPSDiamondLocalTrack>> diamondLocalTracks;
  event.getByToken(tokenDiamondTrack_, diamondLocalTracks);

  const CTPPSGeometry* ctppsGeometry = &iSetup.getData(ctppsGeometryEventToken_);

  // check validity
  bool valid = true;
  valid &= diamondVFATStatus.isValid();
  valid &= pixelTracks.isValid();
  valid &= diamondDigis.isValid();
  valid &= fedInfo.isValid();
  valid &= diamondRecHits.isValid();
  valid &= diamondLocalTracks.isValid();

  if (!valid) {
    if (verbosity_) {
      edm::LogProblem("CTPPSDiamondDQMSource")
          << "ERROR in CTPPSDiamondDQMSource::analyze > some of the required inputs are not valid. Skipping this "
             "event.\n"
          << "    diamondVFATStatus.isValid = " << diamondVFATStatus.isValid() << "\n"
          << "    pixelTracks.isValid = " << pixelTracks.isValid() << "\n"
          << "    diamondDigis.isValid = " << diamondDigis.isValid() << "\n"
          << "    fedInfo.isValid = " << fedInfo.isValid() << "\n"
          << "    diamondRecHits.isValid = " << diamondRecHits.isValid() << "\n"
          << "    diamondLocalTracks.isValid = " << diamondLocalTracks.isValid();
    }

    return;
  }

  //------------------------------
  // RP Plots
  //------------------------------

  //------------------------------
  // Correlation Plots
  //------------------------------

  // Using CTPPSDiamondDigi
  for (const auto& digis : *diamondDigis) {
    const CTPPSDiamondDetId detId(digis.detId());
    CTPPSDiamondDetId detId_pot(digis.detId());

    for (const auto& digi : digis) {
      detId_pot.setPlane(0);
      detId_pot.setChannel(0);
      if (detId.channel() == CHANNEL_OF_VFAT_CLOCK)
        continue;
      if (potPlots_.find(detId_pot) == potPlots_.end())
        continue;
      //Leading without trailing investigation
      if (digi.leadingEdge() != 0 || digi.trailingEdge() != 0) {
        ++(potPlots_[detId_pot].HitCounter);
        if (digi.leadingEdge() != 0) {
          potPlots_[detId_pot].leadingEdgeCumulative_all->Fill(HPTDC_BIN_WIDTH_NS * digi.leadingEdge());
        }
        if (digi.leadingEdge() != 0 && digi.trailingEdge() == 0) {
          ++(potPlots_[detId_pot].LeadingOnlyCounter);
          potPlots_[detId_pot].leadingEdgeCumulative_le->Fill(HPTDC_BIN_WIDTH_NS * digi.leadingEdge());
        }
        if (digi.leadingEdge() == 0 && digi.trailingEdge() != 0) {
          ++(potPlots_[detId_pot].TrailingOnlyCounter);
          potPlots_[detId_pot].trailingEdgeCumulative_te->Fill(HPTDC_BIN_WIDTH_NS * digi.trailingEdge());
        }
        if (digi.leadingEdge() != 0 && digi.trailingEdge() != 0) {
          ++(potPlots_[detId_pot].CompleteCounter);
          potPlots_[detId_pot].leadingTrailingCorrelationPot->Fill(HPTDC_BIN_WIDTH_NS * digi.leadingEdge(),
                                                                   HPTDC_BIN_WIDTH_NS * digi.trailingEdge());
        }
      }

      // HPTDC Errors
      const HPTDCErrorFlags hptdcErrors = digi.hptdcErrorFlags();
      if (detId.channel() == 6 || detId.channel() == 7)  // ch6 for HPTDC 0 and ch7 for HPTDC 1
      {
        int verticalIndex = 2 * detId.plane() + (detId.channel() - 6);
        for (unsigned short hptdcErrorIndex = 1; hptdcErrorIndex < 16; ++hptdcErrorIndex)
          if (hptdcErrors.errorId(hptdcErrorIndex - 1))
            potPlots_[detId_pot].HPTDCErrorFlags_2D->Fill(hptdcErrorIndex, verticalIndex);
      }
      if (digi.multipleHit())
        ++(potPlots_[detId_pot].MHCounter);
    }
  }

  // EC Errors
  for (const auto& vfat_status : *diamondVFATStatus) {
    const CTPPSDiamondDetId detId(vfat_status.detId());
    CTPPSDiamondDetId detId_pot(vfat_status.detId());
    detId_pot.setPlane(0);
    detId_pot.setChannel(0);
    for (const auto& status : vfat_status) {
      if (!status.isOK())
        continue;
      if (potPlots_.find(detId_pot) == potPlots_.end())
        continue;
      if (channelPlots_.find(detId) == channelPlots_.end())
        continue;

      // Check Event Number
      for (const auto& optorx : *fedInfo) {
        if (detId.arm() == 1 && optorx.fedId() == CTPPS_FED_ID_56) {
          potPlots_[detId_pot].ECCheck->Fill((int)((optorx.lv1() & 0xFF) - ((unsigned int)status.ec() & 0xFF)) & 0xFF);
          if ((static_cast<int>((optorx.lv1() & 0xFF) - status.ec()) != EC_difference_56_) &&
              (static_cast<uint8_t>((optorx.lv1() & 0xFF) - status.ec()) < 128))
            EC_difference_56_ = static_cast<int>(optorx.lv1() & 0xFF) - (static_cast<unsigned int>(status.ec()) & 0xFF);
          if (EC_difference_56_ != 1 && EC_difference_56_ != -500 && std::abs(EC_difference_56_) < 127) {
            if (detId.channel() == 6 || detId.channel() == 7)
              potPlots_[detId_pot].HPTDCErrorFlags_2D->Fill(16, 2 * detId.plane() + (detId.channel() - 6));
            if (verbosity_)
              edm::LogProblem("CTPPSDiamondDQMSource")
                  << "FED " << CTPPS_FED_ID_56 << ": ECError at EV: 0x" << std::hex << optorx.lv1() << "\t\tVFAT EC: 0x"
                  << static_cast<unsigned int>(status.ec()) << "\twith ID: " << std::dec << detId
                  << "\tdiff: " << EC_difference_56_;
          }
        } else if (detId.arm() == 0 && optorx.fedId() == CTPPS_FED_ID_45) {
          potPlots_[detId_pot].ECCheck->Fill((int)((optorx.lv1() & 0xFF) - status.ec()) & 0xFF);
          if ((static_cast<int>((optorx.lv1() & 0xFF) - status.ec()) != EC_difference_45_) &&
              (static_cast<uint8_t>((optorx.lv1() & 0xFF) - status.ec()) < 128))
            EC_difference_45_ = static_cast<int>(optorx.lv1() & 0xFF) - (static_cast<unsigned int>(status.ec()) & 0xFF);
          if (EC_difference_45_ != 1 && EC_difference_45_ != -500 && std::abs(EC_difference_45_) < 127) {
            if (detId.channel() == 6 || detId.channel() == 7)
              potPlots_[detId_pot].HPTDCErrorFlags_2D->Fill(16, 2 * detId.plane() + (detId.channel() - 6));
            if (verbosity_)
              edm::LogProblem("CTPPSDiamondDQMSource")
                  << "FED " << CTPPS_FED_ID_45 << ": ECError at EV: 0x" << std::hex << optorx.lv1() << "\t\tVFAT EC: 0x"
                  << static_cast<unsigned int>(status.ec()) << "\twith ID: " << std::dec << detId
                  << "\tdiff: " << EC_difference_45_;
          }
        }
      }
    }
  }

  // Using CTPPSDiamondRecHit
  std::unordered_map<unsigned int, std::set<unsigned int>> planes;
  std::unordered_map<unsigned int, std::set<unsigned int>> planes_inclusive;

  auto lumiCache = luminosityBlockCache(event.getLuminosityBlock().index());
  for (const auto& rechits : *diamondRecHits) {
    CTPPSDiamondDetId detId_pot(rechits.detId());
    detId_pot.setPlane(0);
    detId_pot.setChannel(0);
    const CTPPSDiamondDetId detId(rechits.detId());

    for (const auto& rechit : rechits) {
      planes_inclusive[detId_pot].insert(detId.plane());
      if (excludeMultipleHits_ && rechit.multipleHits() > 0)
        continue;
      if (rechit.toT() != 0 && centralOOT_ != -999 && rechit.ootIndex() == centralOOT_)
        planes[detId_pot].insert(detId.plane());

      if (potPlots_.find(detId_pot) == potPlots_.end())
        continue;

      float UFSDShift = 0.0;
      if (rechit.yWidth() < 3)
        UFSDShift = 0.5;  // Display trick for UFSD that have 2 pixels with same X

      if (rechit.toT() != 0 && centralOOT_ != -999 && rechit.ootIndex() == centralOOT_) {
        TH2F* hitHistoTmp = potPlots_[detId_pot].hitDistribution2d->getTH2F();
        TAxis* hitHistoTmpYAxis = hitHistoTmp->GetYaxis();
        int startBin = hitHistoTmpYAxis->FindBin(rechit.x() - horizontalShiftOfDiamond_ - 0.5 * rechit.xWidth());
        int numOfBins = rechit.xWidth() * INV_DISPLAY_RESOLUTION_FOR_HITS_MM;
        for (int i = 0; i < numOfBins; ++i) {
          hitHistoTmp->Fill(detId.plane() + UFSDShift, hitHistoTmpYAxis->GetBinCenter(startBin + i));
        }

        hitHistoTmp = lumiCache->hitDistribution2dMap[detId_pot].get();
        hitHistoTmpYAxis = hitHistoTmp->GetYaxis();
        startBin = hitHistoTmpYAxis->FindBin(rechit.x() - horizontalShiftOfDiamond_ - 0.5 * rechit.xWidth());
        numOfBins = rechit.xWidth() * INV_DISPLAY_RESOLUTION_FOR_HITS_MM;
        for (int i = 0; i < numOfBins; ++i) {
          hitHistoTmp->Fill(detId.plane() + UFSDShift, hitHistoTmpYAxis->GetBinCenter(startBin + i));
        }
      }

      if (rechit.toT() != 0) {
        // Both
        potPlots_[detId_pot].leadingEdgeCumulative_both->Fill(rechit.time() + 25 * rechit.ootIndex());
        potPlots_[detId_pot].timeOverThresholdCumulativePot->Fill(rechit.toT());

        TH2F* hitHistoOOTTmp = potPlots_[detId_pot].hitDistribution2dOOT->getTH2F();
        TAxis* hitHistoOOTTmpYAxis = hitHistoOOTTmp->GetYaxis();
        int startBin = hitHistoOOTTmpYAxis->FindBin(rechit.x() - horizontalShiftOfDiamond_ - 0.5 * rechit.xWidth());
        int numOfBins = rechit.xWidth() * INV_DISPLAY_RESOLUTION_FOR_HITS_MM;
        for (int i = 0; i < numOfBins; ++i) {
          hitHistoOOTTmp->Fill(detId.plane() + 0.25 * rechit.ootIndex(),
                               hitHistoOOTTmpYAxis->GetBinCenter(startBin + i));
        }
      } else {
        if (rechit.ootIndex() != CTPPSDiamondRecHit::TIMESLICE_WITHOUT_LEADING) {
          // Only leading
          TH2F* hitHistoOOTTmp = potPlots_[detId_pot].hitDistribution2dOOT_le->getTH2F();
          TAxis* hitHistoOOTTmpYAxis = hitHistoOOTTmp->GetYaxis();
          int startBin = hitHistoOOTTmpYAxis->FindBin(rechit.x() - horizontalShiftOfDiamond_ - 0.5 * rechit.xWidth());
          int numOfBins = rechit.xWidth() * INV_DISPLAY_RESOLUTION_FOR_HITS_MM;
          for (int i = 0; i < numOfBins; ++i) {
            hitHistoOOTTmp->Fill(detId.plane() + 0.25 * rechit.ootIndex(),
                                 hitHistoOOTTmpYAxis->GetBinCenter(startBin + i));
          }
        }
      }
      if (rechit.ootIndex() != CTPPSDiamondRecHit::TIMESLICE_WITHOUT_LEADING &&
          potPlots_[detId_pot].activity_per_bx.count(rechit.ootIndex()) > 0)
        potPlots_[detId_pot].activity_per_bx.at(rechit.ootIndex())->Fill(event.bunchCrossing());
    }
  }

  for (const auto& plt : potPlots_) {
    plt.second.activePlanes->Fill(planes[plt.first].size());
    plt.second.activePlanesInclusive->Fill(planes_inclusive[plt.first].size());
  }

  // Using CTPPSDiamondLocalTrack
  for (const auto& tracks : *diamondLocalTracks) {
    CTPPSDiamondDetId detId_pot(tracks.detId());
    detId_pot.setPlane(0);
    detId_pot.setChannel(0);
    const CTPPSDiamondDetId detId(tracks.detId());

    for (const auto& track : tracks) {
      if (!track.isValid())
        continue;
      if (track.ootIndex() == CTPPSDiamondRecHit::TIMESLICE_WITHOUT_LEADING)
        continue;
      if (excludeMultipleHits_ && track.multipleHits() > 0)
        continue;
      if (potPlots_.find(detId_pot) == potPlots_.end())
        continue;

      TH2F* trackHistoOOTTmp = potPlots_[detId_pot].trackDistributionOOT->getTH2F();
      TAxis* trackHistoOOTTmpYAxis = trackHistoOOTTmp->GetYaxis();
      int startBin = trackHistoOOTTmpYAxis->FindBin(track.x0() - horizontalShiftOfDiamond_ - track.x0Sigma());
      int numOfBins = 2 * track.x0Sigma() * INV_DISPLAY_RESOLUTION_FOR_HITS_MM;
      for (int i = 0; i < numOfBins; ++i) {
        trackHistoOOTTmp->Fill(track.ootIndex(), trackHistoOOTTmpYAxis->GetBinCenter(startBin + i));
      }

      if (centralOOT_ != -999 && track.ootIndex() == centralOOT_) {
        TH1F* trackHistoInTimeTmp = potPlots_[detId_pot].trackDistribution->getTH1F();
        int startBin = trackHistoInTimeTmp->FindBin(track.x0() - horizontalShiftOfDiamond_ - track.x0Sigma());
        int numOfBins = 2 * track.x0Sigma() * INV_DISPLAY_RESOLUTION_FOR_HITS_MM;
        for (int i = 0; i < numOfBins; ++i) {
          trackHistoInTimeTmp->Fill(trackHistoInTimeTmp->GetBinCenter(startBin + i));
        }
      }
    }
  }

  // Channel efficiency using CTPPSDiamondLocalTrack
  for (const auto& tracks : *diamondLocalTracks) {
    CTPPSDiamondDetId detId_pot(tracks.detId());
    detId_pot.setPlane(0);
    detId_pot.setChannel(0);
    for (const auto& track : tracks) {
      // Find hits and planes in the track
      int numOfHits = 0;
      std::set<int> planesInTrackSet;
      for (const auto& vec : *diamondRecHits) {
        const CTPPSDiamondDetId detid(vec.detId());
        if (detid.arm() != detId_pot.arm())
          continue;

        for (const auto& hit : vec) {
          // first check if the hit contributes to the track
          if (track.containsHit(hit)) {
            ++numOfHits;
            planesInTrackSet.insert(detid.plane());
          }
        }
      }

      if (numOfHits > 0 && numOfHits <= 10 && planesInTrackSet.size() > 2) {
        for (int plane = 0; plane < 4; ++plane) {
          for (int channel = 0; channel < 12; ++channel) {
            int map_index = plane * 100 + channel;
            if (potPlots_[detId_pot].effDoublecountingChMap.find(map_index) ==
                potPlots_[detId_pot].effDoublecountingChMap.end()) {
              potPlots_[detId_pot].effTriplecountingChMap[map_index] = 0;
              potPlots_[detId_pot].effDoublecountingChMap[map_index] = 0;
            }
            CTPPSDiamondDetId detId(detId_pot.arm(), CTPPS_DIAMOND_STATION_ID, CTPPS_DIAMOND_RP_ID, plane, channel);
            if (channelAlignedWithTrack(ctppsGeometry, detId, track, 0.2)) {
              // Channel should fire
              ++(potPlots_[detId_pot].effDoublecountingChMap[map_index]);
              for (const auto& rechits : *diamondRecHits) {
                CTPPSDiamondDetId detId_hit(rechits.detId());
                if (detId_hit == detId) {
                  for (const auto& rechit : rechits) {
                    if (track.containsHit(rechit, 1)) {
                      // Channel fired
                      ++(potPlots_[detId_pot].effTriplecountingChMap[map_index]);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // Tomography of diamonds using pixel
  for (const auto& rechits : *diamondRecHits) {
    CTPPSDiamondDetId detId_pot(rechits.detId());
    detId_pot.setPlane(0);
    detId_pot.setChannel(0);
    const CTPPSDiamondDetId detId(rechits.detId());
    for (const auto& rechit : rechits) {
      if (excludeMultipleHits_ && rechit.multipleHits() > 0)
        continue;
      if (rechit.toT() == 0)
        continue;
      if (!pixelTracks.isValid())
        continue;
      if (potPlots_.find(detId_pot) == potPlots_.end())
        continue;

      for (const auto& ds : *pixelTracks) {
        if (ds.size() > 1)
          continue;
        const CTPPSPixelDetId pixId(ds.detId());
        if (pixId.station() != CTPPS_PIXEL_STATION_ID || pixId.rp() != CTPPS_FAR_RP_ID)
          continue;
        for (const auto& lt : ds) {
          if (lt.isValid() && pixId.arm() == detId_pot.arm()) {
            if (rechit.ootIndex() != CTPPSDiamondRecHit::TIMESLICE_WITHOUT_LEADING && rechit.ootIndex() >= 0 &&
                potPlots_[detId_pot].pixelTomographyAll.count(rechit.ootIndex()) > 0 &&
                lt.x0() - horizontalShiftBwDiamondPixels_ < 24)
              potPlots_[detId_pot]
                  .pixelTomographyAll.at(rechit.ootIndex())
                  ->Fill(lt.x0() - horizontalShiftBwDiamondPixels_ + 25 * detId.plane(), lt.y0());
          }
        }
      }
    }
  }

  //------------------------------
  // Clock Plots
  //------------------------------
  // Commented out to save space in the DQM files, but code should be kept
  // for ( const auto& digis : *diamondDigis ) {
  //   const CTPPSDiamondDetId detId( digis.detId() );
  //   CTPPSDiamondDetId detId_pot( digis.detId() );
  //   if ( detId.channel() == CHANNEL_OF_VFAT_CLOCK ) {
  //     detId_pot.setPlane( 0 );
  //     detId_pot.setChannel( 0 );
  //     for ( const auto& digi : digis ) {
  //       if ( digi.leadingEdge() != 0 )  {
  //         if ( detId.plane() == 1 ) {
  //           potPlots_[detId_pot].clock_Digi1_le->Fill( HPTDC_BIN_WIDTH_NS * digi.leadingEdge() );
  //           potPlots_[detId_pot].clock_Digi1_te->Fill( HPTDC_BIN_WIDTH_NS * digi.trailingEdge() );
  //         }
  //         if ( detId.plane() == 3 ) {
  //           potPlots_[detId_pot].clock_Digi3_le->Fill( HPTDC_BIN_WIDTH_NS * digi.leadingEdge() );
  //           potPlots_[detId_pot].clock_Digi3_te->Fill( HPTDC_BIN_WIDTH_NS * digi.trailingEdge() );
  //         }
  //       }
  //     }
  //   }
  // }

  //------------------------------
  // Plane Plots
  //------------------------------

  // Using CTPPSDiamondDigi
  std::unordered_map<unsigned int, unsigned int> channelsPerPlane;
  for (const auto& digis : *diamondDigis) {
    const CTPPSDiamondDetId detId(digis.detId());
    CTPPSDiamondDetId detId_plane(digis.detId());
    for (const auto& digi : digis) {
      detId_plane.setChannel(0);
      if (detId.channel() == CHANNEL_OF_VFAT_CLOCK)
        continue;
      if (planePlots_.find(detId_plane) == planePlots_.end())
        continue;

      if (digi.leadingEdge() != 0) {
        planePlots_[detId_plane].digiProfileCumulativePerPlane->Fill(detId.channel());
        if (channelsPerPlane.find(detId_plane) != channelsPerPlane.end())
          channelsPerPlane[detId_plane]++;
        else
          channelsPerPlane[detId_plane] = 0;
      }
    }
  }

  for (const auto& plt : channelsPerPlane) {
    planePlots_[plt.first].hit_multiplicity->Fill(plt.second);
  }

  // Using CTPPSDiamondRecHit
  for (const auto& rechits : *diamondRecHits) {
    CTPPSDiamondDetId detId_plane(rechits.detId());
    detId_plane.setChannel(0);
    for (const auto& rechit : rechits) {
      if (excludeMultipleHits_ && rechit.multipleHits() > 0)
        continue;
      if (rechit.toT() == 0)
        continue;
      if (planePlots_.find(detId_plane) != planePlots_.end()) {
        if (centralOOT_ != -999 && rechit.ootIndex() == centralOOT_) {
          TH1F* hitHistoTmp = planePlots_[detId_plane].hitProfile->getTH1F();
          int startBin = hitHistoTmp->FindBin(rechit.x() - horizontalShiftOfDiamond_ - 0.5 * rechit.xWidth());
          int numOfBins = rechit.xWidth() * INV_DISPLAY_RESOLUTION_FOR_HITS_MM;
          for (int i = 0; i < numOfBins; ++i) {
            hitHistoTmp->Fill(hitHistoTmp->GetBinCenter(startBin + i));
          }
        }
      }
    }
  }

  //Tomography of diamonds using pixel and Efficiency WRT Pixels
  for (const auto& ds : *pixelTracks) {
    const CTPPSPixelDetId pixId(ds.detId());
    if (pixId.station() != CTPPS_PIXEL_STATION_ID || pixId.rp() != CTPPS_FAR_RP_ID)
      continue;
    if (ds.size() > 1)
      continue;
    for (const auto& lt : ds) {
      if (lt.isValid()) {
        // For efficieny
        CTPPSDiamondDetId detId_pot(pixId.arm(), CTPPS_DIAMOND_STATION_ID, CTPPS_DIAMOND_RP_ID);
        potPlots_[detId_pot].pixelTracksMap.Fill(lt.x0() - horizontalShiftBwDiamondPixels_, lt.y0());

        std::set<CTPPSDiamondDetId> planesWitHits_set;
        for (const auto& rechits : *diamondRecHits) {
          CTPPSDiamondDetId detId_plane(rechits.detId());
          detId_plane.setChannel(0);
          for (const auto& rechit : rechits) {
            if (excludeMultipleHits_ && rechit.multipleHits() > 0)
              continue;
            if (rechit.ootIndex() == CTPPSDiamondRecHit::TIMESLICE_WITHOUT_LEADING || rechit.toT() == 0)
              continue;
            if (planePlots_.find(detId_plane) == planePlots_.end())
              continue;
            if (pixId.arm() == detId_plane.arm() && lt.x0() - horizontalShiftBwDiamondPixels_ < 24) {
              planePlots_[detId_plane].pixelTomography_far->Fill(
                  lt.x0() - horizontalShiftBwDiamondPixels_ + 25 * rechit.ootIndex(), lt.y0());
              if (centralOOT_ != -999 && rechit.ootIndex() == centralOOT_)
                planesWitHits_set.insert(detId_plane);
            }
          }
        }

        for (auto& planeId : planesWitHits_set)
          planePlots_[planeId].pixelTracksMapWithDiamonds.Fill(lt.x0() - horizontalShiftBwDiamondPixels_, lt.y0());
      }
    }
  }

  //------------------------------
  // Channel Plots
  //------------------------------

  // digi profile cumulative
  for (const auto& digis : *diamondDigis) {
    const CTPPSDiamondDetId detId(digis.detId());
    for (const auto& digi : digis) {
      if (detId.channel() == CHANNEL_OF_VFAT_CLOCK)
        continue;
      if (channelPlots_.find(detId) != channelPlots_.end()) {
        // HPTDC Errors
        const HPTDCErrorFlags hptdcErrors = digi.hptdcErrorFlags();
        for (unsigned short hptdcErrorIndex = 1; hptdcErrorIndex < 16; ++hptdcErrorIndex)
          if (hptdcErrors.errorId(hptdcErrorIndex - 1))
            channelPlots_[detId].HPTDCErrorFlags->Fill(hptdcErrorIndex);
        if (digi.multipleHit())
          ++(channelPlots_[detId].MHCounter);

        // Check dropped trailing edges
        if (digi.leadingEdge() != 0 || digi.trailingEdge() != 0) {
          ++(channelPlots_[detId].HitCounter);
          if (digi.leadingEdge() != 0 && digi.trailingEdge() == 0) {
            ++(channelPlots_[detId].LeadingOnlyCounter);
            channelPlots_[detId].leadingEdgeCumulative_le->Fill(HPTDC_BIN_WIDTH_NS * digi.leadingEdge());
          }
          if (digi.leadingEdge() == 0 && digi.trailingEdge() != 0) {
            ++(channelPlots_[detId].TrailingOnlyCounter);
            channelPlots_[detId].trailingEdgeCumulative_te->Fill(HPTDC_BIN_WIDTH_NS * digi.trailingEdge());
          }
          if (digi.leadingEdge() != 0 && digi.trailingEdge() != 0) {
            ++(channelPlots_[detId].CompleteCounter);
            channelPlots_[detId].LeadingTrailingCorrelationPerChannel->Fill(HPTDC_BIN_WIDTH_NS * digi.leadingEdge(),
                                                                            HPTDC_BIN_WIDTH_NS * digi.trailingEdge());
          }
        }
      }
    }
  }

  // Using CTPPSDiamondRecHit

  for (const auto& rechits : *diamondRecHits) {
    CTPPSDiamondDetId detId(rechits.detId());
    for (const auto& rechit : rechits) {
      if (excludeMultipleHits_ && rechit.multipleHits() > 0)
        continue;
      if (channelPlots_.find(detId) != channelPlots_.end()) {
        if (rechit.ootIndex() != CTPPSDiamondRecHit::TIMESLICE_WITHOUT_LEADING && rechit.toT() != 0) {
          channelPlots_[detId].leadingEdgeCumulative_both->Fill(rechit.time() + 25 * rechit.ootIndex());
          channelPlots_[detId].TimeOverThresholdCumulativePerChannel->Fill(rechit.toT());
        }
        ++(lumiCache->hitsCounterMap[detId]);
      }

      if (rechit.ootIndex() != CTPPSDiamondRecHit::TIMESLICE_WITHOUT_LEADING &&
          channelPlots_[detId].activity_per_bx.count(rechit.ootIndex()) > 0)
        channelPlots_[detId].activity_per_bx.at(rechit.ootIndex())->Fill(event.bunchCrossing());
    }
  }

  // Tomography of diamonds using pixel
  for (const auto& rechits : *diamondRecHits) {
    const CTPPSDiamondDetId detId(rechits.detId());
    for (const auto& rechit : rechits) {
      if (excludeMultipleHits_ && rechit.multipleHits() > 0)
        continue;
      if (rechit.ootIndex() == CTPPSDiamondRecHit::TIMESLICE_WITHOUT_LEADING || rechit.toT() == 0)
        continue;
      if (!pixelTracks.isValid())
        continue;
      if (channelPlots_.find(detId) == channelPlots_.end())
        continue;

      for (const auto& ds : *pixelTracks) {
        const CTPPSPixelDetId pixId(ds.detId());
        if (pixId.station() != CTPPS_PIXEL_STATION_ID || pixId.rp() != CTPPS_FAR_RP_ID)
          continue;
        if (ds.size() > 1)
          continue;
        for (const auto& lt : ds) {
          if (lt.isValid() && pixId.arm() == detId.arm() && lt.x0() - horizontalShiftBwDiamondPixels_ < 24)
            channelPlots_[detId].pixelTomography_far->Fill(
                lt.x0() - horizontalShiftBwDiamondPixels_ + 25 * rechit.ootIndex(), lt.y0());
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSDiamondDQMSource::globalEndLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup&) {
  auto lumiCache = luminosityBlockCache(iLumi.index());
  for (auto& plot : potPlots_) {
    *(plot.second.hitDistribution2d_lumisection->getTH2F()) = *(lumiCache->hitDistribution2dMap[plot.first]);
  }
  for (auto& plot : channelPlots_) {
    auto hitsCounterPerLumisection = lumiCache->hitsCounterMap[plot.first];
    if (hitsCounterPerLumisection != 0) {
      plot.second.hit_rate->Fill((double)hitsCounterPerLumisection / SEC_PER_LUMI_SECTION);
    }

    double HundredOverHitCounter = .0;
    if (plot.second.HitCounter != 0)
      HundredOverHitCounter = 100. / plot.second.HitCounter;
    plot.second.HPTDCErrorFlags->setBinContent(16, HundredOverHitCounter * plot.second.MHCounter);
    plot.second.leadingWithoutTrailing->setBinContent(1, HundredOverHitCounter * plot.second.LeadingOnlyCounter);
    plot.second.leadingWithoutTrailing->setBinContent(2, HundredOverHitCounter * plot.second.TrailingOnlyCounter);
    plot.second.leadingWithoutTrailing->setBinContent(3, HundredOverHitCounter * plot.second.CompleteCounter);
  }

  for (auto& plot : potPlots_) {
    double HundredOverHitCounterPot = 0.;
    if (plot.second.HitCounter != 0)
      HundredOverHitCounterPot = 100. / plot.second.HitCounter;
    plot.second.leadingWithoutTrailingCumulativePot->setBinContent(
        1, HundredOverHitCounterPot * plot.second.LeadingOnlyCounter);
    plot.second.leadingWithoutTrailingCumulativePot->setBinContent(
        2, HundredOverHitCounterPot * plot.second.TrailingOnlyCounter);
    plot.second.leadingWithoutTrailingCumulativePot->setBinContent(
        3, HundredOverHitCounterPot * plot.second.CompleteCounter);

    plot.second.MHComprensive->Reset();
    CTPPSDiamondDetId rpId(plot.first);
    for (auto& chPlot : channelPlots_) {
      CTPPSDiamondDetId chId(chPlot.first);
      if (chId.arm() == rpId.arm() && chId.rp() == rpId.rp()) {
        plot.second.MHComprensive->Fill(chId.plane(), chId.channel(), chPlot.second.HPTDCErrorFlags->getBinContent(16));
      }
    }
  }

  // Efficiencies of single channels
  for (auto& plot : potPlots_) {
    plot.second.EfficiencyOfChannelsInPot->Reset();
    for (auto& element : plot.second.effTriplecountingChMap) {
      if (plot.second.effDoublecountingChMap[element.first] > 0) {
        int plane = element.first / 100;
        int channel = element.first % 100;
        double counted = element.second;
        double total = plot.second.effDoublecountingChMap[element.first];
        double efficiency = counted / total;
        //         double error = std::sqrt( efficiency * ( 1 - efficiency ) / total );

        plot.second.EfficiencyOfChannelsInPot->Fill(plane, channel, 100 * efficiency);
      }
    }
  }

  // Efficeincy wrt pixels  //TODO
  for (auto& plot : planePlots_) {
    TH2F* hitHistoTmp = plot.second.EfficiencyWRTPixelsInPlane->getTH2F();

    CTPPSDiamondDetId detId_pot(plot.first);
    detId_pot.setPlane(0);

    hitHistoTmp->Divide(&(plot.second.pixelTracksMapWithDiamonds), &(potPlots_[detId_pot].pixelTracksMap));
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSDiamondDQMSource);
