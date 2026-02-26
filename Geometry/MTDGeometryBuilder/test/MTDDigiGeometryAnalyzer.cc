// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetType.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetUnit.h"
#include "DataFormats/GeometrySurface/interface/MediumProperties.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/Rounding.h"

#include <fstream>

#include "CLHEP/Random/RandFlat.h"

using namespace cms_rounding;

// class declaration

class MTDDigiGeometryAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit MTDDigiGeometryAnalyzer(const edm::ParameterSet&);
  ~MTDDigiGeometryAnalyzer() override = default;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  inline std::string fround(const double in, const size_t prec) const {
    std::stringstream ss;
    ss << std::setprecision(prec) << std::fixed << std::setw(14) << roundIfNear0(in);
    return ss.str();
  }

  inline std::string fvecround(const auto& vecin, const size_t prec) const {
    std::stringstream ss;
    ss << std::setprecision(prec) << std::fixed << std::setw(14) << roundVecIfNear0(vecin);
    return ss.str();
  }

  void checkRectangularMTDTopology(const RectangularMTDTopology&);
  void checkPixelsAcceptance(const GeomDetUnit& det);
  void CheckETLstructure(const MTDGeometry&);

  std::stringstream sunitt_;

  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;

  // Constants to define the bins for Eta
  static constexpr int n_bin_Eta = 3;
  static constexpr double eta_bins_edges_neg[n_bin_Eta + 1] = {-3.0, -2.5, -2.1, -1.5};
  static constexpr double eta_bins_edges_pos[n_bin_Eta + 1] = {1.5, 2.1, 2.5, 3.0};

  // LGAD counter per Disc, DiscSide, and Sector: [disk][discSide][sector]
  static constexpr int n_discSide = 2;
  static constexpr int n_sector = 3;  // Use size 3 to allow 1-based indexing (1 to 2)
  uint32_t LGADsPerDiscSideSector_[4][n_discSide][n_sector] = {};

  // Counter for total LGADs per disk per eta bin: [disk][eta_bin]
  uint32_t LGADsPerDiskperEtaBin_[4][n_bin_Eta] = {{0}};
};

MTDDigiGeometryAnalyzer::MTDDigiGeometryAnalyzer(const edm::ParameterSet& iConfig) {
  mtdgeoToken_ = esConsumes<MTDGeometry, MTDDigiGeometryRecord>();
}

// ------------ method called to produce the data  ------------
void MTDDigiGeometryAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //
  // get the MTDGeometry
  //
  auto pDD = iSetup.getTransientHandle(mtdgeoToken_);
  edm::LogVerbatim("MTDDigiGeometryAnalyzer")
      << "MTDGeometry:\n"
      << " # detectors  = " << pDD->detUnits().size() << "\n"
      << " # types      = " << pDD->detTypes().size() << "\n"
      << " # BTL dets   = " << pDD->detsBTL().size() << "\n"
      << " # ETL dets   = " << pDD->detsETL().size() << "\n"
      << " # layers " << pDD->geomDetSubDetector(1) << "   = " << pDD->numberOfLayers(1) << "\n"
      << " # layers " << pDD->geomDetSubDetector(2) << "   = " << pDD->numberOfLayers(2) << "\n"
      << " # dets       = " << pDD->dets().size() << "\n"
      << " # detUnitIds = " << pDD->detUnitIds().size() << "\n"
      << " # detIds     = " << pDD->detIds().size() << "\n";

  sunitt_ << "MTDGeometry:\n"
          << " # detectors  = " << pDD->detUnits().size() << "\n"
          << " # types      = " << pDD->detTypes().size() << "\n"
          << " # BTL dets   = " << pDD->detsBTL().size() << "\n"
          << " # ETL dets   = " << pDD->detsETL().size() << "\n"
          << " # layers " << pDD->geomDetSubDetector(1) << "   = " << pDD->numberOfLayers(1) << "\n"
          << " # layers " << pDD->geomDetSubDetector(2) << "   = " << pDD->numberOfLayers(2) << "\n"
          << " # dets       = " << pDD->dets().size() << "\n"
          << " # detUnitIds = " << pDD->detUnitIds().size() << "\n"
          << " # detIds     = " << pDD->detIds().size() << "\n";

  for (auto const& it : pDD->detTypes()) {
    if (dynamic_cast<const MTDGeomDetType*>((it)) != nullptr) {
      const PixelTopology& p = (dynamic_cast<const MTDGeomDetType*>((it)))->specificTopology();
      const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(p);
      auto pitchval = topo.pitch();
      edm::LogVerbatim("MTDDigiGeometryAnalyzer")
          << "\n Subdetector " << it->subDetector() << " MTD Det " << it->name() << "\n"
          << " Rows     " << topo.nrows() << " Columns " << topo.ncolumns() << " ROCS X   " << topo.rocsX()
          << " ROCS Y  " << topo.rocsY() << " Rows/ROC " << topo.rowsperroc() << " Cols/ROC " << topo.colsperroc()
          << " Pitch X " << fround(pitchval.first, 4) << " Pitch Y " << fround(pitchval.second, 4)
          << " Sensor Interpad X " << fround(topo.gapxInterpad(), 4) << " Sensor Interpad Y "
          << fround(topo.gapyInterpad(), 4) << " Sensor Border X " << fround(topo.gapxBorder(), 4)
          << " Sensor Border Y " << fround(topo.gapyBorder(), 4) << "\n";
      sunitt_ << "\n Subdetector " << it->subDetector() << " MTD Det " << it->name() << "\n"
              << " Rows     " << topo.nrows() << " Columns " << topo.ncolumns() << " ROCS X   " << topo.rocsX()
              << " ROCS Y  " << topo.rocsY() << " Rows/ROC " << topo.rowsperroc() << " Cols/ROC " << topo.colsperroc()
              << " Pitch X " << fround(pitchval.first, 2) << " Pitch Y " << fround(pitchval.second, 2)
              << " Sensor Interpad X " << fround(topo.gapxInterpad(), 2) << " Sensor Interpad Y "
              << fround(topo.gapyInterpad(), 2) << " Sensor Border X " << fround(topo.gapxBorder(), 2)
              << " Sensor Border Y " << fround(topo.gapyBorder(), 2) << "\n";
      checkRectangularMTDTopology(topo);
    }
  }

  edm::LogVerbatim("MTDDigiGeometryAnalyzer") << "\nAcceptance of BTL module:";
  sunitt_ << "\nAcceptance of BTL module:";
  auto const& btldet = *(dynamic_cast<const MTDGeomDetUnit*>(pDD->detsBTL().front()));
  checkPixelsAcceptance(btldet);
  edm::LogVerbatim("MTDDigiGeometryAnalyzer") << "\nAcceptance of ETL module:";
  sunitt_ << "\nAcceptance of ETL module:";
  auto const& etldet = *(dynamic_cast<const MTDGeomDetUnit*>(pDD->detsETL().front()));
  checkPixelsAcceptance(etldet);

  // ETL structure prints
  edm::LogVerbatim("MTDDigiGeometryAnalyzer") << "\n";
  CheckETLstructure(*pDD);

  edm::LogVerbatim("MTDUnitTest") << sunitt_.str();
}

void MTDDigiGeometryAnalyzer::checkRectangularMTDTopology(const RectangularMTDTopology& topo) {
  edm::LogVerbatim("MTDDigiGeometryAnalyzer") << "Pixel center location:\n";
  sunitt_ << "Pixel center location:\n";
  LocalPoint center(0, 0, 0);
  for (int r = 0; r < topo.nrows(); r++) {
    for (int c = 0; c < topo.ncolumns(); c++) {
      edm::LogVerbatim("MTDDigiGeometryAnalyzer") << std::setw(7) << r << std::setw(7) << c << " "
                                                  << fvecround(topo.pixelToModuleLocalPoint(center, r, c), 4) << "\n";
      sunitt_ << std::setw(7) << r << std::setw(7) << c << " "
              << fvecround(topo.pixelToModuleLocalPoint(center, r, c), 2) << "\n";
    }
  }
}

void MTDDigiGeometryAnalyzer::checkPixelsAcceptance(const GeomDetUnit& det) {
  const Bounds& bounds = det.surface().bounds();
  const RectangularPlaneBounds* tb = dynamic_cast<const RectangularPlaneBounds*>(&bounds);
  if (tb == nullptr)
    return;  // not trapezoidal

  double length = tb->length();
  double width = tb->width();
  edm::LogVerbatim("MTDDigiGeometryAnalyzer")
      << " X (width) = " << fround(width, 4) << " Y (length) = " << fround(length, 4);
  sunitt_ << " X (width) = " << fround(width, 2) << " Y (length) = " << fround(length, 2);

  const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(det.topology());
  const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

  const size_t maxindex = 100000;
  size_t inpixel(0);
  for (size_t index = 0; index < maxindex; index++) {
    double ax = CLHEP::RandFlat::shoot(-width * 0.5, width * 0.5);
    double ay = CLHEP::RandFlat::shoot(-length * 0.5, length * 0.5);
    LocalPoint hit(ax, ay, 0);
    auto const indici = topo.pixelIndex(hit);
    assert(indici.first < topo.nrows() && indici.second < topo.ncolumns());  // sanity check on the index definition
    if (topo.isInPixel(hit)) {
      inpixel++;
    }
  }
  double acc = (double)inpixel / (double)maxindex;
  double accerr = std::sqrt(acc * (1. - acc) / (double)maxindex);
  edm::LogVerbatim("MTDDigiGeometryAnalyzer") << " Acceptance: " << fround(acc, 3) << " +/- " << fround(accerr, 3);
  sunitt_ << " Acceptance: " << fround(acc, 3) << " +/- " << fround(accerr, 3);
}

void MTDDigiGeometryAnalyzer::CheckETLstructure(const MTDGeometry& geom) {
  edm::LogVerbatim("MTDDigiGeometryAnalyzer") << "\n--- ETL Structure Validation ---";
  sunitt_ << "\n--- ETL Structure Validation ---";

  // Reset counters
  for (int d = 0; d < 4; ++d) {
    for (int eta = 0; eta < n_bin_Eta; ++eta) {
      LGADsPerDiskperEtaBin_[d][eta] = 0;
    }
    for (int k = 0; k < n_discSide; ++k) {
      for (int l = 0; l < n_sector; ++l) {
        LGADsPerDiscSideSector_[d][k][l] = 0;
      }
    }
  }

  uint32_t totalETLdets = 0;
  for (const auto& det : geom.detsETL()) {
    const GeomDet* thedet = det;
    ETLDetId detId(thedet->geographicalId());

    // Get the global position of the detector center
    const GlobalPoint& global_point = thedet->position();
    double eta = global_point.eta();

    int discSide = detId.discSide();  // 0 to 1
    int sector = detId.sector();      // 1 to 2

    int idet = 999;
    if ((detId.zside() == -1) && (detId.nDisc() == 1)) {
      idet = 0;
    } else if ((detId.zside() == -1) && (detId.nDisc() == 2)) {
      idet = 1;
    } else if ((detId.zside() == 1) && (detId.nDisc() == 1)) {
      idet = 2;
    } else if ((detId.zside() == 1) && (detId.nDisc() == 2)) {
      idet = 3;
    } else {
      edm::LogWarning("EtlDigiHitsValidation") << "Unknown ETL DetId configuration: " << detId;
      continue;
    }

    totalETLdets++;

    // Count LGADs per Disc, Side, Sector
    LGADsPerDiscSideSector_[idet][discSide][sector]++;

    // Count LGADs per disk per eta bin
    const double* eta_edges = (idet < 2) ? eta_bins_edges_neg : eta_bins_edges_pos;

    for (int j = 0; j < n_bin_Eta; j++) {
      double lower_edge = eta_edges[j];
      double upper_edge = eta_edges[j + 1];

      // Check if the center of the LGAD is within the bin
      if ((eta >= lower_edge && eta < upper_edge) || (idet < 2 && j == n_bin_Eta - 1 && eta <= upper_edge)) {
        LGADsPerDiskperEtaBin_[idet][j]++;
        break;  // Found the bin
      }
    }
  }

  // --- Print Summary ---

  edm::LogVerbatim("MTDDigiGeometryAnalyzer") << " Total ETL Detectors (LGADs): " << totalETLdets << "\n";
  sunitt_ << " Total ETL Detectors (LGADs): " << totalETLdets << "\n";
  const char* diskNames[4] = {"Disc 1 (-Z)", "Disc 2 (-Z)", "Disc 1 (+Z)", "Disc 2 (+Z)"};

  edm::LogVerbatim("MTDDigiGeometryAnalyzer") << "\n--- LGADs per Eta Bin and per Disk, DiscSide, Sector ---\n";
  sunitt_ << "\n--- LGADs per Eta Bin and per Disk, DiscSide, Sector ---\n";
  for (int d = 0; d < 4; ++d) {  // Physical Disk loop (0-3)
    std::string disk_name = diskNames[d];
    uint32_t total_disk = 0;
    for (int k = 0; k < n_discSide; ++k) {
      for (int l = 1; l < n_sector; ++l) {
        total_disk += LGADsPerDiscSideSector_[d][k][l];
      }
    }
    edm::LogVerbatim("MTDDigiGeometryAnalyzer") << "Region: " << disk_name << " | Total LGADs: " << total_disk << "\n";
    edm::LogVerbatim("MTDDigiGeometryAnalyzer") << "  - LGADs per Eta Bin:\n";
    sunitt_ << "Region: " << disk_name << " | Total LGADs: " << total_disk << "\n";
    sunitt_ << "  - LGADs per Eta Bin:\n";
    const double* eta_edges = (d < 2) ? eta_bins_edges_neg : eta_bins_edges_pos;
    for (int j = 0; j < n_bin_Eta; ++j) {
      edm::LogVerbatim("MTDDigiGeometryAnalyzer")
          << "    Eta [" << std::setprecision(1) << std::fixed << eta_edges[j] << ", " << eta_edges[j + 1]
          << "): " << LGADsPerDiskperEtaBin_[d][j] << "\n";
      sunitt_ << "    Eta [" << std::setprecision(1) << std::fixed << eta_edges[j] << ", " << eta_edges[j + 1]
              << "): " << LGADsPerDiskperEtaBin_[d][j] << "\n";
    }
    for (int k = 0; k < n_discSide; ++k) {
      uint32_t total_discside = 0;
      for (int l = 1; l < n_sector; ++l) {
        total_discside += LGADsPerDiscSideSector_[d][k][l];
      }
      edm::LogVerbatim("MTDDigiGeometryAnalyzer") << "  - Side: " << k << " | Total LGADs: " << total_discside << "\n";
      edm::LogVerbatim("MTDDigiGeometryAnalyzer") << "    - Sectors: ";
      sunitt_ << "  - Side: " << k << " | Total LGADs: " << total_discside << "\n";
      sunitt_ << "    - Sectors: ";
      for (int l = 1; l < n_sector; ++l) {
        if (LGADsPerDiscSideSector_[d][k][l] > 0) {
          edm::LogVerbatim("MTDDigiGeometryAnalyzer")
              << "Sec " << l << ": " << LGADsPerDiscSideSector_[d][k][l] << " | ";
          sunitt_ << "Sec " << l << ": " << LGADsPerDiscSideSector_[d][k][l] << " | ";
        }
      }
      edm::LogVerbatim("MTDDigiGeometryAnalyzer") << "\n";
      sunitt_ << "\n";
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(MTDDigiGeometryAnalyzer);
