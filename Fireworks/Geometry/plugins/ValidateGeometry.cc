// -*- C++ -*-
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"

#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include <string>
#include <iostream>

#include <TEveGeoNode.h>
#include <TGeoVolume.h>
#include <TGeoBBox.h>
#include <TGeoArb8.h>
#include <TFile.h>
#include <TH1.h>

class ValidateGeometry : public edm::EDAnalyzer {
public:
  explicit ValidateGeometry(const edm::ParameterSet&);
  ~ValidateGeometry() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void validateRPCGeometry(const int regionNumber, const char* regionName);

  void validateDTChamberGeometry();
  void validateDTLayerGeometry();

  void validateCSChamberGeometry(const int endcap, const char* detname);

  void validateCSCLayerGeometry(const int endcap, const char* detname);

  void validateCaloGeometry(DetId::Detector detector, int subdetector, const char* detname);

  void validateTrackerGeometry(const TrackerGeometry::DetContainer& dets, const char* detname);

  void validatePixelTopology(const TrackerGeometry::DetContainer& dets, const char* detname);

  void validateStripTopology(const TrackerGeometry::DetContainer& dets, const char* detname);

  void compareTransform(const GlobalPoint& point, const TGeoMatrix* matrix);

  void compareShape(const GeomDet* det, const float* shape);

  double getDistance(const GlobalPoint& point1, const GlobalPoint& point2);

  void makeHistograms(const char* detector);
  void makeHistogram(const std::string& name, std::vector<double>& data);

  std::string infileName_;
  std::string outfileName_;

  edm::ESHandle<RPCGeometry> rpcGeometry_;
  edm::ESHandle<DTGeometry> dtGeometry_;
  edm::ESHandle<CSCGeometry> cscGeometry_;
  edm::ESHandle<CaloGeometry> caloGeometry_;
  edm::ESHandle<TrackerGeometry> trackerGeometry_;

  FWGeometry fwGeometry_;

  TFile* outFile_;

  std::vector<double> globalDistances_;
  std::vector<double> topWidths_;
  std::vector<double> bottomWidths_;
  std::vector<double> lengths_;
  std::vector<double> thicknesses_;

  void clearData() {
    globalDistances_.clear();
    topWidths_.clear();
    bottomWidths_.clear();
    lengths_.clear();
    thicknesses_.clear();
  }

  bool dataEmpty() {
    return (globalDistances_.empty() && topWidths_.empty() && bottomWidths_.empty() && lengths_.empty() &&
            thicknesses_.empty());
  }

  bool doTracker_;
  bool doMuon_;
  bool doCalo_;
};

ValidateGeometry::ValidateGeometry(const edm::ParameterSet& iConfig)
    : infileName_(iConfig.getUntrackedParameter<std::string>("infileName")),
      outfileName_(iConfig.getUntrackedParameter<std::string>("outfileName")) {
  doTracker_ = iConfig.getUntrackedParameter<bool>("Tracker", true);
  doMuon_ = iConfig.getUntrackedParameter<bool>("Muon", true);
  doCalo_ = iConfig.getUntrackedParameter<bool>("Calo", true);

  fwGeometry_.loadMap(infileName_.c_str());

  outFile_ = new TFile(outfileName_.c_str(), "RECREATE");
}

ValidateGeometry::~ValidateGeometry() {}

void ValidateGeometry::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  if (doMuon_) {
    eventSetup.get<MuonGeometryRecord>().get(rpcGeometry_);

    if (rpcGeometry_.isValid()) {
      std::cout << "Validating RPC -z endcap geometry" << std::endl;
      validateRPCGeometry(-1, "RPC -z endcap");

      std::cout << "Validating RPC +z endcap geometry" << std::endl;
      validateRPCGeometry(+1, "RPC +z endcap");

      std::cout << "Validating RPC barrel geometry" << std::endl;
      validateRPCGeometry(0, "RPC barrel");
    } else
      fwLog(fwlog::kWarning) << "Invalid RPC geometry" << std::endl;

    eventSetup.get<MuonGeometryRecord>().get(dtGeometry_);

    if (dtGeometry_.isValid()) {
      std::cout << "Validating DT chamber geometry" << std::endl;
      validateDTChamberGeometry();

      std::cout << "Validating DT layer geometry" << std::endl;
      validateDTLayerGeometry();
    } else
      fwLog(fwlog::kWarning) << "Invalid DT geometry" << std::endl;

    eventSetup.get<MuonGeometryRecord>().get(cscGeometry_);

    if (cscGeometry_.isValid()) {
      std::cout << "Validating CSC -z geometry" << std::endl;
      validateCSChamberGeometry(-1, "CSC chamber -z endcap");

      std::cout << "Validating CSC +z geometry" << std::endl;
      validateCSChamberGeometry(+1, "CSC chamber +z endcap");

      std::cout << "Validating CSC layer -z geometry" << std::endl;
      validateCSCLayerGeometry(-1, "CSC layer -z endcap");

      std::cout << "Validating CSC layer +z geometry" << std::endl;
      validateCSCLayerGeometry(+1, "CSC layer +z endcap");
    } else
      fwLog(fwlog::kWarning) << "Invalid CSC geometry" << std::endl;
  }

  if (doTracker_) {
    eventSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometry_);

    if (trackerGeometry_.isValid()) {
      std::cout << "Validating TIB geometry and topology" << std::endl;
      validateTrackerGeometry(trackerGeometry_->detsTIB(), "TIB");
      validateStripTopology(trackerGeometry_->detsTIB(), "TIB");

      std::cout << "Validating TOB geometry and topology" << std::endl;
      validateTrackerGeometry(trackerGeometry_->detsTOB(), "TOB");
      validateStripTopology(trackerGeometry_->detsTOB(), "TOB");

      std::cout << "Validating TEC geometry and topology" << std::endl;
      validateTrackerGeometry(trackerGeometry_->detsTEC(), "TEC");
      validateStripTopology(trackerGeometry_->detsTEC(), "TEC");

      std::cout << "Validating TID geometry and topology" << std::endl;
      validateTrackerGeometry(trackerGeometry_->detsTID(), "TID");
      validateStripTopology(trackerGeometry_->detsTID(), "TID");

      std::cout << "Validating PXB geometry and topology" << std::endl;
      validateTrackerGeometry(trackerGeometry_->detsPXB(), "PXB");
      validatePixelTopology(trackerGeometry_->detsPXB(), "PXB");

      std::cout << "Validating PXF geometry and topology" << std::endl;
      validateTrackerGeometry(trackerGeometry_->detsPXF(), "PXF");
      validatePixelTopology(trackerGeometry_->detsPXF(), "PXF");
    } else
      fwLog(fwlog::kWarning) << "Invalid Tracker geometry" << std::endl;
  }

  if (doCalo_) {
    eventSetup.get<CaloGeometryRecord>().get(caloGeometry_);

    if (caloGeometry_.isValid()) {
      std::cout << "Validating EB geometry" << std::endl;
      validateCaloGeometry(DetId::Ecal, EcalBarrel, "EB");

      std::cout << "Validating EE geometry" << std::endl;
      validateCaloGeometry(DetId::Ecal, EcalEndcap, "EE");

      std::cout << "Validating ES geometry" << std::endl;
      validateCaloGeometry(DetId::Ecal, EcalPreshower, "ES");

      std::cout << "Validating HB geometry" << std::endl;
      validateCaloGeometry(DetId::Hcal, HcalBarrel, "HB");

      std::cout << "Validating HE geometry" << std::endl;
      validateCaloGeometry(DetId::Hcal, HcalEndcap, "HE");

      std::cout << "Validating HO geometry" << std::endl;
      validateCaloGeometry(DetId::Hcal, HcalOuter, "HO");

      std::cout << "Validating HF geometry" << std::endl;
      validateCaloGeometry(DetId::Hcal, HcalForward, "HF");

      std::cout << "Validating Castor geometry" << std::endl;
      validateCaloGeometry(DetId::Calo, HcalCastorDetId::SubdetectorId, "Castor");

      std::cout << "Validating ZDC geometry" << std::endl;
      validateCaloGeometry(DetId::Calo, HcalZDCDetId::SubdetectorId, "ZDC");
    } else
      fwLog(fwlog::kWarning) << "Invalid Calo geometry" << std::endl;
  }
}

void ValidateGeometry::validateRPCGeometry(const int regionNumber, const char* regionName) {
  clearData();

  std::vector<double> centers;

  auto const& rolls = rpcGeometry_->rolls();

  for (auto it = rolls.begin(), itEnd = rolls.end(); it != itEnd; ++it) {
    const RPCRoll* roll = *it;

    if (roll) {
      RPCDetId rpcDetId = roll->id();

      if (rpcDetId.region() == regionNumber) {
        const GeomDetUnit* det = rpcGeometry_->idToDetUnit(rpcDetId);
        GlobalPoint gp = det->surface().toGlobal(LocalPoint(0.0, 0.0, 0.0));

        const TGeoMatrix* matrix = fwGeometry_.getMatrix(rpcDetId.rawId());

        if (!matrix) {
          std::cout << "Failed to get matrix of RPC with detid: " << rpcDetId.rawId() << std::endl;
          continue;
        }

        compareTransform(gp, matrix);

        const float* shape = fwGeometry_.getShapePars(rpcDetId.rawId());

        if (!shape) {
          std::cout << "Failed to get shape of RPC with detid: " << rpcDetId.rawId() << std::endl;
          continue;
        }

        compareShape(det, shape);

        const float* parameters = fwGeometry_.getParameters(rpcDetId.rawId());

        if (parameters == nullptr) {
          std::cout << "Parameters empty for RPC with detid: " << rpcDetId.rawId() << std::endl;
          continue;
        }

        // Yes, I know that below I'm comparing the equivalence
        // of floating point numbers

        int nStrips = roll->nstrips();
        assert(nStrips == parameters[0]);

        float stripLength = roll->specificTopology().stripLength();
        assert(stripLength == parameters[1]);

        float pitch = roll->specificTopology().pitch();
        assert(pitch == parameters[2]);

        float offset = -0.5 * nStrips * pitch;

        for (int strip = 1; strip <= roll->nstrips(); ++strip) {
          LocalPoint centreOfStrip1 = roll->centreOfStrip(strip);
          LocalPoint centreOfStrip2 = LocalPoint((strip - 0.5) * pitch + offset, 0.0);

          centers.push_back(centreOfStrip1.x() - centreOfStrip2.x());
        }
      }
    }
  }

  std::string hn(regionName);
  makeHistogram(hn + ": centreOfStrip", centers);

  makeHistograms(regionName);
}

void ValidateGeometry::validateDTChamberGeometry() {
  clearData();

  auto const& chambers = dtGeometry_->chambers();

  for (auto it = chambers.begin(), itEnd = chambers.end(); it != itEnd; ++it) {
    const DTChamber* chamber = *it;

    if (chamber) {
      DTChamberId chId = chamber->id();
      GlobalPoint gp = chamber->surface().toGlobal(LocalPoint(0.0, 0.0, 0.0));

      const TGeoMatrix* matrix = fwGeometry_.getMatrix(chId.rawId());

      if (!matrix) {
        std::cout << "Failed to get matrix of DT chamber with detid: " << chId.rawId() << std::endl;
        continue;
      }

      compareTransform(gp, matrix);

      const float* shape = fwGeometry_.getShapePars(chId.rawId());

      if (!shape) {
        std::cout << "Failed to get shape of DT chamber with detid: " << chId.rawId() << std::endl;
        continue;
      }

      compareShape(chamber, shape);
    }
  }

  makeHistograms("DT chamber");
}

void ValidateGeometry::validateDTLayerGeometry() {
  clearData();

  std::vector<double> wire_positions;

  auto const& layers = dtGeometry_->layers();

  for (auto it = layers.begin(), itEnd = layers.end(); it != itEnd; ++it) {
    const DTLayer* layer = *it;

    if (layer) {
      DTLayerId layerId = layer->id();
      GlobalPoint gp = layer->surface().toGlobal(LocalPoint(0.0, 0.0, 0.0));

      const TGeoMatrix* matrix = fwGeometry_.getMatrix(layerId.rawId());

      if (!matrix) {
        std::cout << "Failed to get matrix of DT layer with detid: " << layerId.rawId() << std::endl;
        continue;
      }

      compareTransform(gp, matrix);

      const float* shape = fwGeometry_.getShapePars(layerId.rawId());

      if (!shape) {
        std::cout << "Failed to get shape of DT layer with detid: " << layerId.rawId() << std::endl;
        continue;
      }

      compareShape(layer, shape);

      const float* parameters = fwGeometry_.getParameters(layerId.rawId());

      if (parameters == nullptr) {
        std::cout << "Parameters empty for DT layer with detid: " << layerId.rawId() << std::endl;
        continue;
      }

      float width = layer->surface().bounds().width();
      assert(width == parameters[6]);

      float thickness = layer->surface().bounds().thickness();
      assert(thickness == parameters[7]);

      float length = layer->surface().bounds().length();
      assert(length == parameters[8]);

      int firstChannel = layer->specificTopology().firstChannel();
      assert(firstChannel == parameters[3]);

      int lastChannel = layer->specificTopology().lastChannel();
      int nChannels = parameters[5];
      assert(nChannels == (lastChannel - firstChannel) + 1);

      for (int wireN = firstChannel; wireN - lastChannel <= 0; ++wireN) {
        double localX1 = layer->specificTopology().wirePosition(wireN);
        double localX2 = (wireN - (firstChannel - 1) - 0.5) * parameters[0] - nChannels / 2.0 * parameters[0];

        wire_positions.push_back(localX1 - localX2);

        //std::cout<<"wireN, localXpos: "<< wireN <<" "<< localX1 <<" "<< localX2 <<std::endl;
      }
    }
  }

  makeHistogram("DT layer wire localX", wire_positions);

  makeHistograms("DT layer");
}

void ValidateGeometry::validateCSChamberGeometry(const int endcap, const char* detname) {
  clearData();

  auto const& chambers = cscGeometry_->chambers();

  for (auto it = chambers.begin(), itEnd = chambers.end(); it != itEnd; ++it) {
    const CSCChamber* chamber = *it;

    if (chamber && chamber->id().endcap() == endcap) {
      DetId detId = chamber->geographicalId();
      GlobalPoint gp = chamber->surface().toGlobal(LocalPoint(0.0, 0.0, 0.0));

      const TGeoMatrix* matrix = fwGeometry_.getMatrix(detId.rawId());

      if (!matrix) {
        std::cout << "Failed to get matrix of CSC chamber with detid: " << detId.rawId() << std::endl;
        continue;
      }

      compareTransform(gp, matrix);

      const float* shape = fwGeometry_.getShapePars(detId.rawId());

      if (!shape) {
        std::cout << "Failed to get shape of CSC chamber with detid: " << detId.rawId() << std::endl;
        continue;
      }

      compareShape(chamber, shape);
    }
  }

  makeHistograms(detname);
}

void ValidateGeometry::validateCSCLayerGeometry(const int endcap, const char* detname) {
  clearData();
  std::vector<double> strip_positions;
  std::vector<double> wire_positions;

  std::vector<double> me11_wiresLocal;
  std::vector<double> me12_wiresLocal;
  std::vector<double> me13_wiresLocal;
  std::vector<double> me14_wiresLocal;
  std::vector<double> me21_wiresLocal;
  std::vector<double> me22_wiresLocal;
  std::vector<double> me31_wiresLocal;
  std::vector<double> me32_wiresLocal;
  std::vector<double> me41_wiresLocal;
  std::vector<double> me42_wiresLocal;

  auto const& layers = cscGeometry_->layers();

  for (auto it = layers.begin(), itEnd = layers.end(); it != itEnd; ++it) {
    const CSCLayer* layer = *it;

    if (layer && layer->id().endcap() == endcap) {
      DetId detId = layer->geographicalId();
      GlobalPoint gp = layer->surface().toGlobal(LocalPoint(0.0, 0.0, 0.0));

      const TGeoMatrix* matrix = fwGeometry_.getMatrix(detId.rawId());

      if (!matrix) {
        std::cout << "Failed to get matrix of CSC layer with detid: " << detId.rawId() << std::endl;
        continue;
      }

      compareTransform(gp, matrix);

      const float* shape = fwGeometry_.getShapePars(detId.rawId());

      if (!shape) {
        std::cout << "Failed to get shape of CSC layer with detid: " << detId.rawId() << std::endl;
        continue;
      }

      compareShape(layer, shape);

      double length;

      if (shape[0] == 1) {
        length = shape[4];
      }

      else {
        std::cout << "Failed to get trapezoid from shape for CSC layer with detid: " << detId.rawId() << std::endl;
        continue;
      }

      const float* parameters = fwGeometry_.getParameters(detId.rawId());

      if (parameters == nullptr) {
        std::cout << "Parameters empty for CSC layer with detid: " << detId.rawId() << std::endl;
        continue;
      }

      int yAxisOrientation = layer->geometry()->topology()->yAxisOrientation();
      assert(yAxisOrientation == parameters[0]);

      float centreToIntersection = layer->geometry()->topology()->centreToIntersection();
      assert(centreToIntersection == parameters[1]);

      float yCentre = layer->geometry()->topology()->yCentreOfStripPlane();
      assert(yCentre == parameters[2]);

      float phiOfOneEdge = layer->geometry()->topology()->phiOfOneEdge();
      assert(phiOfOneEdge == parameters[3]);

      float stripOffset = layer->geometry()->topology()->stripOffset();
      assert(stripOffset == parameters[4]);

      float angularWidth = layer->geometry()->topology()->angularWidth();
      assert(angularWidth == parameters[5]);

      for (int nStrip = 1; nStrip <= layer->geometry()->numberOfStrips(); ++nStrip) {
        float xOfStrip1 = layer->geometry()->xOfStrip(nStrip);

        double stripAngle = phiOfOneEdge + yAxisOrientation * (nStrip - (0.5 - stripOffset)) * angularWidth;
        double xOfStrip2 = yAxisOrientation * (centreToIntersection - yCentre) * tan(stripAngle);

        strip_positions.push_back(xOfStrip1 - xOfStrip2);
      }

      int station = layer->id().station();
      int ring = layer->id().ring();

      double wireSpacingInGroup = layer->geometry()->wireTopology()->wireSpacing();
      assert(wireSpacingInGroup == parameters[6]);

      double wireSpacing = 0.0;
      // we calculate an average wire group
      // spacing from radialExtentOfTheWirePlane / numOfWireGroups

      double extentOfWirePlane = 0.0;

      if (ring == 2) {
        if (station == 1)
          extentOfWirePlane = 174.81;  //wireSpacing = 174.81/64;
        else
          extentOfWirePlane = 323.38;  //wireSpacing = 323.38/64;
      } else if (station == 1 && (ring == 1 || ring == 4))
        extentOfWirePlane = 150.5;  //wireSpacing = 150.5/48;
      else if (station == 1 && ring == 3)
        extentOfWirePlane = 164.47;  //wireSpacing = 164.47/32;
      else if (station == 2 && ring == 1)
        extentOfWirePlane = 189.97;  //wireSpacing = 189.97/112;
      else if (station == 3 && ring == 1)
        extentOfWirePlane = 170.01;  //wireSpacing = 170.01/96;
      else if (station == 4 && ring == 1)
        extentOfWirePlane = 149.73;  //wireSpacing = 149.73/96;

      float wireAngle = layer->geometry()->wireTopology()->wireAngle();
      assert(wireAngle == parameters[7]);

      //float cosWireAngle = cos(wireAngle);

      /* NOTE
         Some parameters don't seem available in a public interface
         so have to perhaps hard-code. This may not be too bad as there
         seems to be a lot of degeneracy. 
      */

      double alignmentPinToFirstWire;
      double yAlignmentFrame = 3.49;

      if (station == 1) {
        if (ring == 1 || ring == 4) {
          alignmentPinToFirstWire = 1.065;
          yAlignmentFrame = 0.0;
        }

        else  // ME12, ME 13
          alignmentPinToFirstWire = 2.85;
      }

      else if (station == 4 && ring == 1)
        alignmentPinToFirstWire = 3.04;

      else if (station == 3 && ring == 1)
        alignmentPinToFirstWire = 2.84;

      else  // ME21, ME22, ME32, ME42
        alignmentPinToFirstWire = 2.87;

      double yOfFirstWire = (yAlignmentFrame - length) + alignmentPinToFirstWire;

      int nWireGroups = layer->geometry()->numberOfWireGroups();
      double E = extentOfWirePlane / nWireGroups;

      for (int nWireGroup = 1; nWireGroup <= nWireGroups; ++nWireGroup) {
        LocalPoint centerOfWireGroup = layer->geometry()->localCenterOfWireGroup(nWireGroup);
        double yOfWire1 = centerOfWireGroup.y();

        //double yOfWire2 = (-0.5 - (nWireGroups*0.5 - 1) + (nWireGroup-1))*E;
        //yOfWire2 += 0.5*E;
        double yOfWire2 = yOfFirstWire + ((nWireGroup - 1) * E);
        yOfWire2 += wireSpacing * 0.5;

        double ydiff_local = yOfWire1 - yOfWire2;
        wire_positions.push_back(ydiff_local);

        //GlobalPoint globalPoint = layer->surface().toGlobal(LocalPoint(0.0,yOfWire1,0.0));

        /*
        float fwLocalPoint[3] = 
        {
          0.0, yOfWire2, 0.0
        };
        
        float fwGlobalPoint[3]; 
        fwGeometry_.localToGlobal(detId.rawId(), fwLocalPoint, fwGlobalPoint);
        double ydiff_global = globalPoint.y() - fwGlobalPoint[1]; 
        */

        if (station == 1) {
          if (ring == 1) {
            me11_wiresLocal.push_back(ydiff_local);
          } else if (ring == 2) {
            me12_wiresLocal.push_back(ydiff_local);
          } else if (ring == 3) {
            me13_wiresLocal.push_back(ydiff_local);
          } else if (ring == 4) {
            me14_wiresLocal.push_back(ydiff_local);
          }
        } else if (station == 2) {
          if (ring == 1) {
            me21_wiresLocal.push_back(ydiff_local);
          } else if (ring == 2) {
            me22_wiresLocal.push_back(ydiff_local);
          }
        } else if (station == 3) {
          if (ring == 1) {
            me31_wiresLocal.push_back(ydiff_local);
          } else if (ring == 2) {
            me32_wiresLocal.push_back(ydiff_local);
          }
        } else if (station == 4) {
          if (ring == 1) {
            me41_wiresLocal.push_back(ydiff_local);
          } else if (ring == 2) {
            me42_wiresLocal.push_back(ydiff_local);
          }
        }
      }
    }
  }

  std::string hn(detname);
  makeHistogram(hn + ": xOfStrip", strip_positions);

  makeHistogram(hn + ": local yOfWire", wire_positions);

  makeHistogram("ME11: local yOfWire", me11_wiresLocal);
  makeHistogram("ME12: local yOfWire", me12_wiresLocal);
  makeHistogram("ME13: local yOfWire", me13_wiresLocal);
  makeHistogram("ME14: local yOfWire", me14_wiresLocal);
  makeHistogram("ME21: local yOfWire", me21_wiresLocal);
  makeHistogram("ME22: local yOfWire", me22_wiresLocal);
  makeHistogram("ME31: local yOfWire", me31_wiresLocal);
  makeHistogram("ME32: local yOfWire", me32_wiresLocal);
  makeHistogram("ME41: local yOfWire", me41_wiresLocal);
  makeHistogram("ME42: local yOfWire", me42_wiresLocal);

  makeHistograms(detname);
}

void ValidateGeometry::validateCaloGeometry(DetId::Detector detector, int subdetector, const char* detname) {
  clearData();

  const CaloSubdetectorGeometry* geometry = caloGeometry_->getSubdetectorGeometry(detector, subdetector);

  const std::vector<DetId>& ids = geometry->getValidDetIds(detector, subdetector);

  for (auto it = ids.begin(), iEnd = ids.end(); it != iEnd; ++it) {
    unsigned int rawId = (*it).rawId();

    const float* points = fwGeometry_.getCorners(rawId);

    if (points == nullptr) {
      std::cout << "Failed to get points of " << detname << " element with detid: " << rawId << std::endl;
      continue;
    }

    auto cellGeometry = geometry->getGeometry(*it);
    const CaloCellGeometry::CornersVec& corners = cellGeometry->getCorners();

    assert(corners.size() == 8);

    for (unsigned int i = 0, offset = 0; i < 8; ++i) {
      offset = 2 * i;

      double distance = getDistance(GlobalPoint(points[i + offset], points[i + 1 + offset], points[i + 2 + offset]),
                                    GlobalPoint(corners[i].x(), corners[i].y(), corners[i].z()));

      globalDistances_.push_back(distance);
    }
  }

  makeHistograms(detname);
}

void ValidateGeometry::validateTrackerGeometry(const TrackerGeometry::DetContainer& dets, const char* detname) {
  clearData();

  for (TrackerGeometry::DetContainer::const_iterator it = dets.begin(), itEnd = dets.end(); it != itEnd; ++it) {
    GlobalPoint gp =
        (trackerGeometry_->idToDet((*it)->geographicalId()))->surface().toGlobal(LocalPoint(0.0, 0.0, 0.0));
    unsigned int rawId = (*it)->geographicalId().rawId();

    const TGeoMatrix* matrix = fwGeometry_.getMatrix(rawId);

    if (!matrix) {
      std::cout << "Failed to get matrix of " << detname << " element with detid: " << rawId << std::endl;
      continue;
    }

    compareTransform(gp, matrix);

    const float* shape = fwGeometry_.getShapePars(rawId);

    if (!shape) {
      std::cout << "Failed to get shape of " << detname << " element with detid: " << rawId << std::endl;
      continue;
    }

    compareShape(*it, shape);
  }

  makeHistograms(detname);
}

void ValidateGeometry::validatePixelTopology(const TrackerGeometry::DetContainer& dets, const char* detname) {
  std::vector<double> pixelLocalXs;
  std::vector<double> pixelLocalYs;

  for (TrackerGeometry::DetContainer::const_iterator it = dets.begin(), itEnd = dets.end(); it != itEnd; ++it) {
    unsigned int rawId = (*it)->geographicalId().rawId();

    const float* parameters = fwGeometry_.getParameters(rawId);

    if (parameters == nullptr) {
      std::cout << "Parameters empty for " << detname << " element with detid: " << rawId << std::endl;
      continue;
    }

    if (const PixelGeomDetUnit* det =
            dynamic_cast<const PixelGeomDetUnit*>(trackerGeometry_->idToDetUnit((*it)->geographicalId()))) {
      if (const PixelTopology* rpt = &det->specificTopology()) {
        int nrows = rpt->nrows();
        int ncolumns = rpt->ncolumns();

        for (int row = 1; row <= nrows; ++row) {
          for (int column = 1; column <= ncolumns; ++column) {
            LocalPoint localPoint = rpt->localPosition(MeasurementPoint(row, column));

            pixelLocalXs.push_back(localPoint.x() - fireworks::pixelLocalX(row, parameters));
            pixelLocalYs.push_back(localPoint.y() - fireworks::pixelLocalY(column, parameters));
          }
        }
      }

      else
        std::cout << "No topology for " << detname << " " << rawId << std::endl;
    }

    else
      std::cout << "No geomDetUnit for " << detname << " " << rawId << std::endl;
  }

  std::string hn(detname);
  makeHistogram(hn + " pixelLocalX", pixelLocalXs);
  makeHistogram(hn + " pixelLocalY", pixelLocalYs);
}

void ValidateGeometry::validateStripTopology(const TrackerGeometry::DetContainer& dets, const char* detname) {
  std::vector<double> radialStripLocalXs;
  std::vector<double> rectangularStripLocalXs;

  for (TrackerGeometry::DetContainer::const_iterator it = dets.begin(), itEnd = dets.end(); it != itEnd; ++it) {
    unsigned int rawId = (*it)->geographicalId().rawId();

    const float* parameters = fwGeometry_.getParameters(rawId);

    if (parameters == nullptr) {
      std::cout << "Parameters empty for " << detname << " element with detid: " << rawId << std::endl;
      continue;
    }

    if (const StripGeomDetUnit* det =
            dynamic_cast<const StripGeomDetUnit*>(trackerGeometry_->idToDet((*it)->geographicalId()))) {
      // NOTE: why the difference in dets vs. units between these and pixels? The dynamic cast above
      // fails for many of the detids...

      const StripTopology* st = dynamic_cast<const StripTopology*>(&det->specificTopology());

      if (st) {
        //assert(parameters[0] == 0);
        int nstrips = st->nstrips();
        assert(parameters[1] == nstrips);
        assert(parameters[2] == st->stripLength());

        if (const RadialStripTopology* rst =
                dynamic_cast<const RadialStripTopology*>(&(det->specificType().specificTopology()))) {
          assert(parameters[0] == 1);
          assert(parameters[3] == rst->yAxisOrientation());
          assert(parameters[4] == rst->originToIntersection());
          assert(parameters[5] == rst->phiOfOneEdge());
          assert(parameters[6] == rst->angularWidth());

          for (uint16_t strip = 1; strip <= nstrips; ++strip) {
            float stripAngle1 = rst->stripAngle(strip);
            float stripAngle2 = parameters[3] * (parameters[5] + strip * parameters[6]);

            assert((stripAngle1 - stripAngle2) == 0);

            LocalPoint stripPosition = st->localPosition(strip);

            float stripX = parameters[4] * tan(stripAngle2);
            radialStripLocalXs.push_back(stripPosition.x() - stripX);
          }
        }

        else if (dynamic_cast<const RectangularStripTopology*>(&(det->specificType().specificTopology()))) {
          assert(parameters[0] == 2);
          assert(parameters[3] == st->pitch());

          for (uint16_t strip = 1; strip <= nstrips; ++strip) {
            LocalPoint stripPosition = st->localPosition(strip);
            float stripX = -parameters[1] * 0.5 * parameters[3];
            stripX += strip * parameters[3];
            rectangularStripLocalXs.push_back(stripPosition.x() - stripX);
          }
        }

        else if (dynamic_cast<const TrapezoidalStripTopology*>(&(det->specificType().specificTopology()))) {
          assert(parameters[0] == 3);
          assert(parameters[3] == st->pitch());
        }

        else
          std::cout << "Failed to get pitch for " << detname << " " << rawId << std::endl;
      }

      else
        std::cout << "Failed cast to StripTopology for " << detname << " " << rawId << std::endl;
    }

    //else
    //  std::cout<<"Failed cast to StripGeomDetUnit for "<< detname <<" "<< rawId <<std::endl;
  }

  std::string hn(detname);
  makeHistogram(hn + " radial strip localX", radialStripLocalXs);
  makeHistogram(hn + " rectangular strip localX", rectangularStripLocalXs);
}

void ValidateGeometry::compareTransform(const GlobalPoint& gp, const TGeoMatrix* matrix) {
  double local[3] = {0.0, 0.0, 0.0};

  double global[3];

  matrix->LocalToMaster(local, global);

  double distance = getDistance(GlobalPoint(global[0], global[1], global[2]), gp);
  globalDistances_.push_back(distance);
}

void ValidateGeometry::compareShape(const GeomDet* det, const float* shape) {
  double shape_topWidth;
  double shape_bottomWidth;
  double shape_length;
  double shape_thickness;

  if (shape[0] == 1) {
    shape_topWidth = shape[2];
    shape_bottomWidth = shape[1];
    shape_length = shape[4];
    shape_thickness = shape[3];
  }

  else if (shape[0] == 2) {
    shape_topWidth = shape[1];
    shape_bottomWidth = shape[1];
    shape_length = shape[2];
    shape_thickness = shape[3];
  }

  else {
    std::cout << "Failed to get box or trapezoid from shape" << std::endl;
    return;
  }

  double topWidth, bottomWidth;
  double length, thickness;

  const Bounds* bounds = &(det->surface().bounds());

  if (const TrapezoidalPlaneBounds* tpbs = dynamic_cast<const TrapezoidalPlaneBounds*>(bounds)) {
    std::array<const float, 4> const& ps = tpbs->parameters();

    assert(ps.size() == 4);

    bottomWidth = ps[0];
    topWidth = ps[1];
    thickness = ps[2];
    length = ps[3];
  }

  else if ((dynamic_cast<const RectangularPlaneBounds*>(bounds))) {
    length = det->surface().bounds().length() * 0.5;
    topWidth = det->surface().bounds().width() * 0.5;
    bottomWidth = topWidth;
    thickness = det->surface().bounds().thickness() * 0.5;
  }

  else {
    std::cout << "Failed to get bounds" << std::endl;
    return;
  }

  //assert((tgeotrap && trapezoid) || (tgeobbox && rectangle));

  /*
  std::cout<<"topWidth: "<< shape_topWidth <<" "<< topWidth <<std::endl;
  std::cout<<"bottomWidth: "<< shape_bottomWidth <<" "<< bottomWidth <<std::endl;
  std::cout<<"length: "<< shape_length <<" "<< length <<std::endl;
  std::cout<<"thickness: "<< shape_thickness <<" "<< thickness <<std::endl;
  */

  topWidths_.push_back(fabs(shape_topWidth - topWidth));
  bottomWidths_.push_back(fabs(shape_bottomWidth - bottomWidth));
  lengths_.push_back(fabs(shape_length - length));
  thicknesses_.push_back(fabs(shape_thickness - thickness));

  return;
}

double ValidateGeometry::getDistance(const GlobalPoint& p1, const GlobalPoint& p2) {
  /*
  std::cout<<"X: "<< p1.x() <<" "<< p2.x() <<std::endl;
  std::cout<<"Y: "<< p1.y() <<" "<< p2.y() <<std::endl;
  std::cout<<"Z: "<< p1.z() <<" "<< p2.z() <<std::endl;
  */

  return sqrt((p1.x() - p2.x()) * (p1.x() - p2.x()) + (p1.y() - p2.y()) * (p1.y() - p2.y()) +
              (p1.z() - p2.z()) * (p1.z() - p2.z()));
}

void ValidateGeometry::makeHistograms(const char* detector) {
  outFile_->cd();

  std::string d(detector);

  std::string gdn = d + ": distance between points in global coordinates";
  makeHistogram(gdn, globalDistances_);

  std::string twn = d + ": absolute difference between top widths (along X)";
  makeHistogram(twn, topWidths_);

  std::string bwn = d + ": absolute difference between bottom widths (along X)";
  makeHistogram(bwn, bottomWidths_);

  std::string ln = d + ": absolute difference between lengths (along Y)";
  makeHistogram(ln, lengths_);

  std::string tn = d + ": absolute difference between thicknesses (along Z)";
  makeHistogram(tn, thicknesses_);

  return;
}

void ValidateGeometry::makeHistogram(const std::string& name, std::vector<double>& data) {
  if (data.empty())
    return;

  std::vector<double>::iterator it;

  it = std::min_element(data.begin(), data.end());
  double minE = *it;

  it = std::max_element(data.begin(), data.end());
  double maxE = *it;

  std::vector<double>::iterator itEnd = data.end();

  TH1D hist(name.c_str(), name.c_str(), 100, minE * (1 + 0.10), maxE * (1 + 0.10));

  for (it = data.begin(); it != itEnd; ++it)
    hist.Fill(*it);

  hist.GetXaxis()->SetTitle("[cm]");
  hist.Write();
}

void ValidateGeometry::beginJob() { outFile_->cd(); }

void ValidateGeometry::endJob() {
  std::cout << "Done. " << std::endl;
  std::cout << "Results written to " << outfileName_ << std::endl;
  outFile_->Close();
}

DEFINE_FWK_MODULE(ValidateGeometry);
