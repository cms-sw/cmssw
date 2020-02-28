// -*- C++ -*-
//
/* 
 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/

//
// Original Author:  Riccardo Ranieri
//         Created:  Wed May 3 10:30:00 CEST 2006
//     Modified by:  Michael Case, April 2010.
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerDebugNavigator.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerStringToEnum.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"

// output
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <bitset>

class ModuleInfo : public edm::one::EDAnalyzer<> {
public:
  explicit ModuleInfo(const edm::ParameterSet&);
  ~ModuleInfo() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  bool fromDDD_;
  bool printDDD_;
};

static const double density_units = 6.24151e+18;

ModuleInfo::ModuleInfo(const edm::ParameterSet& ps) {
  fromDDD_ = ps.getParameter<bool>("fromDDD");
  printDDD_ = ps.getUntrackedParameter<bool>("printDDD", true);
}

ModuleInfo::~ModuleInfo() {}

// ------------ method called to produce the data  ------------
void ModuleInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogInfo("ModuleInfo") << "begins";

  // output file
  std::ofstream Output("ModuleInfo.log", std::ios::out);
  // TEC output as Martin Weber's
  std::ofstream TECOutput("TECLayout_CMSSW.dat", std::ios::out);
  // Numbering Scheme
  std::ofstream NumberingOutput("ModuleNumbering.dat", std::ios::out);
  //

  //
  // get the GeometricDet
  //
  edm::ESHandle<GeometricDet> rDD;
  edm::ESHandle<std::vector<GeometricDetExtra> > rDDE;
  iSetup.get<IdealGeometryRecord>().get(rDD);
  iSetup.get<IdealGeometryRecord>().get(rDDE);

  edm::LogInfo("ModuleInfo") << " Top node is  " << rDD.product() << " " << rDD.product()->name() << std::endl;
  edm::LogInfo("ModuleInfo") << " And Contains  Daughters: " << rDD.product()->deepComponents().size() << std::endl;
  CmsTrackerDebugNavigator nav(*rDDE.product());
  nav.dump(*rDD.product(), *rDDE.product());
  //
  //first instance tracking geometry
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get(pDD);
  edm::ESHandle<TrackerTopology> tTopo_handle;
  iSetup.get<TrackerTopologyRcd>().get(tTopo_handle);
  const TrackerTopology* tTopo = tTopo_handle.product();
  //

  // counters
  unsigned int pxbN = 0;
  unsigned int pxb_fullN = 0;
  unsigned int pxb_halfN = 0;
  unsigned int pxfN = 0;
  unsigned int pxf_1x2N = 0;
  unsigned int pxf_1x5N = 0;
  unsigned int pxf_2x3N = 0;
  unsigned int pxf_2x4N = 0;
  unsigned int pxf_2x5N = 0;
  unsigned int tibN = 0;
  unsigned int tib_L12_rphiN = 0;
  unsigned int tib_L12_sterN = 0;
  unsigned int tib_L34_rphiN = 0;
  unsigned int tidN = 0;
  unsigned int tid_r1_rphiN = 0;
  unsigned int tid_r1_sterN = 0;
  unsigned int tid_r2_rphiN = 0;
  unsigned int tid_r2_sterN = 0;
  unsigned int tid_r3_rphiN = 0;
  unsigned int tobN = 0;
  unsigned int tob_L12_rphiN = 0;
  unsigned int tob_L12_sterN = 0;
  unsigned int tob_L34_rphiN = 0;
  unsigned int tob_L56_rphiN = 0;
  unsigned int tecN = 0;
  unsigned int tec_r1_rphiN = 0;
  unsigned int tec_r1_sterN = 0;
  unsigned int tec_r2_rphiN = 0;
  unsigned int tec_r2_sterN = 0;
  unsigned int tec_r3_rphiN = 0;
  unsigned int tec_r4_rphiN = 0;
  unsigned int tec_r5_rphiN = 0;
  unsigned int tec_r5_sterN = 0;
  unsigned int tec_r6_rphiN = 0;
  unsigned int tec_r7_rphiN = 0;
  //
  double volume_total = 0.0;
  double weight_total = 0.0;
  double activeSurface_total = 0.0;
  double volume_pxb = 0.0;
  double weight_pxb = 0.0;
  double activeSurface_pxb = 0.0;
  double volume_pxf = 0.0;
  double weight_pxf = 0.0;
  double activeSurface_pxf = 0.0;
  double volume_tib = 0.0;
  double weight_tib = 0.0;
  double activeSurface_tib = 0.0;
  double volume_tid = 0.0;
  double weight_tid = 0.0;
  double activeSurface_tid = 0.0;
  double volume_tob = 0.0;
  double weight_tob = 0.0;
  double activeSurface_tob = 0.0;
  double volume_tec = 0.0;
  double weight_tec = 0.0;
  double activeSurface_tec = 0.0;
  //

  std::vector<const GeometricDet*> modules = (*rDD).deepComponents();
  Output << "************************ List of modules with positions ************************" << std::endl;
  // MEC: 2010-04-13: need to find corresponding GeometricDetExtra.
  std::vector<GeometricDetExtra>::const_iterator gdei(rDDE->begin()), gdeEnd(rDDE->end());
  for (auto& module : modules) {
    unsigned int rawid = module->geographicalID().rawId();
    DetId id(rawid);
    gdei = rDDE->begin();
    for (; gdei != gdeEnd; ++gdei) {
      if (gdei->geographicalId() == module->geographicalId())
        break;
    }

    if (gdei == gdeEnd)
      throw cms::Exception("ModuleInfo") << "THERE IS NO MATCHING DetId in the GeometricDetExtra";  //THIS never happens!

    GeometricDet::NavRange detPos = module->navpos();
    Output << std::fixed << std::setprecision(6);  // set as default 6 decimal digits
    std::bitset<32> binary_rawid(rawid);
    Output << " ******** raw Id = " << rawid << " (" << binary_rawid << ") ";
    if (fromDDD_ && printDDD_) {
      Output << "\t nav type = " << detPos;
    }
    Output << std::endl;
    int subdetid = module->geographicalID().subdetId();
    double volume = gdei->volume() / 1000;  // mm3->cm3
    double density = gdei->density() / density_units;
    double weight = gdei->weight() / density_units / 1000.;    // [kg], hence the factor 1000;
    double thickness = module->bounds()->thickness() * 10000;  // cm-->um
    double activeSurface = volume / (thickness / 10000);       // cm2 (thickness in um)

    volume_total += volume;
    weight_total += weight;
    activeSurface_total += activeSurface;
    switch (subdetid) {
        // PXB
      case PixelSubdetector::PixelBarrel: {
        pxbN++;
        volume_pxb += volume;
        weight_pxb += weight;
        activeSurface_pxb += activeSurface;
        std::string name = module->name();
        if (name == "PixelBarrelActiveFull")
          pxb_fullN++;
        if (name == "PixelBarrelActiveHalf")
          pxb_halfN++;
        unsigned int theLayer = tTopo->pxbLayer(id);
        unsigned int theLadder = tTopo->pxbLadder(id);
        unsigned int theModule = tTopo->pxbModule(id);

        Output << " PXB"
               << "\t"
               << "Layer " << theLayer << " Ladder " << theLadder << "\t"
               << " module " << theModule << " " << name << "\t";
        if (fromDDD_ && printDDD_) {
          Output << "son of " << gdei->parents()[gdei->parents().size() - 3].logicalPart().name() << std::endl;
        } else {
          Output << " NO DDD Hierarchy available " << std::endl;
        }
        break;
      }

        // PXF
      case PixelSubdetector::PixelEndcap: {
        pxfN++;
        volume_pxf += volume;
        weight_pxf += weight;
        activeSurface_pxf += activeSurface;
        std::string name = module->name();
        if (name == "PixelForwardActive1x2")
          pxf_1x2N++;
        if (name == "PixelForwardActive1x5")
          pxf_1x5N++;
        if (name == "PixelForwardActive2x3")
          pxf_2x3N++;
        if (name == "PixelForwardActive2x4")
          pxf_2x4N++;
        if (name == "PixelForwardActive2x5")
          pxf_2x5N++;
        unsigned int thePanel = tTopo->pxfPanel(id);
        unsigned int theDisk = tTopo->pxfDisk(id);
        unsigned int theBlade = tTopo->pxfBlade(id);
        unsigned int theModule = tTopo->pxfModule(id);
        std::string side;
        side = (tTopo->pxfSide(id) == 1) ? "-" : "+";
        Output << " PXF" << side << "\t"
               << "Disk " << theDisk << " Blade " << theBlade << " Panel " << thePanel << "\t"
               << " module " << theModule << "\t" << name << "\t";
        if (fromDDD_ && printDDD_) {
          Output << "son of " << gdei->parents()[gdei->parents().size() - 3].logicalPart().name() << std::endl;
        } else {
          Output << " NO DDD Hierarchy available " << std::endl;
        }
        break;
      }

        // TIB
      case StripSubdetector::TIB: {
        tibN++;
        volume_tib += volume;
        weight_tib += weight;
        activeSurface_tib += activeSurface;
        std::string name = module->name();
        if (name == "TIBActiveRphi0")
          tib_L12_rphiN++;
        if (name == "TIBActiveSter0")
          tib_L12_sterN++;
        if (name == "TIBActiveRphi2")
          tib_L34_rphiN++;
        unsigned int theLayer = tTopo->tibLayer(id);
        std::vector<unsigned int> theString = tTopo->tibStringInfo(id);
        unsigned int theModule = tTopo->tibModule(id);
        std::string side;
        std::string part;
        side = (theString[0] == 1) ? "-" : "+";
        part = (theString[1] == 1) ? "int" : "ext";

        Output << " TIB" << side << "\t"
               << "Layer " << theLayer << " " << part << "\t"
               << "string " << theString[2] << "\t"
               << " module " << theModule << " " << name << "\t";
        if (fromDDD_ && printDDD_) {
          Output << "son of " << gdei->parents()[gdei->parents().size() - 3].logicalPart().name();
        } else {
          Output << " NO DDD Hierarchy available ";
        }
        Output << " " << module->translation().X() << "   \t" << module->translation().Y() << "   \t"
               << module->translation().Z() << std::endl;
        break;
      }

        // TID
      case StripSubdetector::TID: {
        tidN++;
        volume_tid += volume;
        weight_tid += weight;
        activeSurface_tid += activeSurface;
        std::string name = module->name();
        if (name == "TIDModule0RphiActive")
          tid_r1_rphiN++;
        if (name == "TIDModule0StereoActive")
          tid_r1_sterN++;
        if (name == "TIDModule1RphiActive")
          tid_r2_rphiN++;
        if (name == "TIDModule1StereoActive")
          tid_r2_sterN++;
        if (name == "TIDModule2RphiActive")
          tid_r3_rphiN++;
        unsigned int theDisk = tTopo->tidWheel(id);
        unsigned int theRing = tTopo->tidRing(id);
        std::string side;
        std::string part;
        side = (tTopo->tidSide(id) == 1) ? "-" : "+";
        part = (tTopo->tidOrder(id) == 1) ? "back" : "front";
        Output << " TID" << side << "\t"
               << "Disk " << theDisk << " Ring " << theRing << " " << part << "\t"
               << " module " << tTopo->tidModule(id) << "\t" << name << "\t";
        if (fromDDD_ && printDDD_) {
          Output << "son of " << gdei->parents()[gdei->parents().size() - 3].logicalPart().name();
        } else {
          Output << " NO DDD Hierarchy available ";
        }
        Output << " " << module->translation().X() << "   \t" << module->translation().Y() << "   \t"
               << module->translation().Z() << std::endl;
        break;
      }

        // TOB
      case StripSubdetector::TOB: {
        tobN++;
        volume_tob += volume;
        weight_tob += weight;
        activeSurface_tob += activeSurface;
        std::string name = module->name();
        if (name == "TOBActiveRphi0")
          tob_L12_rphiN++;
        if (name == "TOBActiveSter0")
          tob_L12_sterN++;
        if (name == "TOBActiveRphi2")
          tob_L34_rphiN++;
        if (name == "TOBActiveRphi4")
          tob_L56_rphiN++;
        unsigned int theLayer = tTopo->tobLayer(id);
        unsigned int theModule = tTopo->tobModule(id);
        std::string side;
        std::string part;
        side = (tTopo->tobSide(id) == 1) ? "-" : "+";

        Output << " TOB" << side << "\t"
               << "Layer " << theLayer << "\t"
               << "rod " << tTopo->tobRod(id) << " module " << theModule << "\t" << name << "\t";
        if (fromDDD_ && printDDD_) {
          Output << "son of " << gdei->parents()[gdei->parents().size() - 3].logicalPart().name();
        } else {
          Output << " NO DDD Hierarchy available ";
        }
        Output << " " << module->translation().X() << "   \t" << module->translation().Y() << "   \t"
               << module->translation().Z() << std::endl;
        break;
      }

        // TEC
      case StripSubdetector::TEC: {
        tecN++;
        volume_tec += volume;
        weight_tec += weight;
        activeSurface_tec += activeSurface;
        std::string name = module->name();
        if (name == "TECModule0RphiActive")
          tec_r1_rphiN++;
        if (name == "TECModule0StereoActive")
          tec_r1_sterN++;
        if (name == "TECModule1RphiActive")
          tec_r2_rphiN++;
        if (name == "TECModule1StereoActive")
          tec_r2_sterN++;
        if (name == "TECModule2RphiActive")
          tec_r3_rphiN++;
        if (name == "TECModule3RphiActive")
          tec_r4_rphiN++;
        if (name == "TECModule4RphiActive")
          tec_r5_rphiN++;
        if (name == "TECModule4StereoActive")
          tec_r5_sterN++;
        if (name == "TECModule5RphiActive")
          tec_r6_rphiN++;
        if (name == "TECModule6RphiActive")
          tec_r7_rphiN++;
        unsigned int theWheel = tTopo->tecWheel(id);
        unsigned int theModule = tTopo->tecModule(id);
        unsigned int theRing = tTopo->tecRing(id);
        std::string side;
        std::string petal;
        side = (tTopo->tecSide(id) == 1) ? "-" : "+";
        petal = (tTopo->tecOrder(id) == 1) ? "back" : "front";
        Output << " TEC" << side << "\t"
               << "Wheel " << theWheel << " Petal " << tTopo->tecPetalNumber(id) << " " << petal << " Ring " << theRing
               << "\t"
               << "\t"
               << " module " << theModule << "\t" << name << "\t";
        if (fromDDD_ && printDDD_) {
          Output << "son of " << gdei->parents()[gdei->parents().size() - 3].logicalPart().name();
        } else {
          Output << " NO DDD Hierarchy available ";
        }
        Output << " " << module->translation().X() << "   \t" << module->translation().Y() << "   \t"
               << module->translation().Z() << std::endl;

        // TEC output as Martin Weber's
        int out_side = (tTopo->tecSide(id) == 1) ? -1 : 1;
        unsigned int out_disk = tTopo->tecWheel(id);
        unsigned int out_sector = tTopo->tecPetalNumber(id);
        int out_petal = (tTopo->tecOrder(id) == 1) ? 1 : -1;
        // swap sector numbers for TEC-
        if (out_side == -1) {
          // fine for back petals, substract 1 for front petals
          if (out_petal == -1) {
            out_sector = (out_sector + 6) % 8 + 1;
          }
        }
        unsigned int out_ring = tTopo->tecRing(id);
        int out_sensor = 0;
        if (name == "TECModule0RphiActive")
          out_sensor = -1;
        if (name == "TECModule0StereoActive")
          out_sensor = 1;
        if (name == "TECModule1RphiActive")
          out_sensor = -1;
        if (name == "TECModule1StereoActive")
          out_sensor = 1;
        if (name == "TECModule2RphiActive")
          out_sensor = -1;
        if (name == "TECModule3RphiActive")
          out_sensor = -1;
        if (name == "TECModule4RphiActive")
          out_sensor = -1;
        if (name == "TECModule4StereoActive")
          out_sensor = 1;
        if (name == "TECModule5RphiActive")
          out_sensor = -1;
        if (name == "TECModule6RphiActive")
          out_sensor = -1;
        unsigned int out_module;
        if (out_ring == 1 || out_ring == 2 || out_ring == 5) {
          // rings with stereo modules
          // create number odd by default
          out_module = 2 * (tTopo->tecModule(id) - 1) + 1;
          if (out_sensor == 1) {
            // in even rings, stereo modules are the even ones
            if (out_ring == 2)
              out_module += 1;
          } else
              // in odd rings, stereo modules are the odd ones
              if (out_ring != 2)
            out_module += 1;
        } else {
          out_module = tTopo->tecModule(id);
        }
        double out_x = module->translation().X();
        double out_y = module->translation().Y();
        double out_z = module->translation().Z();
        double out_r = sqrt(module->translation().X() * module->translation().X() +
                            module->translation().Y() * module->translation().Y());
        double out_phi_rad = atan2(module->translation().Y(), module->translation().X());
        TECOutput << out_side << " " << out_disk << " " << out_sector << " " << out_petal << " " << out_ring << " "
                  << out_module << " " << out_sensor << " " << out_x << " " << out_y << " " << out_z << " " << out_r
                  << " " << out_phi_rad << std::endl;
        //
        break;
      }
      default:
        Output << " WARNING no Silicon Strip detector, I got a " << rawid << std::endl;
        ;
    }

    // Local axes from Reco
    const GeomDet* geomdet = pDD->idToDet(module->geographicalID());
    // Global Coordinates (i,j,k)
    LocalVector xLocal(1, 0, 0);
    LocalVector yLocal(0, 1, 0);
    LocalVector zLocal(0, 0, 1);
    // Versor components
    GlobalVector xGlobal = (geomdet->surface()).toGlobal(xLocal);
    GlobalVector yGlobal = (geomdet->surface()).toGlobal(yLocal);
    GlobalVector zGlobal = (geomdet->surface()).toGlobal(zLocal);
    //

    // Output: set as default 4 decimal digits (0.1 um or 0.1 deg/rad)
    // active area center
    Output << "\t"
           << "volume " << std::fixed << std::setprecision(3) << volume << " cm3 \t"
           << "density " << std::fixed << std::setprecision(3) << density << " g/cm3 \t"
           << "weight " << std::fixed << std::setprecision(6) << weight << " kg \t"
           << "thickness " << std::fixed << std::setprecision(0) << thickness << " um \t"
           << " active area " << std::fixed << std::setprecision(2) << activeSurface << " cm2" << std::endl;
    Output << "\tActive Area Center" << std::endl;
    Output << "\t O = (" << std::fixed << std::setprecision(4) << module->translation().X() << "," << std::fixed
           << std::setprecision(4) << module->translation().Y() << "," << std::fixed << std::setprecision(4)
           << module->translation().Z() << ")" << std::endl;
    //
    double polarRadius = std::sqrt(module->translation().X() * module->translation().X() +
                                   module->translation().Y() * module->translation().Y());
    double phiDeg = atan2(module->translation().Y(), module->translation().X()) * 360. / 6.283185307;
    double phiRad = atan2(module->translation().Y(), module->translation().X());
    //
    Output << "\t\t polar radius " << std::fixed << std::setprecision(4) << polarRadius << "\t"
           << "phi [deg] " << std::fixed << std::setprecision(4) << phiDeg << "\t"
           << "phi [rad] " << std::fixed << std::setprecision(4) << phiRad << std::endl;
    // active area versors (rotation matrix)
    DD3Vector x, y, z;
    module->rotation().GetComponents(x, y, z);
    Output << "\tActive Area Rotation Matrix" << std::endl;
    Output << "\t z = n = (" << std::fixed << std::setprecision(4) << z.X() << "," << std::fixed << std::setprecision(4)
           << z.Y() << "," << std::fixed << std::setprecision(4) << z.Z() << ")" << std::endl
           << "\t [Rec] = (" << std::fixed << std::setprecision(4) << zGlobal.x() << "," << std::fixed
           << std::setprecision(4) << zGlobal.y() << "," << std::fixed << std::setprecision(4) << zGlobal.z() << ")"
           << std::endl
           << "\t x = t = (" << std::fixed << std::setprecision(4) << x.X() << "," << std::fixed << std::setprecision(4)
           << x.Y() << "," << std::fixed << std::setprecision(4) << x.Z() << ")" << std::endl
           << "\t [Rec] = (" << std::fixed << std::setprecision(4) << xGlobal.x() << "," << std::fixed
           << std::setprecision(4) << xGlobal.y() << "," << std::fixed << std::setprecision(4) << xGlobal.z() << ")"
           << std::endl
           << "\t y = k = (" << std::fixed << std::setprecision(4) << y.X() << "," << std::fixed << std::setprecision(4)
           << y.Y() << "," << std::fixed << std::setprecision(4) << y.Z() << ")" << std::endl
           << "\t [Rec] = (" << std::fixed << std::setprecision(4) << yGlobal.x() << "," << std::fixed
           << std::setprecision(4) << yGlobal.y() << "," << std::fixed << std::setprecision(4) << yGlobal.z() << ")"
           << std::endl;

    // NumberingScheme
    NumberingOutput << rawid;
    if (fromDDD_ && printDDD_) {
      NumberingOutput << " " << detPos;
    }
    NumberingOutput << " " << std::fixed << std::setprecision(4) << module->translation().X() << " " << std::fixed
                    << std::setprecision(4) << module->translation().Y() << " " << std::fixed << std::setprecision(4)
                    << module->translation().Z() << " " << std::endl;
    //
  }

  // params
  // Pixel
  unsigned int chan_per_psi = 52 * 80;
  unsigned int psi_pxb = 16 * pxb_fullN + 8 * pxb_halfN;
  unsigned int chan_pxb = psi_pxb * chan_per_psi;
  unsigned int psi_pxf = 2 * pxf_1x2N + 5 * pxf_1x5N + 6 * pxf_2x3N + 8 * pxf_2x4N + 10 * pxf_2x5N;
  unsigned int chan_pxf = psi_pxf * chan_per_psi;
  // Strip
  unsigned int chan_per_apv = 128;
  unsigned int apv_tib = 6 * (tib_L12_rphiN + tib_L12_sterN) + 4 * tib_L34_rphiN;
  unsigned int chan_tib = apv_tib * chan_per_apv;
  unsigned int apv_tid = 6 * (tid_r1_rphiN + tid_r1_sterN) + 6 * (tid_r2_rphiN + tid_r2_sterN) + 4 * tid_r3_rphiN;
  unsigned int chan_tid = apv_tid * chan_per_apv;
  unsigned int apv_tob = 4 * (tob_L12_rphiN + tob_L12_sterN) + 4 * tob_L34_rphiN + 6 * tob_L56_rphiN;
  unsigned int chan_tob = apv_tob * chan_per_apv;
  unsigned int apv_tec = 6 * (tec_r1_rphiN + tec_r1_sterN) + 6 * (tec_r2_rphiN + tec_r2_sterN) + 4 * tec_r3_rphiN +
                         4 * tec_r4_rphiN + 6 * (tec_r5_rphiN + tec_r5_sterN) + 4 * tec_r6_rphiN + 4 * tec_r7_rphiN;
  unsigned int chan_tec = apv_tec * chan_per_apv;
  unsigned int psi_tot = psi_pxb + psi_pxf;
  unsigned int apv_tot = apv_tib + apv_tid + apv_tob + apv_tec;
  unsigned int chan_pixel = chan_pxb + chan_pxf;
  unsigned int chan_strip = chan_tib + chan_tid + chan_tob + chan_tec;
  unsigned int chan_tot = chan_pixel + chan_strip;
  //

  // summary
  Output << "---------------------" << std::endl;
  Output << " Counters " << std::endl;
  Output << "---------------------" << std::endl;
  Output << " PXB    = " << pxbN << std::endl;
  Output << "   Full = " << pxb_fullN << std::endl;
  Output << "   Half = " << pxb_halfN << std::endl;
  Output << "   Active Silicon Detectors" << std::endl;
  Output << "     Weight  = " << weight_pxb << " kg" << std::endl;
  Output << "     Volume  = " << volume_pxb << " cm3" << std::endl;
  Output << "     Surface = " << activeSurface_pxb << " cm2" << std::endl;
  Output << "        PSI46s   = " << psi_pxb << std::endl;
  Output << "        channels = " << chan_pxb << std::endl;
  Output << " PXF    = " << pxfN << std::endl;
  Output << "   1x2 = " << pxf_1x2N << std::endl;
  Output << "   1x5 = " << pxf_1x5N << std::endl;
  Output << "   2x3 = " << pxf_2x3N << std::endl;
  Output << "   2x4 = " << pxf_2x4N << std::endl;
  Output << "   2x5 = " << pxf_2x5N << std::endl;
  Output << "   Active Silicon Detectors" << std::endl;
  Output << "     Weight  = " << weight_pxf << " kg" << std::endl;
  Output << "     Volume  = " << volume_pxf << " cm3" << std::endl;
  Output << "     Surface = " << activeSurface_pxf << " cm2" << std::endl;
  Output << "        PSI46s   = " << psi_pxf << std::endl;
  Output << "        channels = " << chan_pxf << std::endl;
  Output << " TIB    = " << tibN << std::endl;
  Output << "   L12 rphi   = " << tib_L12_rphiN << std::endl;
  Output << "   L12 stereo = " << tib_L12_sterN << std::endl;
  Output << "   L34        = " << tib_L34_rphiN << std::endl;
  Output << "   Active Silicon Detectors" << std::endl;
  Output << "     Weight  = " << weight_tib << " kg" << std::endl;
  Output << "     Volume  = " << volume_tib << " cm3" << std::endl;
  Output << "     Surface = " << activeSurface_tib << " cm2" << std::endl;
  Output << "        APV25s   = " << apv_tib << std::endl;
  Output << "        channels = " << chan_tib << std::endl;
  Output << " TID    = " << tidN << std::endl;
  Output << "   r1 rphi    = " << tid_r1_rphiN << std::endl;
  Output << "   r1 stereo  = " << tid_r1_sterN << std::endl;
  Output << "   r2 rphi    = " << tid_r2_rphiN << std::endl;
  Output << "   r2 stereo  = " << tid_r2_sterN << std::endl;
  Output << "   r3 rphi    = " << tid_r3_rphiN << std::endl;
  Output << "   Active Silicon Detectors" << std::endl;
  Output << "     Weight  = " << weight_tid << " kg" << std::endl;
  Output << "     Volume  = " << volume_tid << " cm3" << std::endl;
  ;
  Output << "     Surface = " << activeSurface_tid << " cm2" << std::endl;
  Output << "        APV25s   = " << apv_tid << std::endl;
  Output << "        channels = " << chan_tid << std::endl;
  Output << " TOB    = " << tobN << std::endl;
  Output << "   L12 rphi   = " << tob_L12_rphiN << std::endl;
  Output << "   L12 stereo = " << tob_L12_sterN << std::endl;
  Output << "   L34        = " << tob_L34_rphiN << std::endl;
  Output << "   L56        = " << tob_L56_rphiN << std::endl;
  Output << "   Active Silicon Detectors" << std::endl;
  Output << "     Weight  = " << weight_tob << " kg" << std::endl;
  Output << "     Volume  = " << volume_tob << " cm3" << std::endl;
  Output << "     Surface = " << activeSurface_tob << " cm2" << std::endl;
  Output << "        APV25s   = " << apv_tob << std::endl;
  Output << "        channels = " << chan_tob << std::endl;
  Output << " TEC    = " << tecN << std::endl;
  Output << "   r1 rphi    = " << tec_r1_rphiN << std::endl;
  Output << "   r1 stereo  = " << tec_r1_sterN << std::endl;
  Output << "   r2 rphi    = " << tec_r2_rphiN << std::endl;
  Output << "   r2 stereo  = " << tec_r2_sterN << std::endl;
  Output << "   r3 rphi    = " << tec_r3_rphiN << std::endl;
  Output << "   r4 rphi    = " << tec_r4_rphiN << std::endl;
  Output << "   r5 rphi    = " << tec_r5_rphiN << std::endl;
  Output << "   r5 stereo  = " << tec_r5_sterN << std::endl;
  Output << "   r6 rphi    = " << tec_r6_rphiN << std::endl;
  Output << "   r7 rphi    = " << tec_r7_rphiN << std::endl;
  Output << "   Active Silicon Detectors" << std::endl;
  Output << "     Weight  = " << weight_tec << " kg" << std::endl;
  Output << "     Volume  = " << volume_tec << " cm3" << std::endl;
  Output << "     Surface = " << activeSurface_tec << " cm2" << std::endl;
  Output << "        APV25s   = " << apv_tec << std::endl;
  Output << "        channels = " << chan_tec << std::endl;
  Output << "---------------------" << std::endl;
  Output << " Total Weight      = " << weight_total << " kg" << std::endl;
  Output << " Total Volume      = " << volume_total << " cm3" << std::endl;
  Output << " Total Active Area = " << activeSurface_total << " cm2" << std::endl;
  Output << "        PSI46s   = " << psi_tot << std::endl;
  Output << "        APV25s   = " << apv_tot << std::endl;
  Output << "        pixel channels = " << chan_pixel << std::endl;
  Output << "        strip channels = " << chan_strip << std::endl;
  Output << "        total channels = " << chan_tot << std::endl;
  //
}

//define this as a plug-in
DEFINE_FWK_MODULE(ModuleInfo);
