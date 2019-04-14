///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalModuleAlgo.cc
// Description: Geometry factory class for HGCal (EE and HESil)
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>

#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalModuleAlgo.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

DDHGCalModuleAlgo::DDHGCalModuleAlgo() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModuleAlgo: Creating an instance";
#endif
}

DDHGCalModuleAlgo::~DDHGCalModuleAlgo() {}

void DDHGCalModuleAlgo::initialize(const DDNumericArguments& nArgs,
                                   const DDVectorArguments& vArgs,
                                   const DDMapArguments&,
                                   const DDStringArguments& sArgs,
                                   const DDStringVectorArguments& vsArgs) {
  wafer = vsArgs["WaferName"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModuleAlgo: " << wafer.size()
				<< " wafers";
  for (unsigned int i = 0; i < wafer.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Wafer[" << i << "] " << wafer[i];
#endif
  materials = vsArgs["MaterialNames"];
  names = vsArgs["VolumeNames"];
  thick = vArgs["Thickness"];
  for (unsigned int i = 0; i < materials.size(); ++i) {
    copyNumber.emplace_back(1);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModuleAlgo: " << materials.size() 
				<< " types of volumes";
  for (unsigned int i = 0; i < names.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << names[i] 
				  << " of thickness " << thick[i] 
				  << " filled with " << materials[i]
				  << " first copy number " << copyNumber[i];
#endif
  layers = dbl_to_int(vArgs["Layers"]);
  layerThick = vArgs["LayerThick"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModuleAlgo: " << layers.size() 
				<< " blocks";
  for (unsigned int i = 0; i < layers.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] of thickness " 
				  << layerThick[i] << " with " << layers[i] 
				  << " layers";
#endif
  layerType = dbl_to_int(vArgs["LayerType"]);
  layerSense = dbl_to_int(vArgs["LayerSense"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModuleAlgo: " << layerType.size() 
				<< " layers";
  for (unsigned int i = 0; i < layerType.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " 
				  << layerType[i] << " sensitive class " 
				  << layerSense[i];
#endif
  zMinBlock = nArgs["zMinBlock"];
  rMaxFine = nArgs["rMaxFine"];
  waferW = nArgs["waferW"];
  waferGap = nArgs["waferGap"];
  sectors = (int)(nArgs["Sectors"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModuleAlgo: zStart " << zMinBlock 
				<< " rFineCoarse " << rMaxFine << " wafer width "
				<< waferW << " gap among wafers " << waferGap 
				<< " sectors " << sectors;
#endif
  slopeB = vArgs["SlopeBottom"];
  slopeT = vArgs["SlopeTop"];
  zFront = vArgs["ZFront"];
  rMaxFront = vArgs["RMaxFront"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModuleAlgo: Bottom slopes " 
				<< slopeB[0] << ":" << slopeB[1] << " and "
				<< slopeT.size() << " slopes for top";
  for (unsigned int i = 0; i < slopeT.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] Zmin " << zFront[i] 
				  << " Rmax " << rMaxFront[i] << " Slope "
				  << slopeT[i];
#endif
  idNameSpace = DDCurrentNamespace::ns();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModuleAlgo: NameSpace " << idNameSpace;
#endif
}

////////////////////////////////////////////////////////////////////
// DDHGCalModuleAlgo methods...
////////////////////////////////////////////////////////////////////

void DDHGCalModuleAlgo::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalModuleAlgo...";
#endif
  copies.clear();
  constructLayers(parent(), cpv);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << copies.size()<<" different wafer copy numbers";
#endif
  copies.clear();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "<<== End of DDHGCalModuleAlgo construction";
#endif
}

void DDHGCalModuleAlgo::constructLayers(const DDLogicalPart& module,
                                        DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModuleAlgo test: \t\tInside Layers";
#endif
  double zi(zMinBlock);
  int laymin(0);
  const double tol(0.01);
  for (unsigned int i = 0; i < layers.size(); i++) {
    double zo = zi + layerThick[i];
    double routF = rMax(zi);
    int laymax = laymin + layers[i];
    double zz = zi;
    double thickTot(0);
    for (int ly = laymin; ly < laymax; ++ly) {
      int ii = layerType[ly];
      int copy = copyNumber[ii];
      double rinB = (layerSense[ly] == 0) ? (zo * slopeB[0]) : (zo * slopeB[1]);
      zz += (0.5 * thick[ii]);
      thickTot += thick[ii];

      std::string name = "HGCal" + names[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalModuleAlgo test: Layer " << ly 
				    << ":" << ii << " Front " << zi << ", " 
				    << routF << " Back " << zo << ", " << rinB 
				    << " superlayer thickness " << layerThick[i];
#endif
      DDName matName(DDSplit(materials[ii]).first,
                     DDSplit(materials[ii]).second);
      DDMaterial matter(matName);
      DDLogicalPart glog;
      if (layerSense[ly] == 0) {
        double alpha = geant_units::piRadians / sectors;
        double rmax = routF * cos(alpha) - tol;
        std::vector<double> pgonZ, pgonRin, pgonRout;
        pgonZ.emplace_back(-0.5 * thick[ii]);
        pgonZ.emplace_back(0.5 * thick[ii]);
        pgonRin.emplace_back(rinB);
        pgonRin.emplace_back(rinB);
        pgonRout.emplace_back(rmax);
        pgonRout.emplace_back(rmax);
        DDSolid solid = DDSolidFactory::polyhedra(DDName(name, idNameSpace),
                                                  sectors, -alpha, 
						  2*geant_units::piRadians,
                                                  pgonZ, pgonRin, pgonRout);
        glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalModuleAlgo: " << solid.name()
				      << " polyhedra of " << sectors 
				      << " sectors covering "
				      << convertRadToDeg(-alpha) << ":"
				      << (360.0+convertRadToDeg(-alpha))
				      << " with " << pgonZ.size() << " sections";
        for (unsigned int k = 0; k < pgonZ.size(); ++k)
          edm::LogVerbatim("HGCalGeom") << "[" << k << "] z " << pgonZ[k] << " R "
					<< pgonRin[k] << ":" << pgonRout[k];
#endif
      } else {
        DDSolid solid =
            DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5 * thick[ii],
                                 rinB, routF, 0.0, 2*geant_units::piRadians);
        glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalModuleAlgo: " << solid.name()
				      << " Tubs made of " << matName 
				      << " of dimensions " << rinB << ", " 
				      << routF << ", " << 0.5 * thick[ii] 
				      << ", 0.0, 360.0";
#endif
        positionSensitive(glog, rinB, routF, cpv);
      }
      DDTranslation r1(0, 0, zz);
      DDRotation rot;
      cpv.position(glog, module, copy, r1, rot);
      ++copyNumber[ii];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalModuleAlgo test: " << glog.name() << " number "
                << copy << " positioned in " << module.name() << " at " << r1
                << " with " << rot << std::endl;
#endif
      zz += (0.5 * thick[ii]);
    }  // End of loop over layers in a block
    zi = zo;
    laymin = laymax;
    if (fabs(thickTot - layerThick[i]) < 0.00001) {
    } else if (thickTot > layerThick[i]) {
      edm::LogError("HGCalGeom")
          << "Thickness of the partition " << layerThick[i]
          << " is smaller than thickness " << thickTot
          << " of all its components **** ERROR ****\n";
    } else if (thickTot < layerThick[i]) {
      edm::LogWarning("HGCalGeom")
          << "Thickness of the partition " << layerThick[i]
          << " does not match with " << thickTot << " of the components\n";
    }
  }  // End of loop over blocks
}

double DDHGCalModuleAlgo::rMax(double z) {
  double r(0);
#ifdef EDM_ML_DEBUG
  unsigned int ik(0);
#endif
  for (unsigned int k = 0; k < slopeT.size(); ++k) {
    if (z < zFront[k]) break;
    r = rMaxFront[k] + (z - zFront[k]) * slopeT[k];
#ifdef EDM_ML_DEBUG
    ik = k;
#endif
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "rMax : " << z << ":" << ik << ":" << r << std::endl;
#endif
  return r;
}

void DDHGCalModuleAlgo::positionSensitive(DDLogicalPart& glog, double rin,
                                          double rout, DDCompactView& cpv) {
  double ww = (waferW + waferGap);
  double dx = 0.5 * ww;
  double dy = 3.0 * dx * tan(30._deg);
  double rr = 2.0 * dx * tan(30._deg);
  int ncol = (int)(2.0 * rout / ww) + 1;
  int nrow = (int)(rout / (ww * tan(30._deg))) + 1;
  int incm(0), inrm(0), kount(0);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << glog.ddname() << " rout " << rout << " Row "
				<< nrow << " Column " << ncol;
#endif
  for (int nr = -nrow; nr <= nrow; ++nr) {
    int inr = (nr >= 0) ? nr : -nr;
    for (int nc = -ncol; nc <= ncol; ++nc) {
      int inc = (nc >= 0) ? nc : -nc;
      if (inr % 2 == inc % 2) {
        double xpos = nc * dx;
        double ypos = nr * dy;
        std::pair<int, int> corner =
            HGCalGeomTools::waferCorner(xpos, ypos, dx, rr, rin, rout, true);
        if (corner.first == (int)(HGCalParameters::k_CornerSize)) {
          double rpos = std::sqrt(xpos * xpos + ypos * ypos);
          DDTranslation tran(xpos, ypos, 0.0);
          DDRotation rotation;
          int copy = inr * 100 + inc;
          if (nc < 0) copy += 10000;
          if (nr < 0) copy += 100000;
          DDName name =
              (rpos < rMaxFine)
                  ? DDName(DDSplit(wafer[0]).first, DDSplit(wafer[0]).second)
                  : DDName(DDSplit(wafer[1]).first, DDSplit(wafer[1]).second);
          cpv.position(name, glog.ddname(), copy, tran, rotation);
          if (inc > incm) incm = inc;
          if (inr > inrm) inrm = inr;
          kount++;
          if (copies.count(copy) == 0) copies.insert(copy);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalModuleAlgo: " << name << " number " << copy
                    << " positioned in " << glog.ddname() << " at " << tran
                    << " with " << rotation << std::endl;
#endif
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModuleAlgo: # of columns " << incm 
				<< " # of rows " << inrm << " and " << kount 
				<< " wafers for " << glog.ddname();
#endif
}
