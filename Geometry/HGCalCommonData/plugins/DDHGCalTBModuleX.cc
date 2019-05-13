#include <algorithm>
#include <cmath>

#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalTBModuleX.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

DDHGCalTBModuleX::DDHGCalTBModuleX()
    : factor_(0.5 * sqrt(2.0)), tan30deg_(tan(30._deg)) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom")
    << "DDHGCalTBModuleX info: Creating instance";
#endif
}

DDHGCalTBModuleX::~DDHGCalTBModuleX() {}

void DDHGCalTBModuleX::initialize(const DDNumericArguments& nArgs,
                                  const DDVectorArguments& vArgs,
                                  const DDMapArguments&,
                                  const DDStringArguments& sArgs,
                                  const DDStringVectorArguments& vsArgs) {
  wafer_ = vsArgs["WaferName"];
  covers_ = vsArgs["CoverName"];
  genMat_ = sArgs["GeneralMaterial"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") 
    << "DDHGCalTBModuleX: Material " << genMat_ << " with " << wafer_.size()
    << " wafers";
  unsigned int i(0);
  for (auto wafer : wafer_) {
    edm::LogVerbatim("HGCalGeom") << "Wafer[" << i << "] " << wafer;
    ++i;
  }
  edm::LogVerbatim("HGCalGeom") 
    << "DDHGCalTBModuleX: " << covers_.size() << " covers";
  i = 0;
  for (auto cover : covers_) {
    edm::LogVerbatim("HGCalGeom") << "Cover[" << i << "] " << cover;
    ++i;
  }
#endif
  materials_ = vsArgs["MaterialNames"];
  names_ = vsArgs["VolumeNames"];
  layerThick_ = vArgs["Thickness"];
  for (unsigned int k = 0; k < layerThick_.size(); ++k)
    copyNumber_.emplace_back(1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") 
    << "DDHGCalTBModuleX: " << materials_.size() << " types of volumes";
  for (unsigned int i = 0; i < names_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") 
      << "Volume [" << i << "] " << names_[i] << " of thickness " 
      << layerThick_[i] << " filled with " << materials_[i]
      << " first copy number " << copyNumber_[i];
#endif
  inOut_        = nArgs["InOut"];
  blockThick_   = vArgs["BlockThick"];
  layerFrontIn_ = dbl_to_int(vArgs["LayerFrontIn"]);
  layerBackIn_  = dbl_to_int(vArgs["LayerBackIn"]);
  if (inOut_ > 1) {
    layerFrontOut_ = dbl_to_int(vArgs["LayerFrontOut"]);
    layerBackOut_  = dbl_to_int(vArgs["LayerBackOut"]);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom")
    << "DDHGCalTBModuleX: " << blockThick_.size() << " blocks with in/out "
    << inOut_;
  for (unsigned int i = 0; i < blockThick_.size(); ++i) {
    if (inOut_ > 1) 
      edm::LogVerbatim("HGCalGeom")
	<< "Block [" << i << "] of thickness " << blockThick_[i]
	<< " with inner layers " << layerFrontIn_[i] << ":" 
	<< layerBackIn_[i] << " and outer layers " << layerFrontOut_[i]
	<< ":" << layerBackOut_[i];
    else
      edm::LogVerbatim("HGCalGeom") 
	<< "Block [" << i << "] of thickness " << blockThick_[i] 
	<< " with inner layers " << layerFrontIn_[i] << ":" << layerBackIn_[i];
  }
#endif
  layerType_  = dbl_to_int(vArgs["LayerType"]);
  layerSense_ = dbl_to_int(vArgs["LayerSense"]);
  maxModule_  = dbl_to_int(vArgs["MaxModule"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom")
      << "DDHGCalTBModuleX: " << layerType_.size() << " layers";
  for (unsigned int i = 0; i < layerType_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") 
      << "Layer [" << i << "] with material type " << layerType_[i] 
      << " sensitive class " << layerSense_[i] << " and " << maxModule_[i]
      << " maximum row/columns";
#endif
  zMinBlock_ = nArgs["zMinBlock"];
  rMaxFine_  = nArgs["rMaxFine"];
  waferW_    = nArgs["waferW"];
  waferGap_  = nArgs["waferGap"];
  absorbW_   = nArgs["absorberW"];
  absorbH_   = nArgs["absorberH"];
  rMax_      = nArgs["rMax"];
  rMaxB_     = nArgs["rMaxB"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") 
    << "DDHGCalTBModuleX: zStart " << zMinBlock_ << " rFineCoarse " 
    << rMaxFine_ << " wafer width " << waferW_ << " gap among wafers "
    << waferGap_ << " absorber width " << absorbW_ << " absorber height "
    << absorbH_ << " rMax " << rMax_ << ":" << rMaxB_;
#endif
  idNameSpace_ = DDCurrentNamespace::ns();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") 
    << "DDHGCalTBModuleX: NameSpace " << idNameSpace_ << " Parent Name "
    << parent().name().name();
#endif
}

////////////////////////////////////////////////////////////////////
// DDHGCalTBModuleX methods...
////////////////////////////////////////////////////////////////////

void DDHGCalTBModuleX::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalTBModuleX...";
#endif
  copies_.clear();
  constructBlocks(parent(), cpv);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") 
    << copies_.size() << " different wafer copy numbers";
#endif
  copies_.clear();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "<<== End of DDHGCalTBModuleX construction";
#endif
}

void DDHGCalTBModuleX::constructBlocks(const DDLogicalPart& parent,
                                       DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalTBModuleX: Inside constructBlock";
#endif
  double zi(zMinBlock_);
  for (unsigned int i = 0; i < blockThick_.size(); i++) {
    double zo = zi + blockThick_[i];
    std::string name = parent.ddname().name() + "Block" + std::to_string(i);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom")
        << "DDHGCalTBModuleX: Block " << i << ":" << name << " z " << zi << ":"
        << zo << " R " << rMaxB_ << " T " << blockThick_[i];
#endif
    DDName matName(DDSplit(genMat_).first, DDSplit(genMat_).second);
    DDMaterial matter(matName);
    DDSolid solid =
        DDSolidFactory::tubs(DDName(name, idNameSpace_), 0.5 * blockThick_[i],
                             0, rMaxB_, 0.0, 2._pi);
    DDLogicalPart glog = DDLogicalPart(solid.ddname(), matter, solid);
    double zz = zi + 0.5 * blockThick_[i];
    DDTranslation r1(0, 0, zz);
    DDRotation rot;
    cpv.position(glog, parent, i, r1, rot);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom")
        << "DDHGCalTBModuleX: " << glog.name() << " number " << i
        << " positioned in " << parent.name() << " at " << r1 << " with "
        << rot;
#endif
    constructLayers(i, layerFrontIn_[i], layerBackIn_[i], -0.5*blockThick_[i],
                    blockThick_[i], false, glog, cpv);
    if (inOut_ > 1)
      constructLayers(i, layerFrontOut_[i], layerBackOut_[i],
                      -0.5 * blockThick_[i], blockThick_[i], true, glog, cpv);
    zi = zo;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") 
    << "DDHGCalTBModuleX: All blocks are " << "placed in " << zMinBlock_ 
    << ":" << zi;
#endif
}

void DDHGCalTBModuleX::constructLayers(int block, int firstLayer, 
				       int lastLayer, double zFront,
				       double totalWidth, bool ignoreCenter,
                                       const DDLogicalPart& module,
                                       DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") 
    << "DDHGCalTBModuleX: \t\tInside Block " << block << " Layers "
    << firstLayer << ":" << lastLayer << " zFront " << zFront 
    << " thickness " << totalWidth << " ignore Center " << ignoreCenter;
#endif
  double zi(zFront), thickTot(0);
  for (int ly = firstLayer; ly <= lastLayer; ++ly) {
    int ii = layerType_[ly];
    int copy = copyNumber_[ii];
    double zz = zi + (0.5 * layerThick_[ii]);
    double zo = zi + layerThick_[ii];
    thickTot += layerThick_[ii];

    std::string name = "HGCal" + names_[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom")
      << "DDHGCalTBModuleX: " << name << " Layer " << ly << ":" << ii 
      << " Z " << zi << ":" << zo << " Thick " << layerThick_[ii]
      << " Sense " << layerSense_[ly];
#endif
    DDName matName(DDSplit(materials_[ii]).first,
                   DDSplit(materials_[ii]).second);
    DDMaterial matter(matName);
    DDLogicalPart glog;
    if (layerSense_[ly] == 0) {
      DDSolid solid = DDSolidFactory::box(DDName(name, idNameSpace_), absorbW_,
                                          absorbH_, 0.5*layerThick_[ii]);
      glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom")
	<< "DDHGCalTBModuleX: " << solid.name() << " box of dimension " 
	<< absorbW_ << ":" << absorbH_ << ":" << 0.5*layerThick_[ii];
#endif
      DDTranslation r1(0, 0, zz);
      DDRotation rot;
      cpv.position(glog, module, copy, r1, rot);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom")
	<< "DDHGCalTBModuleX: " << glog.name() << " number " << copy 
	<< " positioned in " << module.name() << " at " << r1 << " with "
	<< rot;
#endif
    } else if (layerSense_[ly] > 0) {
      positionSensitive(zz, copy, layerSense_[ly], rMax_, maxModule_[ly],
                        ignoreCenter, name, matter, module, cpv);
    }
    ++copyNumber_[ii];
    zi = zo;
  }  // End of loop over layers in a block

  if (fabs(thickTot - totalWidth) < tolerance_) {
  } else if (thickTot > totalWidth) {
    edm::LogError("HGCalGeom") 
      << "Thickness of the partition " << totalWidth << " is smaller than "
      << thickTot << ": total thickness of all its components in "
      << module.name() << " Layers " << firstLayer << ":" << lastLayer << ":"
      << ignoreCenter << "**** ERROR ****";
  } else if (thickTot < totalWidth) {
    edm::LogWarning("HGCalGeom") 
      << "Thickness of the partition " << totalWidth << " does not match with "
      << thickTot << " of the components in " << module.name() << " Layers "
      << firstLayer << ":" << lastLayer << ":" << ignoreCenter;
  }
}

void DDHGCalTBModuleX::positionSensitive(
    double zpos, int copyIn, int type, double rout, int ncrMax,
    bool ignoreCenter, const std::string& nameIn, const DDMaterial& matter,
    const DDLogicalPart& glog, DDCompactView& cpv) {
  double ww = (waferW_ + waferGap_);
  double dx = 0.5 * ww;
  double dy = 3.0 * dx * tan30deg_;
  double rr = 2.0 * dx * tan30deg_;
  int ncol = (int)(2.0 * rout / ww) + 1;
  int nrow = (int)(rout / (ww * tan30deg_)) + 1;
#ifdef EDM_ML_DEBUG
  int incm(0), inrm(0);
  edm::LogVerbatim("HGCalGeom")
    << glog.ddname() << " Copy " << copyIn << " Type " << type << " rout "
    << rout << " Row " << nrow << " column " << ncol << " ncrMax " << ncrMax
    << " Z " << zpos << " Center " << ignoreCenter << " name " << nameIn 
    << " matter " << matter.name();
  int kount(0);
#endif
  if (ncrMax >= 0) {
    nrow = std::min(nrow, ncrMax);
    ncol = std::min(ncol, ncrMax);
  }
  for (int nr = -nrow; nr <= nrow; ++nr) {
    int inr = std::abs(nr);
    for (int nc = -ncol; nc <= ncol; ++nc) {
      int inc = std::abs(nc);
      if ((inr % 2 == inc % 2) && (!ignoreCenter || nc != 0 || nr != 0)) {
        double xpos = nc * dx;
        double ypos = nr * dy;
        double xc[6], yc[6];
        xc[0] = xpos + dx;
        yc[0] = ypos - 0.5 * rr;
        xc[1] = xpos + dx;
        yc[1] = ypos + 0.5 * rr;
        xc[2] = xpos;
        yc[2] = ypos + rr;
        xc[3] = xpos - dx;
        yc[3] = ypos + 0.5 * rr;
        xc[4] = xpos + dx;
        yc[4] = ypos - 0.5 * rr;
        xc[5] = xpos;
        yc[5] = ypos - rr;
        bool cornerAll(true);
        for (int k = 0; k < 6; ++k) {
          double rpos = std::sqrt(xc[k] * xc[k] + yc[k] * yc[k]);
          if (rpos > rout) cornerAll = false;
        }
        if (cornerAll) {
          double rpos = std::sqrt(xpos * xpos + ypos * ypos);
          DDTranslation tran(xpos, ypos, zpos);
          DDRotation rotation;
          int copy = inr * 100 + inc;
          if (nc < 0) copy += 10000;
          if (nr < 0) copy += 100000;
          DDName name, nameX;
          if (type == 1) {
            nameX =
                DDName(DDSplit(covers_[0]).first, DDSplit(covers_[0]).second);
            std::string name0 = nameIn + "M" + std::to_string(copy);
            DDLogicalPart glog1 =
                DDLogicalPart(DDName(name0, idNameSpace_), matter, nameX);
            cpv.position(glog1.ddname(), glog.ddname(), copyIn, tran, rotation);
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom")
                << "DDHGCalTBModuleX: " << glog1.ddname() << " number "
                << copyIn << " positioned in " << glog.ddname() << " at "
                << tran << " with " << rotation;
#endif
            name = (rpos < rMaxFine_) ? DDName(DDSplit(wafer_[0]).first,
                                               DDSplit(wafer_[0]).second)
                                      : DDName(DDSplit(wafer_[1]).first,
                                               DDSplit(wafer_[1]).second);
            DDTranslation tran1;
            cpv.position(name, glog1.ddname(), copy, tran1, rotation);
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom")
                << "DDHGCalTBModuleX: " << name << " number " << copy
                << " positioned in " << glog1.ddname() << " at " << tran1
                << " with " << rotation;
#endif
            if (copies_.count(copy) == 0 && type == 1) copies_.insert(copy);
          } else {
            name = DDName(DDSplit(covers_[type - 1]).first,
                          DDSplit(covers_[type - 1]).second);
            copy += copyIn * 1000000;
            cpv.position(name, glog.ddname(), copy, tran, rotation);
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom")
                << "DDHGCalTBModuleX: " << name << " number " << copy
                << " positioned in " << glog.ddname() << " at " << tran
                << " with " << rotation;
#endif
          }
#ifdef EDM_ML_DEBUG
          if (inc > incm) incm = inc;
          if (inr > inrm) inrm = inr;
          kount++;
#endif
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom")
      << "DDHGCalTBModuleX: # of columns " << incm << " # of rows " << inrm
      << " and " << kount << " wafers for " << glog.ddname();
#endif
}
