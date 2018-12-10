#include "Geometry/HGCalCommonData/interface/HGCalGeomParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDConstant.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <algorithm>
#include <unordered_set>

//#define EDM_ML_DEBUG

const double tolerance      = 0.001; 

HGCalGeomParameters::HGCalGeomParameters() : sqrt3_(std::sqrt(3.0)) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters::HGCalGeomParameters "
                                << "constructor";
#endif
}

HGCalGeomParameters::~HGCalGeomParameters() { 
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters::destructed!!!";
#endif
}

void HGCalGeomParameters::loadGeometryHexagon(const DDFilteredView& _fv, 
                                              HGCalParameters& php,
                                              const std::string & sdTag1,
                                              const DDCompactView* cpv,
                                              const std::string & sdTag2,
                                              const std::string & sdTag3,
                                              HGCalGeometryMode::WaferMode mode) {
 
  DDFilteredView fv = _fv;
  bool dodet(true);
  std::map<int,HGCalGeomParameters::layerParameters> layers;
  std::vector<HGCalParameters::hgtrform> trforms;
  std::vector<bool>                      trformUse;
  
  while (dodet) {
    const DDSolid & sol  = fv.logicalPart().solid();
    // Layers first
    std::vector<int> copy = fv.copyNumbers();
    int nsiz = (int)(copy.size());
    int lay  = (nsiz > 0) ? copy[nsiz-1] : 0;
    int zp   = (nsiz > 2) ? copy[nsiz-3] : -1;
    if (zp != 1) zp = -1;
    if (lay == 0) {
      throw cms::Exception("DDException") << "Funny layer # " << lay << " zp "
                                          << zp << " in " << nsiz 
                                          << " components";
    } else {
      if (std::find(php.layer_.begin(),php.layer_.end(),lay) == 
          php.layer_.end()) php.layer_.emplace_back(lay);
      auto itr = layers.find(lay);
      if (itr == layers.end()) {
        const DDTubs & tube = static_cast<DDTubs>(sol);
        double rin = HGCalParameters::k_ScaleFromDDD*tube.rIn();
        double rout= HGCalParameters::k_ScaleFromDDD*tube.rOut();
        double zp  = HGCalParameters::k_ScaleFromDDD*fv.translation().Z();
        HGCalGeomParameters::layerParameters laypar(rin,rout,zp);
        layers[lay] = laypar;
      }
      DD3Vector x, y, z;
      fv.rotation().GetComponents( x, y, z ) ;
      const CLHEP::HepRep3x3 rotation ( x.X(), y.X(), z.X(),
                                        x.Y(), y.Y(), z.Y(),
                                        x.Z(), y.Z(), z.Z() );
      const CLHEP::HepRotation hr ( rotation );
      double xx = HGCalParameters::k_ScaleFromDDD*fv.translation().X();
      if (std::abs(xx) < tolerance) xx = 0;
      double yy = HGCalParameters::k_ScaleFromDDD*fv.translation().Y();
      if (std::abs(yy) < tolerance) yy = 0;
      const CLHEP::Hep3Vector h3v ( xx, yy, fv.translation().Z() );
      HGCalParameters::hgtrform mytrf;
      mytrf.zp    = zp;
      mytrf.lay   = lay;
      mytrf.sec   = 0;
      mytrf.subsec= 0;
      mytrf.h3v   = h3v;
      mytrf.hr    = hr;
      trforms.emplace_back(mytrf);
      trformUse.emplace_back(false);
    }
    dodet = fv.next();
  }

  // Then wafers
  // This assumes layers are build starting from 1 (which on 25 Jan 2016, they were)
  // to ensure that new copy numbers are always added
  // to the end of the list.
  std::unordered_map<int32_t,int32_t> copies;
  HGCalParameters::layer_map   copiesInLayers(layers.size()+1);
  std::vector<int32_t>         wafer2copy;
  std::vector<HGCalGeomParameters::cellParameters> wafers;  
  std::string attribute = "Volume";
  DDValue val1(attribute, sdTag2, 0.0);
  DDSpecificsMatchesValueFilter filter1{val1};
  DDFilteredView fv1(*cpv,filter1);
  bool ok = fv1.firstChild();
  if (!ok) {
    edm::LogError("HGCalGeom") << " Attribute " << val1
                               << " not found but needed.";
    throw cms::Exception("DDException") << "Attribute " << val1
                                        << " not found but needed.";
  } else {
    dodet = true;
    std::unordered_set<std::string> names;
    while (dodet) {
      const DDSolid & sol  = fv1.logicalPart().solid();
      const std::string & name = fv1.logicalPart().name().name();
      std::vector<int> copy = fv1.copyNumbers();
      int nsiz  = (int)(copy.size());
      int wafer = (nsiz > 0) ? copy[nsiz-1] : 0;
      int layer = (nsiz > 1) ? copy[nsiz-2] : 0;
      if (nsiz < 2) {
        edm::LogError("HGCalGeom") << "Funny wafer # " << wafer << " in "
                                   << nsiz << " components";
        throw cms::Exception("DDException") << "Funny wafer # " << wafer;
      } else {          
        auto itr = copies.find(wafer);
        auto cpy = copiesInLayers[layer].find(wafer);
        if (itr != copies.end() && cpy == copiesInLayers[layer].end()) {
          copiesInLayers[layer][wafer] = itr->second;
      }
         if (itr == copies.end()) {            
          copies[wafer] = wafer2copy.size();
          copiesInLayers[layer][wafer] = wafer2copy.size();
          double xx = HGCalParameters::k_ScaleFromDDD*fv1.translation().X();
          if (std::abs(xx) < tolerance) xx = 0;
          double yy = HGCalParameters::k_ScaleFromDDD*fv1.translation().Y();
          if (std::abs(yy) < tolerance) yy = 0;
          wafer2copy.emplace_back(wafer);
          GlobalPoint p(xx,yy,HGCalParameters::k_ScaleFromDDD*fv1.translation().Z());
          HGCalGeomParameters::cellParameters cell(false,wafer,p);
          wafers.emplace_back(cell);
          if ( names.count(name) == 0 ) {
            std::vector<double> zv, rv;
            if (mode == HGCalGeometryMode::Polyhedra) {
              const DDPolyhedra & polyhedra = static_cast<DDPolyhedra>(sol);
              zv = polyhedra.zVec();
              rv = polyhedra.rMaxVec();
            } else {
              const DDExtrudedPolygon & polygon = static_cast<DDExtrudedPolygon>(sol);
              zv = polygon.zVec();
              rv = polygon.xVec();
            }
            php.waferR_    = rv[0]/std::cos(30.0*CLHEP::deg);
            php.waferSize_ = HGCalParameters::k_ScaleFromDDD * rv[0];
            double dz      = 0.5*(zv[1]-zv[0]);
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom") << "Mode " << mode << " R " 
                                          << php.waferSize_ << ":"
                                          << php.waferR_ << " z " << dz;
#endif
            HGCalParameters::hgtrap mytr;
            mytr.lay = 1;           mytr.bl = php.waferR_; 
            mytr.tl = php.waferR_;  mytr.h = php.waferR_; 
            mytr.dz = dz;           mytr.alpha = 0.0;
            mytr.cellSize = waferSize_;
            php.fillModule(mytr,false);
            names.insert(name);
          }
        }
      }
      dodet = fv1.next();
    }
  }
  
  // Finally the cells
  std::map<int,int>         wafertype;
  std::map<int,HGCalGeomParameters::cellParameters> cellsf, cellsc;
  DDValue val2(attribute, sdTag3, 0.0);
  DDSpecificsMatchesValueFilter filter2{val2};
  DDFilteredView fv2(*cpv,filter2);
  ok = fv2.firstChild();
  if (!ok) {
    edm::LogError("HGCalGeom") << " Attribute " << val2
                               << " not found but needed.";
    throw cms::Exception("DDException") << "Attribute " << val2
                                        << " not found but needed.";
  } else {
    dodet = true;
    while (dodet) {
      const DDSolid & sol  = fv2.logicalPart().solid();
      const std::string & name = sol.name().name();
      std::vector<int> copy = fv2.copyNumbers();
      int nsiz = (int)(copy.size());
      int cellx= (nsiz > 0) ? copy[nsiz-1] : 0;
      int wafer= (nsiz > 1) ? copy[nsiz-2] : 0;
      int cell = cellx%1000;
      int type = cellx/1000;
      if (type != 1 && type != 2) {
        edm::LogError("HGCalGeom") << "Funny cell # " << cell << " type " 
                                   << type << " in " << nsiz << " components";
        throw cms::Exception("DDException") << "Funny cell # " << cell;
      } else {
        auto ktr = wafertype.find(wafer);
        if (ktr == wafertype.end()) wafertype[wafer] = type;
        bool newc(false);
        std::map<int,HGCalGeomParameters::cellParameters>::iterator itr;
        double cellsize = php.cellSize_[0];
        if (type == 1) {
          itr = cellsf.find(cell);
          newc= (itr == cellsf.end());
        } else {
          itr = cellsc.find(cell);
          newc= (itr == cellsc.end());
          cellsize = php.cellSize_[1];
        }
        if (newc) {
          bool half = (name.find("Half") != std::string::npos);
          double xx = HGCalParameters::k_ScaleFromDDD*fv2.translation().X();
          double yy = HGCalParameters::k_ScaleFromDDD*fv2.translation().Y();
          if (half) {
            math::XYZPointD p1(-2.0*cellsize/9.0,0,0);
            math::XYZPointD p2 = fv2.rotation()(p1);
            xx += (HGCalParameters::k_ScaleFromDDD*(p2.X()));
            yy += (HGCalParameters::k_ScaleFromDDD*(p2.Y()));
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom") << "Type " << type << " Cell " 
                                          << cellx << " local " << xx << ":" 
                                          << yy  << " new " << p1 << ":"<< p2;
#endif
          }
          HGCalGeomParameters::cellParameters cp(half,wafer,GlobalPoint(xx,yy,0));
          if (type == 1) {
            cellsf[cell] = cp;
          } else {
            cellsc[cell] = cp;
          }
        }
      }
      dodet = fv2.next();
    }
  }
  
  if (((cellsf.size()+cellsc.size())==0) || (wafers.empty()) || 
      (layers.empty())) {
    edm::LogError("HGCalGeom") << "HGCalGeomParameters : number of cells "
                               << cellsf.size() << ":" << cellsc.size()
                               << " wafers " << wafers.size() << " layers "
                               << layers.size() << " illegal";
    throw cms::Exception("DDException")
      << "HGCalGeomParameters: mismatch between geometry and specpar: cells "
      << cellsf.size() << ":" << cellsc.size() << " wafers " << wafers.size()
      << " layers " << layers.size();
  }

  for (unsigned int i=0; i<layers.size(); ++i) {
    for (auto & layer : layers) {
      if (layer.first == (int)(i+php.firstLayer_)) {
        php.layerIndex_.emplace_back(i);
        php.rMinLayHex_.emplace_back(layer.second.rmin);
        php.rMaxLayHex_.emplace_back(layer.second.rmax);
        php.zLayerHex_.emplace_back(layer.second.zpos);
        break;
      }
    }
  }
  for (unsigned int i=0; i<php.layer_.size(); ++i) {
    for (unsigned int i1=0; i1<trforms.size(); ++i1) {
      if (!trformUse[i1] && php.layerGroup_[trforms[i1].lay-1] == 
          (int)(i+1)) {
        trforms[i1].h3v *= HGCalParameters::k_ScaleFromDDD;
        trforms[i1].lay  = (i+1);
        trformUse[i1]    = true;
        php.fillTrForm(trforms[i1]);
        int nz(1);
        for (unsigned int i2=i1+1; i2<trforms.size(); ++i2) {
          if (!trformUse[i2] && trforms[i2].zp ==  trforms[i1].zp &&
              php.layerGroup_[trforms[i2].lay-1] == (int)(i+1)) {
            php.addTrForm(HGCalParameters::k_ScaleFromDDD*trforms[i2].h3v);
            nz++;
            trformUse[i2] = true;
          }
        }
        if (nz > 0) {
          php.scaleTrForm(double(1.0/nz));
        }
      }
    }
  }

  double rmin = HGCalParameters::k_ScaleFromDDD*php.waferR_;
  for (unsigned i = 0; i < wafer2copy.size(); ++i ) {
    php.waferCopy_.emplace_back(wafer2copy[i]);
    php.waferPosX_.emplace_back(wafers[i].xyz.x());
    php.waferPosY_.emplace_back(wafers[i].xyz.y());
    auto ktr = wafertype.find(wafer2copy[i]);
    int typet = (ktr == wafertype.end()) ? 0 : (ktr->second);
    php.waferTypeT_.emplace_back(typet);
    double r = wafers[i].xyz.perp();
    int    type(3);
    for (int k=1; k<4; ++k) {
      if ((r+rmin)<=php.boundR_[k]) {
       type = k; break;
      }
    }
    php.waferTypeL_.emplace_back(type);
  }
  php.copiesInLayers_ = copiesInLayers;
  php.nSectors_ = (int)(php.waferCopy_.size());

  std::vector<HGCalGeomParameters::cellParameters>::const_iterator itrf = wafers.end();
  for (unsigned int i=0; i<cellsf.size(); ++i) {
    auto itr = cellsf.find(i);
    if (itr == cellsf.end()) {
      edm::LogError("HGCalGeom") << "HGCalGeomParameters: missing info for"
                                 << " fine cell number " << i;
      throw cms::Exception("DDException")
        << "HGCalGeomParameters: missing info for fine cell number " << i;
    } else {
      double xx = (itr->second).xyz.x();
      double yy = (itr->second).xyz.y();
      int    waf= (itr->second).wafer;
      std::pair<double,double> xy = cellPosition(wafers,itrf,waf,xx,yy);
      php.cellFineX_.emplace_back(xy.first);
      php.cellFineY_.emplace_back(xy.second);
      php.cellFineHalf_.emplace_back((itr->second).half);      
    }
  }
  itrf = wafers.end();
  for (unsigned int i=0; i<cellsc.size(); ++i) {
    auto itr = cellsc.find(i);
    if (itr == cellsc.end()) {
      edm::LogError("HGCalGeom") << "HGCalGeomParameters: missing info for"
                                 << " coarse cell number " << i;
      throw cms::Exception("DDException")
        << "HGCalGeomParameters: missing info for coarse cell number " << i;
    } else {
      double xx = (itr->second).xyz.x();
      double yy = (itr->second).xyz.y();
      int    waf= (itr->second).wafer;
      std::pair<double,double> xy = cellPosition(wafers,itrf,waf,xx,yy);
      php.cellCoarseX_.emplace_back(xy.first);
      php.cellCoarseY_.emplace_back(xy.second);
      php.cellCoarseHalf_.emplace_back((itr->second).half);   
    }
  }
  int depth(0);
  for (unsigned int i=0; i<php.layerGroup_.size(); ++i) {
    bool first(true);
    for (unsigned int k=0; k<php.layerGroup_.size(); ++k) {
      if (php.layerGroup_[k] == (int)(i+1)) {
        if (first) {
          php.depth_.emplace_back(i+1);
          php.depthIndex_.emplace_back(depth);
          php.depthLayerF_.emplace_back(k);
          ++depth;
          first = false;
        }
      }
    }
  }
  HGCalParameters::hgtrap mytr = php.getModule(0, false);
  mytr.bl       *= HGCalParameters::k_ScaleFromDDD;
  mytr.tl       *= HGCalParameters::k_ScaleFromDDD;
  mytr.h        *= HGCalParameters::k_ScaleFromDDD;
  mytr.dz       *= HGCalParameters::k_ScaleFromDDD;
  mytr.cellSize *= HGCalParameters::k_ScaleFromDDD;
  double dz  = mytr.dz;
  php.fillModule(mytr, true);
  mytr.dz = 2*dz;
  php.fillModule(mytr, true);
  mytr.dz = 3*dz;
  php.fillModule(mytr, true);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters finds " 
                                << php.zLayerHex_.size() << " layers";
  for (unsigned int i=0; i<php.zLayerHex_.size(); ++i) {
    int k = php.layerIndex_[i];
    edm::LogVerbatim("HGCalGeom") << "Layer[" << i << ":" << k << ":" 
                                  << php.layer_[k] << "] with r = " 
                                  << php.rMinLayHex_[i] << ":" 
                                  << php.rMaxLayHex_[i] << " at z = "  
                                  << php.zLayerHex_[i];
  }
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters has " 
                                << php.depthIndex_.size() << " depths";
  for (unsigned int i=0; i<php.depthIndex_.size(); ++i) {
    int k = php.depthIndex_[i];
    edm::LogVerbatim("HGCalGeom") << "Reco Layer[" << i << ":" << k  
                                  << "]  First Layer " << php.depthLayerF_[i]
                                  << " Depth " << php.depth_[k];
  }
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters finds " 
                                << php.nSectors_ << " wafers";
  for (unsigned int i=0; i<php.waferCopy_.size(); ++i) 
    edm::LogVerbatim("HGCalGeom") << "Wafer[" << i << ": " <<php.waferCopy_[i]
                                  << "] type " << php.waferTypeL_[i] << ":" 
                                  << php.waferTypeT_[i] << " at (" 
                                  << php.waferPosX_[i] << "," 
                                  << php.waferPosY_[i] << ",0)";
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: wafer radius "  
                                << php.waferR_ << " and dimensions of the "
                                << "wafers:";
  edm::LogVerbatim("HGCalGeom") << "Sim[0] " << php.moduleLayS_[0] << " dx "
                                << php.moduleBlS_[0] << ":" 
                                << php.moduleTlS_[0] << " dy "
                                << php.moduleHS_[0] << " dz "
                                << php.moduleDzS_[0] << " alpha " 
                                << php.moduleAlphaS_[0];
  for (unsigned int k=0; k<php.moduleLayR_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "Rec[" << k << "] " << php.moduleLayR_[k]
                                  << " dx " << php.moduleBlR_[k] << ":" 
                                  << php.moduleTlR_[k] << " dy " 
                                  << php.moduleHR_[k]  << " dz " 
                                  << php.moduleDzR_[k] << " alpha "
                                  << php.moduleAlphaR_[k];
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters finds " 
                                << php.cellFineX_.size() 
                               << " fine cells in a  wafer";
  for (unsigned int i=0; i<php.cellFineX_.size(); ++i) 
    edm::LogVerbatim("HGCalGeom") << "Fine Cell[" << i << "] at (" 
                                  << php.cellFineX_[i] << ","
                                  << php.cellFineY_[i] << ",0)";
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters finds "
                                << php.cellCoarseX_.size() 
                                << " coarse cells in a wafer";
  for (unsigned int i=0; i<php.cellCoarseX_.size(); ++i) 
    edm::LogVerbatim("HGCalGeom") << "Coarse Cell[" << i << "] at (" 
                                  << php.cellCoarseX_[i]
                                  << "," << php.cellCoarseY_[i] << ",0)";
  edm::LogVerbatim("HGCalGeom") << "Obtained " << php.trformIndex_.size() 
                                << " transformation matrices";
  for (unsigned int k=0; k<php.trformIndex_.size(); ++k) {
    edm::LogVerbatim("HGCalGeom") << "Matrix[" << k << "] (" << std::hex 
                                  << php.trformIndex_[k]
                                  << std::dec << ") Translation (" 
                                  << php.trformTranX_[k] << ", " 
                                  << php.trformTranY_[k] << ", " 
                                  << php.trformTranZ_[k] << " Rotation ("
                                  << php.trformRotXX_[k] << ", "
                                  << php.trformRotYX_[k] << ", " 
                                  << php.trformRotZX_[k] << ", "
                                  << php.trformRotXY_[k] << ", " 
                                  << php.trformRotYY_[k] << ", "
                                  << php.trformRotZY_[k] << ", " 
                                  << php.trformRotXZ_[k] << ", "
                                  << php.trformRotYZ_[k] << ", " 
                                  << php.trformRotZZ_[k] << ")";
  }
  edm::LogVerbatim("HGCalGeom") << "Dump copiesInLayers for " 
                                << php.copiesInLayers_.size()
                                << " layers";
  for (unsigned int k=0; k<php.copiesInLayers_.size(); ++k) {
    const auto& theModules = php.copiesInLayers_[k];
    edm::LogVerbatim("HGCalGeom") << "Layer " << k << ":" <<theModules.size();
    int k2(0);
    for (std::unordered_map<int, int>::const_iterator itr=theModules.begin();
         itr != theModules.end(); ++itr, ++k2) {
      edm::LogVerbatim("HGCalGeom") << "[" << k2 << "] " << itr->first << ":" 
                                    << itr->second;
    }
  }
#endif
}

void HGCalGeomParameters::loadGeometryHexagon8(const DDFilteredView& _fv, 
                 HGCalParameters& php,
                 int firstLayer) {
 
  DDFilteredView fv = _fv;
  bool dodet(true);
  std::map<int,HGCalGeomParameters::layerParameters>     layers;
  std::map<std::pair<int,int>,HGCalParameters::hgtrform> trforms;
  int levelTop = 3+std::max(php.levelT_[0],php.levelT_[1]);
  while (dodet) {
    const DDSolid & sol  = fv.logicalPart().solid();
    // Layers first
    std::vector<int> copy = fv.copyNumbers();
    int nsiz = (int)(copy.size());
    int lay  = (nsiz > levelTop) ? copy[nsiz-4] : copy[nsiz-1];
    int zside= (nsiz > php.levelZSide_) ? copy[php.levelZSide_] : -1;
    if (zside != 1) zside = -1;
    if (lay == 0) {
      edm::LogError("HGCalGeom") << "Funny layer # " << lay << " zp "
                                 << zside << " in " << nsiz << " components";
      throw cms::Exception("DDException") << "Funny layer # " << lay;
    } else {
      if (std::find(php.layer_.begin(),php.layer_.end(),lay) == 
          php.layer_.end()) php.layer_.emplace_back(lay);
      auto itr = layers.find(lay);
      if (itr == layers.end()) {
        const DDTubs & tube = static_cast<DDTubs>(sol);
        double rin = HGCalParameters::k_ScaleFromDDD*tube.rIn();
        double rout= HGCalParameters::k_ScaleFromDDD*tube.rOut();
        double zp  = HGCalParameters::k_ScaleFromDDD*fv.translation().Z();
        HGCalGeomParameters::layerParameters laypar(rin,rout,zp);
        layers[lay] = laypar;
      }
      if (trforms.find(std::make_pair(lay,zside)) == trforms.end()) {
        DD3Vector x, y, z;
        fv.rotation().GetComponents( x, y, z ) ;
        const CLHEP::HepRep3x3 rotation ( x.X(), y.X(), z.X(),
                                          x.Y(), y.Y(), z.Y(),
                                          x.Z(), y.Z(), z.Z() );
        const CLHEP::HepRotation hr ( rotation );
        double xx = ((std::abs(fv.translation().X()) < tolerance) ? 0 :
                     fv.translation().X());
        double yy = ((std::abs(fv.translation().Y()) < tolerance) ? 0 :
                     fv.translation().Y());
        const CLHEP::Hep3Vector h3v (xx, yy, fv.translation().Z());
        HGCalParameters::hgtrform mytrf;
        mytrf.zp    = zside;
        mytrf.lay   = lay;
        mytrf.sec   = 0;
        mytrf.subsec= 0;
        mytrf.h3v   = h3v;
        mytrf.hr    = hr;
        trforms[std::make_pair(lay,zside)] = mytrf;
      }
    }
    dodet = fv.next();
  }

  double rmin(0), rmax(0);
  for (unsigned int i=0; i<layers.size(); ++i) {
    for (auto & layer : layers) {
      if (layer.first == (int)(i+firstLayer)) {
        php.layerIndex_.emplace_back(i);
        php.rMinLayHex_.emplace_back(layer.second.rmin);
        php.rMaxLayHex_.emplace_back(layer.second.rmax);
        php.zLayerHex_.emplace_back(layer.second.zpos);
        if (i == 0) {
          rmin = layer.second.rmin; rmax = layer.second.rmax;
        } else {
          if (rmin > layer.second.rmin) rmin = layer.second.rmin;
          if (rmax < layer.second.rmax) rmax = layer.second.rmax;
        }
        break;
      }
    }
  }
  php.rLimit_.emplace_back(rmin);
  php.rLimit_.emplace_back(rmax);
  php.depth_      = php.layer_;
  php.depthIndex_ = php.layerIndex_;
  php.depthLayerF_= php.layerIndex_;

  for (unsigned int i=0; i<php.layer_.size(); ++i) {
    for (auto & trform : trforms) {
      if (trform.first.first == (int)(i+firstLayer)) {
        trform.second.h3v *= HGCalParameters::k_ScaleFromDDD;
        php.fillTrForm(trform.second);
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: Minimum/maximum R " 
                                << php.rLimit_[0] << ":" << php.rLimit_[1];
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters finds " 
                                << php.zLayerHex_.size() << " layers";
  for (unsigned int i=0; i<php.zLayerHex_.size(); ++i) {
    int k = php.layerIndex_[i];
    edm::LogVerbatim("HGCalGeom") << "Layer[" << i << ":" << k << ":" 
                                  << php.layer_[k] << "] with r = " 
                                  << php.rMinLayHex_[i] << ":" 
                                  << php.rMaxLayHex_[i] << " at z = "  
                                  << php.zLayerHex_[i];
  }
  edm::LogVerbatim("HGCalGeom") << "Obtained " << php.trformIndex_.size() 
                                << " transformation matrices";
  for (unsigned int k=0; k<php.trformIndex_.size(); ++k) {
    edm::LogVerbatim("HGCalGeom") << "Matrix[" << k << "] (" << std::hex 
                                  << php.trformIndex_[k]
                                  << std::dec << ") Translation (" 
                                  << php.trformTranX_[k] << ", " 
                                  << php.trformTranY_[k] << ", " 
                                  << php.trformTranZ_[k] << " Rotation ("
                                  << php.trformRotXX_[k] << ", "
                                  << php.trformRotYX_[k] << ", " 
                                  << php.trformRotZX_[k] << ", "
                                  << php.trformRotXY_[k] << ", " 
                                  << php.trformRotYY_[k] << ", "
                                  << php.trformRotZY_[k] << ", " 
                                  << php.trformRotXZ_[k] << ", "
                                  << php.trformRotYZ_[k] << ", " 
                                  << php.trformRotZZ_[k] << ")";
  }
#endif
}

void HGCalGeomParameters::loadSpecParsHexagon(const DDFilteredView& fv,
                                              HGCalParameters& php,
                                              const DDCompactView* cpv,
                                              const std::string & sdTag1,
                                              const std::string & sdTag2) {

  DDsvalues_type sv(fv.mergedSpecifics());
  php.boundR_ = getDDDArray("RadiusBound",sv,4);
  std::for_each(php.boundR_.begin(), php.boundR_.end(), [](double &n){ n*=HGCalParameters::k_ScaleFromDDD; });
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: wafer radius ranges"
                                << " for cell grouping " << php.boundR_[0] 
                                << ":" << php.boundR_[1] << ":"
                                << php.boundR_[2] << ":" << php.boundR_[3];
#endif
  php.rLimit_ = getDDDArray("RadiusLimits",sv,2);
  std::for_each(php.rLimit_.begin(), php.rLimit_.end(), [](double &n){ n*=HGCalParameters::k_ScaleFromDDD; });
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: Minimum/maximum R " 
                                << php.rLimit_[0] << ":" << php.rLimit_[1];
#endif
  php.levelT_ = dbl_to_int(getDDDArray("LevelTop",sv,0));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: LevelTop " 
                                << php.levelT_[0];
#endif

  //Grouping of layers
  php.layerGroup_  = dbl_to_int(getDDDArray("GroupingZFine",sv,0));
  php.layerGroupM_ = dbl_to_int(getDDDArray("GroupingZMid",sv,0));
  php.layerGroupO_ = dbl_to_int(getDDDArray("GroupingZOut",sv,0));
  php.slopeMin_    = getDDDArray("Slope",sv,1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: minimum slope " 
                                << php.slopeMin_[0] << " and layer groupings "
                                << "for the 3 ranges:";
  for (unsigned int k=0; k<php.layerGroup_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << php.layerGroup_[k] 
                                  << ":" << php.layerGroupM_[k] << ":"  
                                  << php.layerGroupO_[k];
#endif

  //Wafer size
  std::string attribute = "Volume";
  DDSpecificsMatchesValueFilter filter1{DDValue(attribute, sdTag1, 0.0)};
  DDFilteredView fv1(*cpv,filter1);
  if (fv1.firstChild()) {
    DDsvalues_type sv(fv1.mergedSpecifics());
    const auto & dummy = getDDDArray("WaferSize",sv,0);
    waferSize_ = dummy[0];
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: Wafer Size: " 
        << waferSize_;
#endif

  //Cell size
  DDSpecificsMatchesValueFilter filter2{DDValue(attribute, sdTag2, 0.0)};
  DDFilteredView fv2(*cpv,filter2);
  if (fv2.firstChild()) {
    DDsvalues_type sv(fv2.mergedSpecifics());
    php.cellSize_ = getDDDArray("CellSize",sv,0);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: " 
                                << php.cellSize_.size() << " cells of sizes:";
  for (unsigned int k=0; k<php.cellSize_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << " [" << k << "] " << php.cellSize_[k];
#endif

}

void HGCalGeomParameters::loadSpecParsHexagon8(const DDFilteredView& fv,
                                               HGCalParameters& php) {

  DDsvalues_type sv(fv.mergedSpecifics());
  php.cellThickness_ = getDDDArray("CellThickness",sv,3);
  std::for_each(php.cellThickness_.begin(), php.cellThickness_.end(), [](double &n){ n*=HGCalParameters::k_ScaleFromDDD; });
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: cell Thickness "
                                << php.cellThickness_[0] << ":" 
                                << php.cellThickness_[1] << ":" 
                                << php.cellThickness_[2];
#endif
  php.radius100to200_ = getDDDArray("Radius100to200",sv,5);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: Polynomial "
                                << "parameters for 120 to 200 micron "
                                << "transition "  << php.radius100to200_[0] 
                                << ":" << php.radius100to200_[1] << ":"
                                << php.radius100to200_[2] << ":" 
                                << php.radius100to200_[3] << ":"
                                << php.radius100to200_[4];
#endif
  php.radius200to300_ = getDDDArray("Radius200to300",sv,5);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: Polynomial "
                                << "parameters for 200 to 300 micron "
                                << "transition " << php.radius200to300_[0]
                                << ":" << php.radius200to300_[1] << ":"
                                << php.radius200to300_[2] << ":" 
                                << php.radius200to300_[3] << ":"
                                << php.radius200to300_[4];
#endif
  const auto & dummy = getDDDArray("RadiusCuts",sv,4);
  php.choiceType_ = (int)(dummy[0]);
  php.nCornerCut_ = (int)(dummy[1]);
  php.fracAreaMin_= dummy[2];
  php.zMinForRad_ = HGCalParameters::k_ScaleFromDDD*dummy[3];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: Parameters for the"
                                << " transition " << php.choiceType_ << ":"
                                << php.nCornerCut_ << ":" << php.fracAreaMin_
                                << ":" << php.zMinForRad_;
#endif
  php.radiusMixBoundary_ = DDVectorGetter::get("RadiusMixBoundary");
  std::for_each(php.radiusMixBoundary_.begin(), php.radiusMixBoundary_.end(), [](double &n){ n*=HGCalParameters::k_ScaleFromDDD; });
#ifdef EDM_ML_DEBUG
  for (unsigned int k = 0; k < php.radiusMixBoundary_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "HGCalParameters: Mix[" << k << "] R = "
                                  << php.radiusMixBoundary_[k];
#endif
  php.slopeMin_  = getDDDArray("SlopeBottom",sv,0);
  php.zFrontMin_ = getDDDArray("ZFrontBottom",sv,0);
  std::for_each(php.zFrontMin_.begin(), php.zFrontMin_.end(), [](double &n){ n*=HGCalParameters::k_ScaleFromDDD; });
  php.rMinFront_ = getDDDArray("RMinFront",sv,0);
  std::for_each(php.rMinFront_.begin(), php.rMinFront_.end(), [](double &n){ n*=HGCalParameters::k_ScaleFromDDD; });
#ifdef EDM_ML_DEBUG
  for (unsigned int k = 0; k < php.zFrontMin_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "HGCalParameters: Boundary[" << k 
                                  << "] Bottom Z = " << php.zFrontMin_[k] 
                                  << " Slope = " << php.slopeMin_[k] 
                                  << " rMax = " << php.rMinFront_[k];
#endif
  php.slopeTop_  = getDDDArray("SlopeTop",sv,0);
  php.zFrontTop_ = getDDDArray("ZFrontTop",sv,0);
  std::for_each(php.zFrontTop_.begin(), php.zFrontTop_.end(), [](double &n){ n*=HGCalParameters::k_ScaleFromDDD; });
  php.rMaxFront_ = getDDDArray("RMaxFront",sv,0);
  std::for_each(php.rMaxFront_.begin(), php.rMaxFront_.end(), [](double &n){ n*=HGCalParameters::k_ScaleFromDDD; });
#ifdef EDM_ML_DEBUG
  for (unsigned int k = 0; k < php.zFrontTop_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "HGCalParameters: Boundary[" << k 
                                  << "] Top Z = " << php.zFrontTop_[k] 
                                  << " Slope = " << php.slopeTop_[k] 
                                  << " rMax = " << php.rMaxFront_[k];
#endif
  php.zRanges_   = DDVectorGetter::get("ZRanges");
  std::for_each(php.zRanges_.begin(), php.zRanges_.end(), [](double &n){ n*=HGCalParameters::k_ScaleFromDDD; });
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalParameters: Z-Boundary " 
                                << php.zRanges_[0] << ":" << php.zRanges_[1]
                                << ":" << php.zRanges_[2] << ":"
                                << php.zRanges_[3];
#endif
}

void HGCalGeomParameters::loadSpecParsTrapezoid(const DDFilteredView& fv,
                                                HGCalParameters& php) {

  DDsvalues_type sv(fv.mergedSpecifics());
  php.radiusMixBoundary_ = DDVectorGetter::get("RadiusMixBoundary");
  std::for_each(php.radiusMixBoundary_.begin(), php.radiusMixBoundary_.end(), [](double &n){ n*=HGCalParameters::k_ScaleFromDDD; });
  php.nPhiBinBH_    = dbl_to_int(getDDDArray("NPhiBinBH",sv,0));
  php.layerFrontBH_ = dbl_to_int(getDDDArray("LayerFrontBH",sv,0));
  php.rMinLayerBH_  = getDDDArray("RMinLayerBH",sv,0);
  std::for_each(php.rMinLayerBH_.begin(), php.rMinLayerBH_.end(), [](double &n){ n*=HGCalParameters::k_ScaleFromDDD; });
  php.nCellsFine_   = php.nPhiBinBH_[0];
  php.nCellsCoarse_ = php.nPhiBinBH_[1];
  php.cellSize_.emplace_back(2.0*M_PI/php.nCellsFine_);
  php.cellSize_.emplace_back(2.0*M_PI/php.nCellsCoarse_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters:nCells "
                                << php.nCellsFine_ << ":" << php.nCellsCoarse_
                                << " cellSize: " << php.cellSize_[0] << ":"
                                << php.cellSize_[1];
  for (unsigned int k=0; k<php.layerFrontBH_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: Type[" << k 
                                  << "] Front Layer = " << php.layerFrontBH_[k]
                                  << " rMin = " << php.rMinLayerBH_[k];
  for (unsigned int k = 0; k < php.radiusMixBoundary_.size(); ++k) {
    edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: Mix[" << k 
                                  << "] R = " << php.radiusMixBoundary_[k] 
                                  << " Nphi = " << php.scintCells(k+php.firstLayer_)
                                  << " dPhi = " << php.scintCellSize(k+php.firstLayer_);
  }
#endif
  php.slopeMin_  = getDDDArray("SlopeBottom",sv,0);
  php.zFrontMin_ = getDDDArray("ZFrontBottom",sv,0);
  std::for_each(php.zFrontMin_.begin(), php.zFrontMin_.end(), [](double &n){ n*=HGCalParameters::k_ScaleFromDDD; });
  php.rMinFront_ = getDDDArray("RMinFront",sv,0);
  std::for_each(php.rMinFront_.begin(), php.rMinFront_.end(), [](double &n){ n*=HGCalParameters::k_ScaleFromDDD; });
#ifdef EDM_ML_DEBUG
  for (unsigned int k = 0; k < php.zFrontMin_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "HGCalParameters: Boundary[" << k 
                                  << "] Bottom Z = " << php.zFrontMin_[k] 
                                  << " Slope = " << php.slopeMin_[k] 
                                  << " rMax = " << php.rMinFront_[k];
#endif
  php.slopeTop_  = getDDDArray("SlopeTop",sv,0);
  php.zFrontTop_ = getDDDArray("ZFrontTop",sv,0);
  std::for_each(php.zFrontTop_.begin(), php.zFrontTop_.end(), [](double &n){ n*=HGCalParameters::k_ScaleFromDDD; });
  php.rMaxFront_ = getDDDArray("RMaxFront",sv,0);
  std::for_each(php.rMaxFront_.begin(), php.rMaxFront_.end(), [](double &n){ n*=HGCalParameters::k_ScaleFromDDD; });
#ifdef EDM_ML_DEBUG
  for (unsigned int k = 0; k < php.zFrontTop_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "HGCalParameters: Boundary[" << k 
                                  << "] Top Z = " << php.zFrontTop_[k] 
                                  << " Slope = " << php.slopeTop_[k] 
                                  << " rMax = "  << php.rMaxFront_[k];
#endif
  php.zRanges_   = DDVectorGetter::get("ZRanges");
  std::for_each(php.zRanges_.begin(), php.zRanges_.end(), [](double &n){ n*=HGCalParameters::k_ScaleFromDDD; });
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalParameters: Z-Boundary " 
                                << php.zRanges_[0] << ":" << php.zRanges_[1]
                                << ":" << php.zRanges_[2] << ":"
                                << php.zRanges_[3];
#endif
}

void HGCalGeomParameters::loadWaferHexagon(HGCalParameters& php) {

  double waferW(HGCalParameters::k_ScaleFromDDD*waferSize_), rmin(HGCalParameters::k_ScaleFromDDD*php.waferR_);
  double rin(php.rLimit_[0]), rout(php.rLimit_[1]), rMaxFine(php.boundR_[1]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Input waferWidth " << waferW << ":" 
        << rmin << " R Limits: "  << rin << ":" 
        << rout << " Fine " << rMaxFine;
#endif
  // Clear the vectors
  php.waferCopy_.clear();
  php.waferTypeL_.clear();
  php.waferTypeT_.clear();
  php.waferPosX_.clear();
  php.waferPosY_.clear();
  double dx   = 0.5*waferW;
  double dy   = 3.0*dx*tan(30.0*CLHEP::deg);
  double rr   = 2.0*dx*tan(30.0*CLHEP::deg);
  int    ncol = (int)(2.0*rout/waferW) + 1;
  int    nrow = (int)(rout/(waferW*tan(30.0*CLHEP::deg))) + 1;
  int    ns2   = (2*ncol+1)*(2*nrow+1)*php.layer_.size();
  int    incm(0), inrm(0), kount(0), ntot(0);
  HGCalParameters::layer_map copiesInLayers(php.layer_.size()+1);
  HGCalParameters::waferT_map waferTypes(ns2+1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Row " << nrow << " Column " << ncol;
#endif
  for (int nr=-nrow; nr <= nrow; ++nr) {
    int inr = (nr >= 0) ? nr : -nr;
    for (int nc=-ncol; nc <= ncol; ++nc) {
      int inc = (nc >= 0) ? nc : -nc;
      if (inr%2 == inc%2) {
        double xpos = nc*dx;
        double ypos = nr*dy;
        std::pair<int,int> corner = 
          HGCalGeomTools::waferCorner(xpos, ypos, dx, rr, rin, rout, true);
        double rpos = std::sqrt(xpos*xpos+ypos*ypos);
        int typet = (rpos < rMaxFine) ? 1 : 2;
        int typel(3);
        for (int k=1; k<4; ++k) {
          if ((rpos+rmin)<=php.boundR_[k]) {
            typel = k; break;
          }
        }
        ++ntot;
        if (corner.first > 0) {
          int copy = inr*100 + inc;
          if (nc < 0) copy += 10000;
          if (nr < 0) copy += 100000;
          if (inc > incm) incm = inc;
          if (inr > inrm) inrm = inr;
          kount++;
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << kount << ":" << ntot << " Copy " 
                                        << copy << " Type " << typel << ":"
                                        << typet << " Location " << corner.first
                                        << " Position "  << xpos << ":" << ypos
                                        << " Layers " << php.layer_.size();
#endif
          php.waferCopy_.emplace_back(copy);
          php.waferTypeL_.emplace_back(typel);
          php.waferTypeT_.emplace_back(typet);
          php.waferPosX_.emplace_back(xpos);
          php.waferPosY_.emplace_back(ypos);
          for (unsigned int il=0; il<php.layer_.size(); ++il) {
            std::pair<int,int> corner = 
              HGCalGeomTools::waferCorner(xpos, ypos, dx, rr, 
                                          php.rMinLayHex_[il],
                                          php.rMaxLayHex_[il], true);
            if (corner.first > 0) {
              auto cpy = copiesInLayers[php.layer_[il]].find(copy);
              if (cpy == copiesInLayers[php.layer_[il]].end()) 
                copiesInLayers[php.layer_[il]][copy] = ((corner.first == (int)(HGCalParameters::k_CornerSize)) ? php.waferCopy_.size() : -1);
            }
            if ((corner.first > 0) && 
                (corner.first < (int)(HGCalParameters::k_CornerSize))) {
              int wl = HGCalWaferIndex::waferIndex(php.layer_[il],copy,0,true);
              waferTypes[wl] = std::make_pair(corner.first,corner.second);
            }
          }
        }
      }
    }
  }
  php.copiesInLayers_ = copiesInLayers;
  php.waferTypes_     = waferTypes;
  php.nSectors_       = (int)(php.waferCopy_.size());
  php.waferUVMax_     = 0;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalWaferHexagon: # of columns "
                                << incm << " # of rows " << inrm << " and "
                                << kount << ":" << ntot  << " wafers; R "
                                << rin << ":" << rout;
  edm::LogVerbatim("HGCalGeom") << "Dump copiesInLayers for " 
                                << php.copiesInLayers_.size() << " layers";
  for (unsigned int k=0; k<copiesInLayers.size(); ++k) {
    const auto& theModules = copiesInLayers[k];
    edm::LogVerbatim("HGCalGeom") << "Layer " << k << ":" <<theModules.size();
    int k2(0);
    for (std::unordered_map<int, int>::const_iterator itr=theModules.begin();
         itr != theModules.end(); ++itr,++k2) {
      edm::LogVerbatim("HGCalGeom") << "[" << k2 << "] " << itr->first << ":"
                                    << itr->second;
    }
  }
#endif
}

void HGCalGeomParameters::loadWaferHexagon8(HGCalParameters& php) {

  double waferW(php.waferSize_);
  double waferS(php.sensorSeparation_);
  auto wType = std::make_unique<HGCalWaferType>(php.radius100to200_,
                                                php.radius200to300_, 
                                                HGCalParameters::k_ScaleToDDD*(waferW+waferS),
                                                HGCalParameters::k_ScaleToDDD*php.zMinForRad_,
                                                php.choiceType_,
                                                php.nCornerCut_,
                                                php.fracAreaMin_);
  
  double rout(php.rLimit_[1]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Input waferWidth " << waferW << ":" 
                                << waferS << " R Max: " << rout;
#endif
  // Clear the vectors
  php.waferCopy_.clear();
  php.waferTypeL_.clear();
  php.waferTypeT_.clear();
  php.waferPosX_.clear();
  php.waferPosY_.clear();
  double r     = 0.5*(waferW+waferS);
  double R     = 2.0*r/sqrt3_;
  double dy    = 0.75*R;
  int    N     = (r == 0) ? 3 : ((int)(0.5*rout/r) + 3);
  int    ns1   = (2*N+1)*(2*N+1);
  int    ns2   = ns1*php.zLayerHex_.size();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "r " << r << " dy " << dy << " N " << N
                                << " sizes " << ns1 << ":" << ns2;
  std::vector<int> indtypes(ns1+1);
  indtypes.clear();
#endif
  HGCalParameters::wafer_map wafersInLayers(ns1+1);
  HGCalParameters::wafer_map typesInLayers(ns2+1);
  HGCalParameters::waferT_map waferTypes(ns2+1);
  int ipos(0), lpos(0), uvmax(0);
  std::vector<int> uvmx(php.zLayerHex_.size(),0);
  for (int v = -N; v <= N; ++v) {
    for (int u = -N; u <= N; ++u) {
      int nr = 2*v;
      int nc =-2*u+v;
      double xpos = nc*r;
      double ypos = nr*dy;
      int indx = HGCalWaferIndex::waferIndex(0,u,v);
      php.waferCopy_.emplace_back(indx);
      php.waferPosX_.emplace_back(xpos);
      php.waferPosY_.emplace_back(ypos);
      wafersInLayers[indx] = ipos;
      ++ipos;
      std::pair<int,int> corner = 
        HGCalGeomTools::waferCorner(xpos, ypos, r, R, 0, rout, false);
      if ((corner.first == (int)(HGCalParameters::k_CornerSize)) || 
          ((corner.first > 0) && php.defineFull_)) {
        uvmax = std::max(uvmax,std::max(std::abs(u),std::abs(v)));
      }
      for (unsigned int i=0; i<php.zLayerHex_.size(); ++i) {
        int    lay  = php.layer_[php.layerIndex_[i]];
        double zpos = php.zLayerHex_[i];
        int    type = wType->getType(HGCalParameters::k_ScaleToDDD*xpos,HGCalParameters::k_ScaleToDDD*ypos,HGCalParameters::k_ScaleToDDD*zpos);
        php.waferTypeL_.emplace_back(type);
        int kndx = HGCalWaferIndex::waferIndex(lay,u,v);
        typesInLayers[kndx] = lpos;
        ++lpos;
#ifdef EDM_ML_DEBUG
        indtypes.emplace_back(kndx);
#endif
        std::pair<int,int> corner = 
          HGCalGeomTools::waferCorner(xpos, ypos, r, R,
                                      php.rMinLayHex_[i],
                                      php.rMaxLayHex_[i], false);
#ifdef EDM_ML_DEBUG
        if (((corner.first==0) && std::abs(u) < 5 && std::abs(v) < 5) ||
            (std::abs(u) < 2 && std::abs(v) < 2)) {
          edm::LogVerbatim("HGCalGeom") << "Layer " << lay << " R "
                                        << php.rMinLayHex_[i] << ":"
                                        << php.rMaxLayHex_[i] << " u "
                                        << u << " v " << v << " with "
                                        << corner.first << " corners";
        }
#endif
        if ((corner.first == (int)(HGCalParameters::k_CornerSize)) || 
            ((corner.first > 0) && php.defineFull_)) {
          uvmx[i] = std::max(uvmx[i],std::max(std::abs(u),std::abs(v)));
        }
        if ((corner.first < (int)(HGCalParameters::k_CornerSize)) && 
            (corner.first > 0)) {
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "Layer " << lay << " u|v " << u 
                                        << ":" << v << " with " 
                                        << corner.first << " corners First "
                                        << corner.second;
#endif
          int wl = HGCalWaferIndex::waferIndex(lay,u,v);
          waferTypes[wl] = std::make_pair(corner.first,corner.second);
        }
      }
    }
  }
  php.waferUVMax_     = uvmax;
  php.waferUVMaxLayer_= uvmx;
  php.wafersInLayers_ = wafersInLayers;
  php.typesInLayers_  = typesInLayers;
  php.waferTypes_     = waferTypes;
  php.nSectors_       = (int)(php.waferCopy_.size());
  HGCalParameters::hgtrap mytr;
  mytr.lay = 1;           mytr.bl = php.waferR_; 
  mytr.tl = php.waferR_;  mytr.h = php.waferR_; 
  mytr.alpha = 0.0;       mytr.cellSize = HGCalParameters::k_ScaleToDDD*php.waferSize_;
  for (auto const & dz : php.cellThickness_) {
    mytr.dz = 0.5*HGCalParameters::k_ScaleToDDD*dz; 
    php.fillModule(mytr,false);
  }
  for (unsigned k=0; k<php.cellThickness_.size(); ++k) {
    HGCalParameters::hgtrap mytr = php.getModule(k, false);
    mytr.bl       *= HGCalParameters::k_ScaleFromDDD;
    mytr.tl       *= HGCalParameters::k_ScaleFromDDD;
    mytr.h        *= HGCalParameters::k_ScaleFromDDD;
    mytr.dz       *= HGCalParameters::k_ScaleFromDDD;
    mytr.cellSize *= HGCalParameters::k_ScaleFromDDD;
    php.fillModule(mytr, true);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: Total of "
                                << php.waferCopy_.size() << " wafers";
  for (unsigned int k=0; k<php.waferCopy_.size(); ++k) {
    int id = php.waferCopy_[k];
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << std::hex << id
                                  << std::dec << ":" << HGCalWaferIndex::waferLayer(id)
                                  << ":" << HGCalWaferIndex::waferU(id) << ":"
                                  << HGCalWaferIndex::waferV(id) << " x " 
                                  << php.waferPosX_[k] << " y "
                                  << php.waferPosY_[k] << " index "
                                  << php.wafersInLayers_[id];
  }
  edm::LogVerbatim("HGCalGeom") << "HGCalParameters: Total of "
                                << php.waferTypeL_.size() << " wafer types";
  for (unsigned int k=0; k<php.waferTypeL_.size(); ++k) {
    int id = indtypes[k];
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << php.typesInLayers_[id]
                                  << ":" << php.waferTypeL_[k] 
                                  << " ID " << std::hex << id << std::dec
                                  << ":" << HGCalWaferIndex::waferLayer(id) 
                                  << ":" << HGCalWaferIndex::waferU(id) << ":" 
                                  << HGCalWaferIndex::waferV(id);
  }
#endif
}

void HGCalGeomParameters::loadCellParsHexagon(const DDCompactView* cpv,
                                              HGCalParameters& php) {

  //Special parameters for cell parameters
  std::string attribute = "OnlyForHGCalNumbering"; 
  DDSpecificsHasNamedValueFilter filter1{attribute};
  DDFilteredView fv1(*cpv,filter1);
  bool ok = fv1.firstChild();

  if (ok) {
    php.cellFine_   = dbl_to_int(DDVectorGetter::get("waferFine"));
    php.cellCoarse_ = dbl_to_int(DDVectorGetter::get("waferCoarse"));
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalLoadCellPars: " 
                                << php.cellFine_.size()
                                << " rows for fine cells";
  for (unsigned int k=0; k<php.cellFine_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "]: " << php.cellFine_[k];
  edm::LogVerbatim("HGCalGeom") << "HGCalLoadCellPars: " 
                                << php.cellCoarse_.size()
                                << " rows for coarse cells";
  for (unsigned int k=0; k<php.cellCoarse_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "]: " << php.cellCoarse_[k];
#endif
}

void HGCalGeomParameters::loadCellTrapezoid(HGCalParameters& php) {
  // Find the radius of each eta-partitions
  for (unsigned k=0; k<2; ++k) {
    double rmax = ((k == 0) ? 
                   (php.rMaxLayHex_[php.layerFrontBH_[1]-php.firstLayer_]-1) :
                   (php.rMaxLayHex_[php.rMaxLayHex_.size()-1]));
    double rv   = php.rMinLayerBH_[k];
    double zv   = ((k == 0) ?
                   (php.zLayerHex_[php.layerFrontBH_[1]-php.firstLayer_]) :
                   (php.zLayerHex_[php.zLayerHex_.size()-1]));
    php.radiusLayer_[k].emplace_back(rv);
#ifdef EDM_ML_DEBUG
    double eta =-(std::log(std::tan(0.5*std::atan(rv/zv))));
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] rmax " << rmax << " Z = "
                                  << zv << " dEta = " << php.cellSize_[k]
                                  << "\n[0] new R = " << rv << " Eta = " 
                                  << eta;
    int kount(1);
#endif
    while (rv < rmax) {
      double eta =-(php.cellSize_[k]+std::log(std::tan(0.5*std::atan(rv/zv))));
      rv         = zv*std::tan(2.0*std::atan(std::exp(-eta)));
      php.radiusLayer_[k].emplace_back(rv);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "[" << kount << "] new R = " << rv 
                                    << " Eta = " << eta;
      ++kount;
#endif
    }
  }

  // Find minimum and maximum radius index for each layer
  for (unsigned k=0; k<php.zLayerHex_.size(); ++k) {
    int    kk   = php.scintType(php.firstLayer_+(int)(k));
    std::vector<double>::iterator low, high;
    low         = std::lower_bound(php.radiusLayer_[kk].begin(),
                                   php.radiusLayer_[kk].end(), 
                                   php.rMinLayHex_[k]);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] RLow = " 
                                  << php.rMinLayHex_[k] << " pos " 
                                  << (int)(low-php.radiusLayer_[kk].begin());
#endif
    if (low == php.radiusLayer_[kk].begin()) ++low;
    int    irlow  = (int)(low-php.radiusLayer_[kk].begin());
    double drlow  = php.radiusLayer_[kk][irlow] - php.rMinLayHex_[k];
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "irlow " << irlow << " dr " << drlow 
                                  << " min " << php.minTileSize_;
#endif
    if (drlow < php.minTileSize_) {
      ++irlow;
#ifdef EDM_ML_DEBUG
      drlow       = php.radiusLayer_[kk][irlow] - php.rMinLayHex_[k];
      edm::LogVerbatim("HGCalGeom") << "Modified irlow " << irlow << " dr " 
                                    << drlow;
#endif
    }
    high        = std::lower_bound(php.radiusLayer_[kk].begin(),
                                   php.radiusLayer_[kk].end(), 
                                   php.rMaxLayHex_[k]);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] RHigh = " 
                                  << php.rMaxLayHex_[k] << " pos " 
                                  << (int)(high-php.radiusLayer_[kk].begin());
#endif
    if (high == php.radiusLayer_[kk].end()) --high;
    int    irhigh = (int)(high-php.radiusLayer_[kk].begin());
    double drhigh = php.rMaxLayHex_[k] - php.radiusLayer_[kk][irhigh-1];
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "irhigh " << irhigh << " dr " << drhigh 
                                  << " min " << php.minTileSize_;
#endif
    if (drhigh < php.minTileSize_) {
      --irhigh;
#ifdef EDM_ML_DEBUG
      drhigh = php.rMaxLayHex_[k] - php.radiusLayer_[kk][irhigh-1];
      edm::LogVerbatim("HGCalGeom") << "Modified irhigh " << irhigh << " dr "
                                    << drhigh;
#endif
    }
    php.iradMinBH_.emplace_back(irlow);
    php.iradMaxBH_.emplace_back(irhigh);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGcalGeom") << "Layer " << k << " Type " << kk
                                  << " Low edge " << irlow << ":" << drlow
                                  << " Top edge " << irhigh << ":" << drhigh;
#endif
  }

  // Now define the volumes
  int im(0);
  php.waferUVMax_  = 0;
  HGCalParameters::hgtrap mytr;
  mytr.alpha = 0.0;
  for (unsigned int k=0; k<php.zLayerHex_.size(); ++k) {
    if (php.iradMaxBH_[k] > php.waferUVMax_) php.waferUVMax_ = php.iradMaxBH_[k];
    int    kk   = ((php.firstLayer_+(int)(k)) < php.layerFrontBH_[1]) ? 0 : 1;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom")  << "Layer " << php.firstLayer_+k << ":" 
                                   << kk << " Radius range " 
                                   << php.iradMinBH_[k] << ":"
                                   << php.iradMaxBH_[k];
#endif
    mytr.lay = php.firstLayer_ + k;
    for (int irad=php.iradMinBH_[k]; irad<=php.iradMaxBH_[k]; ++irad) {
      double rmin    = php.radiusLayer_[kk][irad-1];
      double rmax    = php.radiusLayer_[kk][irad];
      mytr.bl        = 0.5*rmin*php.scintCellSize(mytr.lay);
      mytr.tl        = 0.5*rmax*php.scintCellSize(mytr.lay);
      mytr.h         = 0.5*(rmax-rmin);
      mytr.dz        = 0.5*php.waferThick_;
      mytr.cellSize  = 0.5*(rmax+rmin)*php.scintCellSize(mytr.lay);
      php.fillModule(mytr,true);
      mytr.bl       *= HGCalParameters::k_ScaleToDDD;
      mytr.tl       *= HGCalParameters::k_ScaleToDDD;
      mytr.h        *= HGCalParameters::k_ScaleToDDD;
      mytr.dz       *= HGCalParameters::k_ScaleToDDD;
      mytr.cellSize *= HGCalParameters::k_ScaleFromDDD;
      php.fillModule(mytr, false);
      if (irad == php.iradMinBH_[k])   php.firstModule_.emplace_back(im);
      ++im;
      if (irad == php.iradMaxBH_[k]-1) php.lastModule_.emplace_back(im);
    }
  }
  php.nSectors_       = php.waferUVMax_;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Maximum radius index " << php.waferUVMax_;
  for (unsigned int k=0; k< php.firstModule_.size(); ++k)
    edm::LogVerbatim("HGCalGeom")  << "Layer " << k+php.firstLayer_
                                   << " Modules " << php.firstModule_[k]
                                   << ":" << php.lastModule_[k];
#endif
}

std::vector<double> HGCalGeomParameters::getDDDArray(const std::string & str, 
                 const DDsvalues_type & sv,
                 const int nmin) {
  DDValue value(str);
  if (DDfetch(&sv,value)) {
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nmin > 0) {
      if (nval < nmin) {
        edm::LogError("HGCalGeom") << "HGCalGeomParameters : # of " << str 
                                   << " bins " << nval << " < " << nmin 
                                   << " ==> illegal";
        throw cms::Exception("DDException") << "HGCalGeomParameters: cannot get array " << str;
      }
    } else {
      if (nval < 1 && nmin == 0) {
        edm::LogError("HGCalGeom") << "HGCalGeomParameters : # of " << str
                                   << " bins " << nval << " < 1 ==> illegal"
                                   << " (nmin=" << nmin << ")";
        throw cms::Exception("DDException") << "HGCalGeomParameters: cannot get array " << str;
      }
    }
    return fvec;
  } else {
    if (nmin >= 0) {
      edm::LogError("HGCalGeom") << "HGCalGeomParameters: cannot get array "
                                 << str;
      throw cms::Exception("DDException") << "HGCalGeomParameters: cannot get array " << str;
    }
    std::vector<double> fvec;
    return fvec;
  }
}

std::pair<double,double>
HGCalGeomParameters::cellPosition(const std::vector<HGCalGeomParameters::cellParameters>& wafers, 
                                  std::vector<HGCalGeomParameters::cellParameters>::const_iterator& itrf,
                                  int wafer, double xx, double yy) {

  if (itrf == wafers.end()) {
    for (std::vector<HGCalGeomParameters::cellParameters>::const_iterator itr = wafers.begin();
         itr != wafers.end(); ++itr) {
      if (itr->wafer == wafer) {
        itrf = itr;
        break;
      }
    }
  }
  double dx(0), dy(0);
  if (itrf != wafers.end()) {
    dx = (xx - itrf->xyz.x());
    if (std::abs(dx) < tolerance) dx = 0;
    dy = (yy - itrf->xyz.y());
    if (std::abs(dy) < tolerance) dy = 0;
  }
  return std::make_pair(dx,dy);
}
