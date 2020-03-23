#include <cmath>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/GeometryVector/interface/Phi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"

using namespace geant_units::operators;

//#define EDM_ML_DEBUG

class DDTotemAngular : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDTotemAngular();

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  double startAngle_;  //Start angle
  double stepAngle_;   //Step  angle
  double zoffset_;     //Offset in z
  double roffset_;     //Offset in R
  int n_;              //Number of copies
  int startCopyNo_;    //Start copy Number
  int incrCopyNo_;     //Increment copy Number

  std::string rotns_;        //Namespace for rotation matrix
  std::string idNameSpace_;  //Namespace of this and ALL sub-parts
  std::string childName_;    //Children name
};

DDTotemAngular::DDTotemAngular() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("ForwardGeom") << "DDTotemAngular: Creating an instance";
#endif
}

void DDTotemAngular::initialize(const DDNumericArguments& nArgs,
                                const DDVectorArguments&,
                                const DDMapArguments&,
                                const DDStringArguments& sArgs,
                                const DDStringVectorArguments&) {
  startAngle_ = nArgs["startAngle"];
  stepAngle_ = nArgs["stepAngle"];
  zoffset_ = nArgs["zoffset"];
  roffset_ = nArgs["roffset"];
  n_ = int(nArgs["n"]);
  startCopyNo_ = int(nArgs["startCopyNo"]);
  incrCopyNo_ = int(nArgs["incrCopyNo"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("ForwardGeom") << "DDTotemAngular: Parameters for positioning-- " << n_ << " copies in steps of "
                                  << convertRadToDeg(stepAngle_) << " from " << convertRadToDeg(startAngle_)
                                  << " \tZoffset " << zoffset_ << " \tRoffset " << roffset_
                                  << "\tStart and inremental copy nos " << startCopyNo_ << ", " << incrCopyNo_;
#endif
  rotns_ = sArgs["RotNameSpace"];
  idNameSpace_ = DDCurrentNamespace::ns();
  childName_ = sArgs["ChildName"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("ForwardGeom") << "DDTotemAngular debug: Parent " << parent().name() << "\tChild " << childName_
                                  << "\tNameSpace " << idNameSpace_ << "\tRotation Namespace " << rotns_;
#endif
}

void DDTotemAngular::execute(DDCompactView& cpv) {
  double phi = startAngle_;
  int copyNo = startCopyNo_;

  for (int ii = 0; ii < n_; ii++) {
    Geom::Phi0To2pi<double> phitmp = phi;
    DDRotation rotation;
    std::string rotstr("NULL");

    rotstr = "RT" + formatAsDegrees(phitmp);
    rotation = DDRotation(DDName(rotstr, rotns_));
    if (!rotation) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("ForwardGeom") << "DDTotemAngular: Creating a new rotation " << DDName(rotstr, rotns_)
                                      << "\t90, " << convertRadToDeg(phitmp + 90._deg) << ", 0, 0, 90, "
                                      << convertRadToDeg(phitmp);
#endif
      rotation = DDrot(DDName(rotstr, rotns_), 90._deg, 90._deg + phitmp, 0., 0., 90._deg, phitmp);
    }

    DDTranslation tran(roffset_ * cos(phi), roffset_ * sin(phi), zoffset_);

    DDName parentName = parent().name();
    cpv.position(DDName(childName_, idNameSpace_), parentName, copyNo, tran, rotation);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ForwardGeom") << "DDTotemAngular: " << DDName(childName_, idNameSpace_) << " number " << copyNo
                                    << " positioned in " << parentName << " at " << tran << " with " << rotstr << " "
                                    << rotation;
#endif
    phi += stepAngle_;
    copyNo += incrCopyNo_;
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDTotemAngular, "forward:DDTotemAngular");
