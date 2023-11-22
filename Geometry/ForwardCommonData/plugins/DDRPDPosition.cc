///////////////////////////////////////////////////////////////////////////////
// File: DDRPDPosition.cc
// Description: Position inside the mother according to phi
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDutils.h"

//#define EDM_ML_DEBUG

class DDRPDPosition : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDRPDPosition();

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  std::vector<double> xpos_;  //Positions along x-axis
  double ypos_;               //Position along y-axis
  double zpos_;               //Position along z-axis
  std::string childName_;     //Children name
};

DDRPDPosition::DDRPDPosition() { edm::LogVerbatim("ForwardGeom") << "DDRPDPosition test: Creating an instance"; }

void DDRPDPosition::initialize(const DDNumericArguments& nArgs,
                               const DDVectorArguments& vArgs,
                               const DDMapArguments&,
                               const DDStringArguments& sArgs,
                               const DDStringVectorArguments&) {
  xpos_ = vArgs["positionX"];
  ypos_ = nArgs["positionY"];
  zpos_ = nArgs["positionZ"];
  childName_ = sArgs["ChildName"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("ForwardGeom") << "DDRPDPosition: Parameters for positioning-- " << xpos_.size() << " copies of "
                                  << childName_ << " to be positioned inside " << parent().name() << " at y = " << ypos_
                                  << ", z = " << zpos_ << " and at x = (";
  std::ostringstream st1;
  for (const auto& x : xpos_)
    st1 << x << " ";
  edm::LogVerbatim("ForwardGeom") << st1.str() << ")";
#endif
}

void DDRPDPosition::execute(DDCompactView& cpv) {
  DDName child(DDSplit(childName_).first, DDSplit(childName_).second);
  DDName parentName = parent().name();
  DDRotation rot;

  for (unsigned int jj = 0; jj < xpos_.size(); jj++) {
    DDTranslation tran(xpos_[jj], ypos_, zpos_);

    cpv.position(child, parentName, jj + 1, tran, rot);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ForwardGeom") << "DDRPDPosition: " << child << " number " << jj + 1 << " positioned in "
                                    << parentName << " at " << tran << " with no rotation";
#endif
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDRPDPosition, "rpdalgo:DDRPDPosition");
