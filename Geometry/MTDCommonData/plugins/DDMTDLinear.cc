#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

#include <cmath>
#include <memory>

using namespace geant_units::operators;
using namespace angle_units::operators;

class DDMTDLinear : public DDAlgorithm {
public:
  DDMTDLinear()
      : m_n(1), m_startCopyNo(1), m_incrCopyNo(1), m_theta(0.), m_phi(0.), m_delta(0.), m_phi_obj(0.), m_theta_obj(0.) {}

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  int m_n;                     //Number of copies
  int m_startCopyNo;           //Start Copy number
  int m_incrCopyNo;            //Increment in Copy number
  double m_theta;              //Theta
  double m_phi;                //Phi dir[Theta,Phi] ... unit-std::vector in direction Theta, Phi
  double m_delta;              //Delta - distance between two subsequent positions along dir[Theta,Phi]
  double m_phi_obj;            //Phi angle to rotate volumes (indipendent from m_phi traslation direction)
  double m_theta_obj;          //Theta angle to rotate volumes
  std::vector<double> m_base;  //Base values - a 3d-point where the offset is calculated from
                               //base is optional, if omitted base=(0,0,0)
  std::pair<std::string, std::string> m_childNmNs;  //Child name
                                                    //Namespace of the child
};

void DDMTDLinear::initialize(const DDNumericArguments& nArgs,
                             const DDVectorArguments& vArgs,
                             const DDMapArguments&,
                             const DDStringArguments& sArgs,
                             const DDStringVectorArguments&) {
  m_n = int(nArgs["N"]);
  m_startCopyNo = int(nArgs["StartCopyNo"]);
  m_incrCopyNo = int(nArgs["IncrCopyNo"]);
  m_theta = nArgs["Theta"];
  m_phi = nArgs["Phi"];
  m_delta = nArgs["Delta"];
  m_base = vArgs["Base"];
  m_phi_obj = nArgs["Phi_obj"];
  m_theta_obj = nArgs["Theta_obj"];

  LogDebug("DDAlgorithm") << "DDMTDLinear: Parameters for position"
                          << "ing:: n " << m_n << " Direction Theta, Phi, Offset, Delta " << convertRadToDeg(m_theta)
                          << " " << convertRadToDeg(m_phi) << " "
                          << " " << convertRadToDeg(m_delta) << " Base " << m_base[0] << ", " << m_base[1] << ", "
                          << m_base[2] << "Objects placement Phi_obj, Theta_obj " << convertRadToDeg(m_phi_obj) << " "
                          << convertRadToDeg(m_theta_obj);

  m_childNmNs = DDSplit(sArgs["ChildName"]);
  if (m_childNmNs.second.empty())
    m_childNmNs.second = DDCurrentNamespace::ns();

  DDName parentName = parent().name();
  LogDebug("DDAlgorithm") << "DDMTDLinear: Parent " << parentName << "\tChild " << m_childNmNs.first << " NameSpace "
                          << m_childNmNs.second;
}

void DDMTDLinear::execute(DDCompactView& cpv) {
  DDName mother = parent().name();
  DDName ddname(m_childNmNs.first, m_childNmNs.second);
  int copy = m_startCopyNo;

  DDTranslation direction(sin(m_theta) * cos(m_phi), sin(m_theta) * sin(m_phi), cos(m_theta));

  DDTranslation basetr(m_base[0], m_base[1], m_base[2]);

  //rotation is in xy plane
  double thetaZ = m_theta_obj - 0.5_pi;
  double phiZ = m_phi_obj;
  double thetaX = m_theta_obj;
  double thetaY = m_theta_obj;
  double phiX = m_phi_obj;
  double phiY = m_phi_obj + 0.5_pi;

  DDRotation rotation = DDRotation("Rotation");

  if (!rotation) {
    LogDebug("DDAlgorithm") << "DDMTDLinear: Creating a new "
                            << "rotation for " << ddname;

    rotation = DDrot("Rotation", DDcreateRotationMatrix(thetaX, phiX, thetaY, phiY, thetaZ, phiZ));
  }

  for (int i = 0; i < m_n; ++i) {
    DDTranslation tran = basetr + (double(i) * m_delta) * direction;
    cpv.position(ddname, mother, copy, tran, rotation);
    LogDebug("DDAlgorithm") << "DDMTDLinear: " << m_childNmNs.second << ":" << m_childNmNs.first << " number " << copy
                            << " positioned in " << mother << " at " << tran << " with " << rotation;
    copy += m_incrCopyNo;
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDMTDLinear, "mtd:DDMTDLinear");
