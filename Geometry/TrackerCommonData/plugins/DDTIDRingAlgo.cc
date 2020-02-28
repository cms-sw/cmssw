///////////////////////////////////////////////////////////////////////////////
// File: DDTIDRingAlgo.cc
// Description: Position n copies of detectors in alternate positions and
//              also associated ICC's
// Em 17Sep07: Cool inserts moved to DDTIDModulePosAlgo.h
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <string>
#include <vector>

using namespace std;

class DDTIDRingAlgo : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDTIDRingAlgo();
  ~DDTIDRingAlgo() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  string idNameSpace;         //Namespace of this & ALL subparts
  vector<string> moduleName;  //Name of the module
  string iccName;             //Name of the ICC

  int number;              //Number of copies
  double startAngle;       //Phi offset
  double rModule;          //Location of module in R
  vector<double> zModule;  //                   in Z
  double rICC;             //Location of ICC    in R
  double sICC;             //Shift of ICC       per to R
  vector<double> zICC;     //                   in Z
};

DDTIDRingAlgo::DDTIDRingAlgo() { LogDebug("TIDGeom") << "DDTIDRingAlgo info: Creating an instance"; }

DDTIDRingAlgo::~DDTIDRingAlgo() {}

void DDTIDRingAlgo::initialize(const DDNumericArguments& nArgs,
                               const DDVectorArguments& vArgs,
                               const DDMapArguments&,
                               const DDStringArguments& sArgs,
                               const DDStringVectorArguments& vsArgs) {
  idNameSpace = DDCurrentNamespace::ns();
  moduleName = vsArgs["ModuleName"];
  iccName = sArgs["ICCName"];
  DDName parentName = parent().name();
  LogDebug("TIDGeom") << "DDTIDRingAlgo debug: Parent " << parentName << "\tModule " << moduleName[0] << ", "
                      << moduleName[1] << "\tICC " << iccName << "\tNameSpace " << idNameSpace;

  number = int(nArgs["Number"]);
  startAngle = nArgs["StartAngle"];
  rModule = nArgs["ModuleR"];
  zModule = vArgs["ModuleZ"];
  rICC = nArgs["ICCR"];
  sICC = nArgs["ICCShift"];
  zICC = vArgs["ICCZ"];

  LogDebug("TIDGeom") << "DDTIDRingAlgo debug: Parameters for positioning--"
                      << " StartAngle " << startAngle / CLHEP::deg << " Copy Numbers " << number << " Modules at R "
                      << rModule << " Z " << zModule[0] << ", " << zModule[1] << " ICCs at R " << rICC << " Z "
                      << zICC[0] << ", " << zICC[1];
}

void DDTIDRingAlgo::execute(DDCompactView& cpv) {
  double theta = 90. * CLHEP::deg;
  double phiy = 0. * CLHEP::deg;
  double dphi = CLHEP::twopi / number;

  DDName mother = parent().name();
  DDName module;
  DDName icc(DDSplit(iccName).first, DDSplit(iccName).second);

  //Loop over modules
  for (int i = 0; i < number; i++) {
    //First the module
    double phiz = startAngle + i * dphi;
    double xpos = rModule * cos(phiz);
    double ypos = rModule * sin(phiz);
    double zpos, thetay, phix;
    if (i % 2 == 0) {
      phix = phiz + 90. * CLHEP::deg;
      thetay = 0 * CLHEP::deg;
      zpos = zModule[0];
      module = DDName(DDSplit(moduleName[0]).first, DDSplit(moduleName[0]).second);
    } else {
      phix = phiz - 90. * CLHEP::deg;
      thetay = 180 * CLHEP::deg;
      zpos = zModule[1];
      module = DDName(DDSplit(moduleName[1]).first, DDSplit(moduleName[1]).second);
    }

    // stereo face inside toward structure, rphi face outside
    phix = phix - 180. * CLHEP::deg;
    thetay = thetay + 180. * CLHEP::deg;
    //

    DDTranslation trmod(xpos, ypos, zpos);
    double phideg = phiz / CLHEP::deg;
    DDRotation rotation;
    string rotstr = mother.name() + to_string(phideg * 10.);
    rotation = DDRotation(DDName(rotstr, idNameSpace));
    if (!rotation) {
      LogDebug("TIDGeom") << "DDTIDRingAlgo test: Creating a new rotation " << rotstr << "\t" << theta / CLHEP::deg
                          << ", " << phix / CLHEP::deg << ", " << thetay / CLHEP::deg << ", " << phiy / CLHEP::deg
                          << ", " << theta / CLHEP::deg << ", " << phiz / CLHEP::deg;
      rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, thetay, phiy, theta, phiz);
    }

    cpv.position(module, mother, i + 1, trmod, rotation);
    LogDebug("TIDGeom") << "DDTIDRingAlgo test: " << module << " number " << i + 1 << " positioned in " << mother
                        << " at " << trmod << " with " << rotation;

    //Now the ICC
    if (i % 2 == 0) {
      zpos = zICC[0];
      xpos = rICC * cos(phiz) + sICC * sin(phiz);
      ypos = rICC * sin(phiz) - sICC * cos(phiz);
    } else {
      zpos = zICC[1];
      xpos = rICC * cos(phiz) - sICC * sin(phiz);
      ypos = rICC * sin(phiz) + sICC * cos(phiz);
    }
    DDTranslation tricc(xpos, ypos, zpos);
    cpv.position(icc, mother, i + 1, tricc, rotation);
    LogDebug("TIDGeom") << "DDTIDRingAlgo test: " << icc << " number " << i + 1 << " positioned in " << mother << " at "
                        << tricc << " with " << rotation;
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDTIDRingAlgo, "track:DDTIDRingAlgo");
