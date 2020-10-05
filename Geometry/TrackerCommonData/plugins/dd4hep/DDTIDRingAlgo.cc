#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace dd4hep;
using namespace cms;
using namespace cms_units::operators;

static long algorithm(Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  DDAlgoArguments args(ctxt, e);
  Volume mother = ns.volume(args.parentName());
  vector<string> moduleName = args.value<vector<string> >("ModuleName");  //Name of the module
  string iccName = args.value<string>("ICCName");                         //Name of the ICC
  int number = args.value<int>("Number");                                 //Number of copies
  double startAngle = args.value<double>("StartAngle");                   //Phi offset
  double rModule = args.value<double>("ModuleR");                         //Location of module in R
  vector<double> zModule = args.value<vector<double> >("ModuleZ");        //                   in Z
  double rICC = args.value<double>("ICCR");                               //Location of ICC    in R
  double sICC = args.value<double>("ICCShift");                           //Shift of ICC       per to R
  vector<double> zICC = args.value<vector<double> >("ICCZ");              //                   in Z

  LogDebug("TIDGeom") << "Parent " << mother.name() << "\tModule " << moduleName[0] << ", " << moduleName[1] << "\tICC "
                      << iccName << "\tNameSpace " << ns.name();
  LogDebug("TIDGeom") << "Parameters for positioning--"
                      << " StartAngle " << convertRadToDeg(startAngle) << " Copy Numbers " << number << " Modules at R "
                      << rModule << " Z " << zModule[0] << ", " << zModule[1] << " ICCs at R " << rICC << " Z "
                      << zICC[0] << ", " << zICC[1];
  double theta = 90._deg;
  double phiy = 0._deg;
  double dphi = 2_pi / number;

  //Loop over modules
  Volume icc = ns.volume(iccName);
  Volume mod0 = ns.volume(moduleName[0]);
  Volume mod1 = ns.volume(moduleName[1]);
  for (int i = 0; i < number; i++) {
    //First the module
    double phiz = startAngle + i * dphi;
    double xpos = rModule * cos(phiz);
    double ypos = rModule * sin(phiz);
    double zpos, thetay, phix;
    Volume module;
    if (i % 2 == 0) {
      phix = phiz + 90._deg;
      thetay = 0._deg;
      zpos = zModule[0];
      module = mod0;
    } else {
      phix = phiz - 90._deg;
      thetay = 180._deg;
      zpos = zModule[1];
      module = mod1;
    }

    // stereo face inside toward structure, rphi face outside
    phix = phix - 180._deg;
    thetay = thetay + 180._deg;
    //
    Position trmod(xpos, ypos, zpos);
    Rotation3D rotation = makeRotation3D(theta, phix, thetay, phiy, theta, phiz);
    // int copyNr = i+1;
    /* PlacedVolume pv = */ mother.placeVolume(module, i + 1, Transform3D(rotation, trmod));
    LogDebug("TIDGeom") << module.name() << " number " << i + 1 << " positioned in " << mother.name() << " at " << trmod
                        << " with " << rotation;
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
    // int copyNr = i+1;
    Position tricc(xpos, ypos, zpos);
    /* PlacedVolume pv = */ mother.placeVolume(icc, i + 1, Transform3D(rotation, tricc));
    LogDebug("TIDGeom") << iccName << " number " << i + 1 << " positioned in " << mother.name() << " at " << tricc
                        << " with " << rotation;
  }
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTIDRingAlgo, algorithm)
