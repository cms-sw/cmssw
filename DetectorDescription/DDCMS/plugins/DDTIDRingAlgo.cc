#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDCMSPlugins.h"

using namespace std;
using namespace dd4hep;
using namespace dd4hep::cms;

static long algorithm(Detector& /* description */,
                      ParsingContext& ctxt,
                      xml_h e,
                      SensitiveDetector& /* sens */)
{
  Namespace      ns(ctxt, e, true);
  AlgoArguments  args(ctxt, e);
  Volume         mother     = ns.volume(args.parentName());
  vector<string> moduleName = args.value<vector<string> >("ModuleName"); //Name of the module
  string         iccName    = args.value<string>("ICCName");             //Name of the ICC
  int            number     = args.value<int>("Number");                 //Number of copies
  double         startAngle = args.value<double>("StartAngle");          //Phi offset
  double         rModule    = args.value<double>("ModuleR");             //Location of module in R 
  vector<double> zModule    = args.value<vector<double> >("ModuleZ");    //                   in Z
  double         rICC       = args.value<double>("ICCR");                //Location of ICC    in R 
  double         sICC       = args.value<double>("ICCShift");            //Shift of ICC       per to R
  vector<double> zICC       = args.value<vector<double> >("ICCZ");       //                   in Z

  LogDebug("TIDGeom") << "Parent " << mother.name() 
                      << "\tModule " << moduleName[0] << ", "
                      << moduleName[1] << "\tICC " << iccName 
                      << "\tNameSpace " << ns.name;
  LogDebug("TIDGeom") << "Parameters for positioning--"
                      << " StartAngle " << startAngle/CLHEP::deg
                      << " Copy Numbers " << number << " Modules at R " 
                      << rModule << " Z " << zModule[0] << ", " << zModule[1] 
                      << " ICCs at R " << rICC << " Z " << zICC[0] << ", " 
                      << zICC[1]; 
  double theta = 90.*CLHEP::deg;
  double phiy  = 0.*CLHEP::deg;
  double dphi  = CLHEP::twopi/number;

  //Loop over modules
  Volume icc  = ns.volume(iccName);
  Volume mod0 = ns.volume(moduleName[0]);
  Volume mod1 = ns.volume(moduleName[1]);
  for (int i=0; i<number; i++) {
    //First the module
    double phiz = startAngle + i*dphi;
    double xpos = rModule*cos(phiz);
    double ypos = rModule*sin(phiz);
    double zpos, thetay, phix;
    Volume module;
    if (i%2 == 0) {
      phix   = phiz + 90.*CLHEP::deg;
      thetay = 0*CLHEP::deg;
      zpos   = zModule[0];
      module = mod0;
    } else {
      phix   = phiz - 90.*CLHEP::deg;
      thetay = 180*CLHEP::deg;
      zpos   = zModule[1];
      module = mod1;
    }
    
    // stereo face inside toward structure, rphi face outside
    phix   = phix   - 180.*CLHEP::deg;
    thetay = thetay + 180.*CLHEP::deg;
    //
    Position    trmod(xpos, ypos, zpos);
    Rotation3D  rotation = make_rotation3D(theta, phix, thetay, phiy, theta, phiz);
    // int copyNr = i+1;
    /* PlacedVolume pv = */ mother.placeVolume(module, i+1, Transform3D(rotation,trmod));
    LogDebug("TIDGeom") << module.name() << " number "
                        << i+1 << " positioned in " << mother.name() << " at "
                        << trmod << " with " << rotation;
    //Now the ICC
    if (i%2 == 0 ) {
      zpos = zICC[0];
      xpos = rICC*cos(phiz) + sICC*sin(phiz);
      ypos = rICC*sin(phiz) - sICC*cos(phiz);
    } else {
      zpos = zICC[1];
      xpos = rICC*cos(phiz) - sICC*sin(phiz);
      ypos = rICC*sin(phiz) + sICC*cos(phiz);
    }
    // int copyNr = i+1;
    Position tricc(xpos, ypos, zpos);
    /* PlacedVolume pv = */ mother.placeVolume(icc, i+1, Transform3D(rotation,tricc));
    LogDebug("TIDGeom") << iccName << " number " 
                        << i+1 << " positioned in " << mother.name() << " at "
                        << tricc << " with " << rotation;
  }
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTIDRingAlgo,algorithm)
