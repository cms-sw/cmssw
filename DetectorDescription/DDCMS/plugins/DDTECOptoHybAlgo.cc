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
  Namespace      ns(ctxt,e,true);
  AlgoArguments  args(ctxt, e);
  int            startCopyNo = args.value<int>("StartCopyNo");
  double         rpos        = args.value<double>("Rpos");
  double         zpos        = args.value<double>("Zpos");
  double         optoHeight  = args.value<double>("OptoHeight");
  double         optoWidth   = args.value<double>("OptoWidth");
  vector<double> angles      = args.value<vector<double> >("Angles");
  Volume         child       = ns.volume(args.value<string>("ChildName"));
  Volume         mother      = ns.volume(args.parentName());

  LogDebug("TECGeom") << "Parent " << mother.name()
                      << " Child " << child.name() << " NameSpace " << ns.name;
  LogDebug("TECGeom") << "Height of the Hybrid "
                      << optoHeight << " and Width " << optoWidth
                      <<"Rpos " << rpos << " Zpos " << zpos 
                      << " StartCopyNo " << startCopyNo << " Number " 
                      << angles.size();

  // given r positions are for the lower left corner
  rpos += optoHeight/2;
  int    copyNo = startCopyNo;
  for (double angle : angles) {
    double phix = -angle;
    // given phi positions are for the lower left corner
    phix += asin(optoWidth/2/rpos);
    double xpos = rpos * cos(phix);
    double ypos = rpos * sin(phix);
    Position tran(xpos, ypos, zpos);

    Rotation3D rotation;
    double phiy = phix + 90.*CLHEP::deg;
    double phideg = phix/CLHEP::deg;
    if (phideg != 0) {
      string rotstr= ns.ns_name(child.name()) + std::to_string(phideg*1000.);
      auto irot = ctxt.rotations.find(ns.prepend(rotstr));
      if ( irot != ctxt.rotations.end() )   {
        rotation = ns.rotation(ns.prepend(rotstr));
      }
      else  {
        double theta = 90.*CLHEP::deg;
        LogDebug("TECGeom") << "test: Creating a new "
                            << "rotation: " << rotstr << "\t90., " 
                            << phix/CLHEP::deg << ", 90.," << phiy/CLHEP::deg
                            << ", 0, 0";
        rotation = make_rotation3D(theta, phix, theta, phiy, 0., 0.);
      }
    }
    mother.placeVolume(child, copyNo, Transform3D(rotation,tran));
    LogDebug("TECGeom") << "test " << child.name() << " number " 
                        << copyNo << " positioned in " << mother.name() << " at "
                        << tran  << " with " << rotation;
    copyNo++;
  }
  LogDebug("TECGeom") << "<<== End of DDTECOptoHybAlgo construction ...";
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTECOptoHybAlgo,algorithm)
