#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDCMSPlugins.h"

using namespace std;
using namespace dd4hep;
using namespace dd4hep::cms;

static long  algorithm(Detector& /* description */,
                       ParsingContext& ctxt,
                       xml_h e,
                       SensitiveDetector& /* sens */)
{
  Namespace      ns(ctxt,e,true);
  AlgoArguments  args(ctxt, e);
  // Header section of original DDTrackerAngular.h
  int            n           = args.value<int>("N");
  int            startCopyNo = args.find("StartCopyNo") ? args.value<int>("StartCopyNo") : 1;
  int            incrCopyNo  = args.find("IncrCopyNo") ? args.value<int>("IncrCopyNo") : 1;
  double         rangeAngle  = args.value<double>("RangeAngle");      //Range in angle
  double         startAngle  = args.value<double>("StartAngle");      //Start anle
  double         radius      = args.value<double>("Radius");          //Radius
  vector<double> center      = args.value<vector<double> >("Center"); //Phi values
  Volume         mother      = ns.volume(args.parentName());
  Volume         child       = ns.volume(args.value<string>("ChildName"));

  double         delta = 0e0;                                         //Increment in phi
  // Code section of original DDTrackerAngular.cc
  if (fabs(rangeAngle-360.0*CLHEP::deg)<0.001*CLHEP::deg) { 
    delta    =   rangeAngle/double(n);
  } else if (n > 1) {
    delta    =   rangeAngle/double(n-1);
  }  

  LogDebug("TrackerGeom") << "debug: Parameters for positioning:: n "
      << n << " Start, Range, Delta " 
      << startAngle/CLHEP::deg << " " 
      << rangeAngle/CLHEP::deg << " " << delta/CLHEP::deg
      << " Radius " << radius << " Centre " << center[0] 
      << ", " << center[1] << ", "<<center[2];
  LogDebug("TrackerGeom") << "debug: Parent " << mother.name() 
      << "\tChild " << child.name() << " NameSpace "
      << ns.name;

  double theta  = 90.*CLHEP::deg;
  int    copy   = startCopyNo;
  double phi    = startAngle;
  for (int i=0; i<n; i++) {
    double phix = phi;
    double phiy = phix + 90.*CLHEP::deg;
    double phideg = phix/CLHEP::deg;

    Rotation3D rotation;
    if (phideg != 0) {
      string rotstr = ns.ns_name(child.name()) + std::to_string(phideg*10.);
      auto irot = ctxt.rotations.find(ns.prepend(rotstr));
      if ( irot != ctxt.rotations.end() )   {
        rotation = ns.rotation(ns.prepend(rotstr));
      }
      else  {
        LogDebug("TrackerGeom") << "Creating a new "
            << "rotation: " << rotstr << "\t90., " 
            << phix/CLHEP::deg << ", 90.," 
            << phiy/CLHEP::deg <<", 0, 0";
        RotationZYX   rot;
        rotation = make_rotation3D(theta, phix, theta, phiy, 0., 0.);
      }
    }
	
    double xpos = radius*cos(phi) + center[0];
    double ypos = radius*sin(phi) + center[1];
    double zpos = center[2];
    Position tran(xpos, ypos, zpos);
    mother.placeVolume(child, copy, Transform3D(rotation,tran));
    LogDebug("TrackerGeom") << "test " << child.name() << " number " 
                            << copy << " positioned in " << mother.name() << " at "
                            << tran  << " with " << rotation;
    copy += incrCopyNo;
    phi  += delta;
  }
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTrackerAngular,algorithm)
