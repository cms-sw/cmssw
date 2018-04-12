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
  std::cout << "******** My DDTECPhiAlgo!!!!!!\n";

  Namespace     ns(ctxt, e, true);
  AlgoArguments args(ctxt, e);
  Volume        mother      = ns.volume(args.parentName());
  Volume        child       = ns.volume(args.childName());
  double        startAngle  = args.value<double>("StartAngle");  //Start angle
  double        incrAngle   = args.value<double>("IncrAngle");   //Increment in angle
  double        zIn         = args.value<double>("ZIn");         //z position for the even ones
  double        zOut        = args.value<double>("ZOut");        //z position for the odd  ones
  int           number      = args.value<double>("Number");      //Number of copies
  int           startCopyNo = args.find("StartCopyNo") ? args.value<int>("StartCopyNo") : 1; //Start copy number
  int           incrCopyNo  = args.find("IncrCopyNo") ? args.value<int>("IncrCopyNo") : 1;  //Increment in copy number

  LogDebug("TECGeom") << "debug: Parameters for "
      << "positioning--" << "\tStartAngle " 
      << startAngle/CLHEP::deg << "\tIncrAngle " 
      << incrAngle/CLHEP::deg << "\tZ in/out " << zIn << ", " 
      << zOut 	      << "\tCopy Numbers " << number 
      << " Start/Increment " << startCopyNo << ", " 
      << incrCopyNo;
  LogDebug("TECGeom") << "debug: Parent " << mother.name() 
      << "\tChild " << child.name() << " NameSpace " 
      << ns.name;

  if (number > 0) {
    double theta  = 90.*CLHEP::deg;
    int    copyNo = startCopyNo;
    for (int i=0; i<number; i++) {
      double phix = startAngle + i*incrAngle;
      double phiy = phix + 90.*CLHEP::deg;
      Rotation3D rotation = make_rotation3D(theta, phix, theta, phiy, 0, 0);
      Position   tran(0., 0., (i%2 == 0) ? zIn : zOut);
      /* PlacedVolume pv = */ mother.placeVolume(child, copyNo, Transform3D(rotation,tran));
      LogDebug("TECGeom") << "test: " << child.name() <<" number "
          << copyNo << " positioned in " << mother.name() <<" at "
          << tran << " with " << rotation;
      copyNo += incrCopyNo;
    }
  }
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTECPhiAlgo,algorithm)
