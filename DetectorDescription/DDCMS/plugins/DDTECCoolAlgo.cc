#include "DD4hep/DetFactoryHelper.h"
#include "DD4hep/Printout.h"
#include "DetectorDescription/DDCMS/interface/DDCMSPlugins.h"

#include <sstream>

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
  int            startCopyNo = args.find("StartCopyNo") ? args.value<int>("StartCopyNo") : 1;
  double         rPosition   = args.value<double>("RPosition");
  vector<double> phiPosition = args.value<vector<double> >("PhiPosition");
  vector<string> coolInsert  = args.value<vector<string> >("CoolInsert");
  Volume         mother      = ns.volume(args.parentName());

  LogDebug("TECGeom") << "debug: Parent " << mother.name() 
                      <<" NameSpace " << ns.name << " at radial Position " << rPosition;
  if (phiPosition.size() == coolInsert.size()) {
    for (int i=0; i<(int)(phiPosition.size()); i++)   {
      LogDebug("TECGeom") << "debug: Insert[" << i << "]: "
          << coolInsert.at(i) << " at Phi " 
          << phiPosition.at(i)/CLHEP::deg;
    }
  }
  else {
    LogDebug("TECGeom") << "ERROR: Number of inserts does not match the numer of PhiPositions!";
  }

  int copyNo  = startCopyNo;
  // loop over the inserts to be placed
  for (int i = 0; i < (int)(coolInsert.size()); i++) {
    Volume child = ns.volume(ns.real_name(coolInsert.at(i)));
    // get positions
    double xpos =  rPosition*cos(phiPosition.at(i));
    double ypos = -rPosition*sin(phiPosition.at(i));
    // place inserts
    Position tran(xpos, ypos, 0.0);
    mother.placeVolume(child, copyNo, tran);
    LogDebug("TECGeom") << "test " << child.name() << "["  
                        << copyNo << "] positioned in " << mother.name()
                        << " at " << tran
                        << " phi " << phiPosition.at(i)/CLHEP::deg << " r " 
                        << rPosition;
    copyNo++;
  }
  LogDebug("TECGeom") << "Finished....";
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTECCoolAlgo,algorithm)

