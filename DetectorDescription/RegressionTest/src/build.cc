#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDUnits.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/FIPConfiguration.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Math/GenVector/AxisAngle.h"
#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/Rotation3D.h"
#include "Math/GenVector/RotationZ.h"

using namespace std;
using namespace dd::operators;

/*
File setup.xml:
  Material(mixt) Air
  LogicalPart world
  Solid world
  Constant length
  Constant corner
File elements.xml:
  Material(elem) Nitrogen
  Material(elem) Oxygen  
*/
void regressionTest_setup(ClhepEvaluator& eval) {
   
   string ns = "setup"; // current namespace faking the filename 'setup.xml'
   
   // length of a side of  world cube
   eval.set(ns,"length","20.*m"); 
   
   // center of a corner in the cube
   eval.set(ns,"corner","[length]/4.");
   
   // world-solid
   DDName worldName("world",ns);
   DDName airName("Air",ns);
   DDName nitrogenName("Nitrogen","elements");
   DDName oxygenName("Oxygen","elements");
   
   DDSolidFactory::box(worldName, eval.eval(ns,"[length]/2."),
                    eval.eval(ns,"[length]/2."),
		    eval.eval(ns,"[length]/2."));
   
   DDLogicalPart(worldName, // name
                 airName,   // material
		 worldName  // solid
		);

   DDMaterial air(airName,eval.eval(ns,"1.214*mg/cm3")); // mixture for Air
   air.addMaterial(DDMaterial(nitrogenName),eval.eval(ns,"0.75"));
   air.addMaterial(DDMaterial(oxygenName),eval.eval(ns,"0.25"));
   
   cout << air << endl;
   
   DDMaterial(nitrogenName,      // name
              eval.eval(ns,"7"), // Z
	      eval.eval(ns,"14.007*g/mole"), // A
	      eval.eval(ns,"0.808*g/cm3") ); // density
	      
   DDMaterial(oxygenName,      // name
              eval.eval(ns,"8"), // Z
	      eval.eval(ns,"15.999*g/mole"), // A
	      eval.eval(ns,"1.43*g/cm3") ); // density

   cout << air << endl;   

   // Some rotations in the x-y plane (Unit, 30,60,90 degs)
   std::unique_ptr<DDRotationMatrix> r0  = std::make_unique<DDRotationMatrix>();
   std::unique_ptr<DDRotationMatrix> r30 = std::make_unique<DDRotationMatrix>(ROOT::Math::RotationZ( 30._deg ));
   std::unique_ptr<DDRotationMatrix> r60 = std::make_unique<DDRotationMatrix>(ROOT::Math::RotationZ( 60._deg ));
   std::unique_ptr<DDRotationMatrix> r90 = std::make_unique<DDRotationMatrix>(ROOT::Math::RotationZ( 90._deg ));
   
   DDrot( DDName( "Unit", ns ), std::move( r0 ));
   DDrot( DDName( "R30", ns ), std::move( r30 ));
   DDrot( DDName( "R60", ns ), std::move( r60 ));
   DDrot( DDName( "R90", ns ), std::move( r90 ));
   
   DDSolid collectorSolid = DDSolidFactory::shapeless( DDName( "group", ns ));
   
   DDRootDef::instance().set( worldName );
}

///////////////////////////////////////////////////////////////////////////
/*
  naming convention for the 8 corners in the world:
  +++ (upper, +x, +y)
  -++ (lower, +x, +y)
  +-+ (upper, -x, +y)
  and so on ...
*/

// Fills corner +++ with a hierarchy of boxes ...
/*
  File: first.xml
  
*/ 
void regressionTest_first(ClhepEvaluator& eval ) {
  ///load the new cpv
  DDCompactView cpv;
  cout << "main::initialize DDL parser" << endl;
  DDLParser myP(cpv);
  
  cout << "main::about to set configuration" << endl;
  
  string ns("first");
  DDSolid support = DDSolidFactory::box(DDName("support",ns),
					eval.eval(ns,"[setup:corner]/4."),
					eval.eval(ns,"[setup:corner]/8."),
					eval.eval(ns,"[setup:corner]/4.")
					);
  DDSolid sensor = DDSolidFactory::box(DDName("sensor",ns),			   
				       eval.eval(ns,"[setup:corner]/16."),
				       eval.eval(ns,"[setup:corner]/16."),
				       eval.eval(ns,"[setup:corner]/16.")
				       );
  
  DDLogicalPart supportLP(DDName("support",ns),     // name
			  DDName("Oxygen","elements"), // material
			  DDName("support",ns));    // solid
  
  DDLogicalPart sensorLP(DDName("sensor",ns),
			 DDName("Nitrogen","elements"),
			 DDName("sensor",ns));	
  
  DDLogicalPart part(DDName("group",ns),
		     DDName("Air","setup"),
		     DDName("group","setup")    
		      );
  
  DDRotation r30(DDName("R30","setup"));
  DDRotation r60(DDName("R60","setup"));
  DDRotation r90(DDName("R90","setup"));
  DDRotation unit(DDName("Unit","setup"));
  DDTranslation t0;
  DDTranslation t1(eval.eval(ns,"[setup:corner]/8."),
		   eval.eval(ns,"[setup:corner]/16."),
		   eval.eval(ns,"[setup:corner]/8.")
		   );
  DDTranslation t2(eval.eval(ns,"[setup:corner]*1.25*cos(0.)"),
		   eval.eval(ns,"[setup:corner]*1.25*sin(0.)"),
		   eval.eval(ns,"0."));
  DDTranslation t3(eval.eval(ns,"[setup:corner]*1.25*cos(30.*deg)"),
		   eval.eval(ns,"[setup:corner]*1.25*sin(30.*deg)"),
		    eval.eval(ns,"0."));
  DDTranslation t4(eval.eval(ns,"[setup:corner]*1.25*cos(60.*deg)"),
		   eval.eval(ns,"[setup:corner]*1.25*sin(60.*deg)"),
		   eval.eval(ns,"0."));
  DDTranslation t5(eval.eval(ns,"[setup:corner]*1.25*cos(90.*deg)"),
		   eval.eval(ns,"[setup:corner]*1.25*sin(90.*deg)"),
		   eval.eval(ns,"0."));
  
  cpv.position(sensorLP, supportLP, std::string("1"), t1, unit);
  cpv.position(supportLP, part, std::string("1"), t2, unit);
  cpv.position(supportLP, part, std::string("2"), t3, r30);
  cpv.position(supportLP, part, std::string("3"), t4, r60);
  cpv.position(supportLP, part, std::string("4"), t5, r90);
   
  std::unique_ptr<DDRotationMatrix> rm = std::make_unique<DDRotationMatrix>( ROOT::Math::AxisAngle( DD3Vector( 1., 1., 1. ), 20._deg ));
  DDRotation rw = DDrot( DDName( "group", ns ), std::move( rm ));
  DDLogicalPart ws( DDName( "world", "setup" ));
  cpv.position( part, ws, std::string( "1" ), t0, rw );
}


void output(string filename) 
{
  ostream & os(cout);

  os << "Starting Regressiontest Output" << endl;
  ///load the new cpv
  DDCompactView cpv;
  cout << "main::initialize DDL parser" << endl;
  DDLParser myP(cpv);

  cout << "main::about to set configuration" << endl;
  FIPConfiguration cf(cpv);
  cf.readConfig("DetectorDescription/RegressionTest/test/configuration.xml");

  cout << "main::about to start parsing" << endl;
 
  myP.parse(cf);

  cout << "main::completed Parser" << endl;

  DDExpandedView exv(cpv);
  vector<DDTranslation> tvec;
  bool loop=true;
  std::cout << "Before the loop..." << std::endl;
  while(loop) {
    ROOT::Math::AxisAngle ra(exv.rotation());
    os << exv.logicalPart() << endl
       << "  " << exv.logicalPart().material() << endl
       << "  " << exv.logicalPart().solid() << endl
       << "  " << exv.translation() << endl;
    os << "  " << ra.Axis() << CONVERT_TO( ra.Angle(), deg ) << endl;
    tvec.emplace_back(exv.translation());   
    loop = exv.next();
  }
  
  vector<DDTranslation>::iterator it = tvec.begin();
  os << endl << "center points of all solids" << endl;
  for (; it != tvec.end(); ++it) {
    os << (*it).x() << " " << (*it).y() << " " << (*it).z() << endl;
  }
}

void testParser()
{
  try {
    cout << "main:: initialize" << endl;
    DDCompactView cpv;
    cout << "main::initialize DDL parser" << endl;
    DDLParser myP(cpv);

    cout << "main::about to set configuration" << endl;

    FIPConfiguration cf(cpv);
    cf.readConfig("DetectorDescription/RegressionTest/test/configuration.xml");

    cout << "main::about to start parsing" << endl;
 
    myP.parse(cf);

    cout << "main::completed Parser" << endl;
  
    cout << endl << endl << "main::Start checking!" << endl << endl;
  
  }
  catch (cms::Exception& e)
    {
      cout << "main::PROBLEM:" << endl 
	   << "         " << e << endl;
    }  
}

void printRot(const DDRotationMatrix & rot) {
  std::cout << "rot asis\n" << rot << std::endl;
  DD3Vector x,y,z;
  rot.GetComponents(x,y,z);
  std::cout << "components\n" 
	    << x << "\n"
	    << y << "\n"
	    << z << std::endl;
  cout << "phiX=" << x.phi() << " or in degrees = " 
       << CONVERT_TO( x.phi(), deg ) << endl;
  cout << "thetaX=" << x.theta() << " or in degrees = " 
       << CONVERT_TO( x.theta(), deg ) << endl;
  cout << "phiY=" << y.phi() << " or in degrees = " 
       << CONVERT_TO( y.phi(), deg ) << endl;
  cout << "thetaY=" << y.theta() << " or in degrees = " 
       << CONVERT_TO( y.theta(), deg ) << endl;
  cout << "phiZ=" << z.phi() << " or in degrees = " 
       << CONVERT_TO( z.phi(), deg ) << endl;
  cout << "thetaZ=" << z.theta() << " or in degrees = " 
       << CONVERT_TO( z.theta(), deg ) << endl;
  
  cout << "some factor/equations..." << endl;
  cout << " sin(thetaX()) * cos(phiX()) = " 
       << sin(x.theta()) * cos(x.phi()) << endl;
  
}

void testrot()
{
  {
    ROOT::Math::AxisAngle aa(DD3Vector(1.,1.,1.), 20._deg);
    DDRotationMatrix rm(aa); 
    cout << "DD3Vector was " << DD3Vector(1.,1.,1.) << " and the rotation was 20*deg around that axis." << endl;
    printRot(rm);
  }
  {
    DDRotationMatrix rm(1,0,0, 0,-1,0, 0,0,1); 
    cout << "(1,0,0, 0,-1,0, 0,0,1)" << endl;
    printRot(rm);
  }
}
