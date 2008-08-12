using namespace std;

#include <cmath>
#include "CLHEP/Units/SystemOfUnits.h"

#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

#include <Math/RotationX.h>
#include <Math/RotationY.h>
#include <Math/RotationZ.h>
#include <Math/AxisAngle.h>

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
void regressionTest_setup() {
   ExprEvalInterface & eval = ExprEval::instance();
   
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
   DDRotationMatrix * r0  = new DDRotationMatrix();
   DDRotationMatrix * r30 = new DDRotationMatrix(ROOT::Math::RotationZ(30.*deg));   
   DDRotationMatrix * r60 = new DDRotationMatrix(ROOT::Math::RotationZ(60.*deg));   
   DDRotationMatrix * r90 = new DDRotationMatrix(ROOT::Math::RotationZ(90.*deg));   
   
   DDrot(DDName("Unit",ns),r0);
   DDrot(DDName("R30",ns),r30);
   DDrot(DDName("R60",ns),r60);
   DDrot(DDName("R90",ns),r90);
   
   DDSolid collectorSolid = DDSolidFactory::shapeless(DDName("group",ns));
   
   DDRootDef::instance().set(worldName);	      
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
void regressionTest_first() {
   ExprEvalInterface & eval = ExprEval::instance();
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

   DDpos(DDName("sensor",ns), // child
         DDName("support",ns), // parent
	 "1",
	 DDTranslation(eval.eval(ns,"[setup:corner]/8."),
	               eval.eval(ns,"[setup:corner]/16."),
		       eval.eval(ns,"[setup:corner]/8.")
		       ),
	 DDRotation(DDName("Unit","setup"))
	);
	
   
   DDLogicalPart part(DDName("group",ns),
                 DDName("Air","setup"),
		 DDName("group","setup")    
                );
 		
		
   DDpos(DDName("support",ns),
         DDName("group",ns),
	 "1",
	 DDTranslation(eval.eval(ns,"[setup:corner]*1.25*cos(0.)"),
	               eval.eval(ns,"[setup:corner]*1.25*sin(0.)"),
		       eval.eval(ns,"0.")),
	 DDRotation(DDName("Unit","setup"))
	 );

   DDpos(DDName("support",ns),
         DDName("group",ns),
	 "2",
	 DDTranslation(eval.eval(ns,"[setup:corner]*1.25*cos(30.*deg)"),
	               eval.eval(ns,"[setup:corner]*1.25*sin(30.*deg)"),
		       eval.eval(ns,"0.")),
	 DDRotation(DDName("R30","setup"))
	 );

   DDpos(DDName("support",ns),
         DDName("group",ns),
	 "3",
	 DDTranslation(eval.eval(ns,"[setup:corner]*1.25*cos(60.*deg)"),
	               eval.eval(ns,"[setup:corner]*1.25*sin(60.*deg)"),
		       eval.eval(ns,"0.")),
	 DDRotation(DDName("R60","setup"))
	 );

   DDpos(DDName("support",ns),
         DDName("group",ns),
	 "4",
	 DDTranslation(eval.eval(ns,"[setup:corner]*1.25*cos(90.*deg)"),
	               eval.eval(ns,"[setup:corner]*1.25*sin(90.*deg)"),
		       eval.eval(ns,"0.")),
	 DDRotation(DDName("R90","setup"))
	 );



   DDRotationMatrix * rm = new DDRotationMatrix(ROOT::Math::AxisAngle(DD3Vector(1.,1.,1.),20.*deg)); 
   DDpos(DDName("group",ns),
         DDName("world","setup"),
	 "1",
	 DDTranslation(),
	 DDrot(DDName("group",ns), rm)
	 );				  
   			  		   
}


#include <iostream>
#include <fstream>
#include <vector>
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
void output(string filename) 
{
  ostream & os(cout);

  os << "Starting Regressiontest Output" << endl;
  DDCompactView cpv;
  DDExpandedView exv(cpv);
  vector<DDTranslation> tvec;
  bool loop=true;
  while(loop) {
    ROOT::Math::AxisAngle ra(exv.rotation());
    os << exv.logicalPart() << endl
       << "  " << exv.logicalPart().material() << endl
       << "  " << exv.logicalPart().solid() << endl
       << "  " << exv.translation() << endl;
    os << "  " << ra.Axis() << ra.Angle()/deg << endl;
    tvec.push_back(exv.translation());   
    loop = exv.next();
  }
  
  vector<DDTranslation>::iterator it = tvec.begin();
  os << endl << "center points of all solids" << endl;
  for (; it != tvec.end(); ++it) {
    os << (*it).x() << " " << (*it).y() << " " << (*it).z() << endl;
  }
}

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/DDLConfiguration.h"
#include "DetectorDescription/Algorithm/src/AlgoInit.h"
void testParser()
{
  try {
    cout << "main:: initialize" << endl;
    AlgoInit();

    cout << "main::initialize DDL parser" << endl;
    DDLParser* myP = DDLParser::instance();

    cout << "main::about to set configuration" << endl;
    //    myP->SetConfig("configuration.xml");
    DDLConfiguration cf;
    cf.readConfig("configuration.xml");

    cout << "main::about to start parsing" << endl;
 
    myP->parse(cf);

    cout << "main::completed Parser" << endl;
  
    cout << endl << endl << "main::Start checking!" << endl << endl;
  
  }
  catch (DDException& e)
    {
      cout << "main::PROBLEM:" << endl 
	   << "         " << e << endl;
    }  
}

void printRot(const DDRotationMatrix & rot) {
  std::cout << "rot asis\n" << rot << std::endl;
  DD3Vector x,y,z; const_cast<DDRotationMatrix &>(rot).GetComponents(x,y,z);
  std::cout << "components\n" 
	    << x << "\n"
	    << y << "\n"
	    << z << std::endl;
  cout << "phiX=" << x.phi() << " or in degrees = " 
       << x.phi()/deg << endl;
  cout << "thetaX=" << x.theta() << " or in degrees = " 
       << x.theta()/deg << endl;
  cout << "phiY=" << y.phi() << " or in degrees = " 
       << y.phi()/deg << endl;
  cout << "thetaY=" << y.theta() << " or in degrees = " 
       << y.theta()/deg << endl;
  cout << "phiZ=" << z.phi() << " or in degrees = " 
       << z.phi()/deg << endl;
  cout << "thetaZ=" << z.theta() << " or in degrees = " 
       << z.theta()/deg << endl;
  
  cout << "some factor/equations..." << endl;
  cout << " sin(thetaX()) * cos(phiX()) = " 
       << sin(x.theta()) * cos(x.phi()) << endl;
  
}

void testrot()
{
  //  ExprEvalInterface & eval = ExprEval::instance();
  {
    ROOT::Math::AxisAngle aa(DD3Vector(1.,1.,1.), 20.*deg);
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
