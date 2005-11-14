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
   DDRotationMatrix * r0  = new HepRotation();
   DDRotationMatrix * r30 = new HepRotation(Hep3Vector(0,0,1.),30.*deg);   
   DDRotationMatrix * r60 = new HepRotation(Hep3Vector(0,0,1.),60.*deg);   
   DDRotationMatrix * r90 = new HepRotation(Hep3Vector(0,0,1.),90.*deg);   
   
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



   DDRotationMatrix * rm = new HepRotation(Hep3Vector(1.,1.,1.),20.*deg); 
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
    os << exv.logicalPart() << endl
       << "  " << exv.logicalPart().material() << endl
       << "  " << exv.logicalPart().solid() << endl
       << "  " << exv.translation() << endl
       << "  " << exv.rotation().axis() << exv.rotation().delta()/deg << endl; 
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
#include "DetectorDescription/Algorithm/src/AlgoInit.h"
void testParser()
{
  try {
    cout << "main:: initialize" << endl;
    AlgoInit();

    cout << "main::initialize DDL parser" << endl;
    DDLParser* myP = DDLParser::instance();

    cout << "main::about to set configuration" << endl;
    myP->SetConfig("configuration.xml");

    cout << "main::about to start parsing" << endl;
 
    myP->StartParsing();

    cout << "main::completed Parser" << endl;
  
    cout << endl << endl << "main::Start checking!" << endl << endl;
  
  }
  catch (DDException& e)
    {
      cout << "main::PROBLEM:" << endl 
	   << "         " << e << endl;
    }  
}

void testrot()
{
  //  ExprEvalInterface & eval = ExprEval::instance();
  DDRotationMatrix * rm = new HepRotation(Hep3Vector(1.,1.,1.),20.*deg); 
  cout << "Hep3Vector was " << Hep3Vector(1.,1.,1.) << " and the rotation was 20*deg around that axis." << endl;
  cout << "phiX=" << rm->phiX() << " or in degrees = " 
       << rm->phiX()/deg << endl;
  cout << "thetaX=" << rm->thetaX() << " or in degrees = " 
       << rm->thetaX()/deg << endl;
  cout << "phiY=" << rm->phiY() << " or in degrees = " 
       << rm->phiY()/deg << endl;
  cout << "thetaY=" << rm->thetaY() << " or in degrees = " 
       << rm->thetaY()/deg << endl;
  cout << "phiZ=" << rm->phiZ() << " or in degrees = " 
       << rm->phiZ()/deg << endl;
  cout << "thetaZ=" << rm->thetaZ() << " or in degrees = " 
       << rm->thetaZ()/deg << endl;
  
  cout << "some factor/equations..." << endl;
  cout << " sin(thetaX()) * cos(phiX()) = " 
       << sin(rm->thetaX()) * cos(rm->phiX()) << endl;
}
