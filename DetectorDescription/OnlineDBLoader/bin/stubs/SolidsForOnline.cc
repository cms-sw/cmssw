#include <memory>

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include <DetectorDescription/Core/interface/DDCompactView.h>
#include <DetectorDescription/Core/interface/DDValue.h>
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

#include <Geometry/Records/interface/IdealGeometryRecord.h>

#include <iostream>
#include <istream>
#include <fstream>
#include <string>


class SolidsForOnline : public edm::EDAnalyzer {

public:
 
  explicit SolidsForOnline( const edm::ParameterSet& );
  ~SolidsForOnline();
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  virtual void beginRun( const edm::Run&, const edm::EventSetup& );
  
private: 

  std::string filename_;  
};

SolidsForOnline::SolidsForOnline( const edm::ParameterSet& iConfig ) { }


SolidsForOnline::~SolidsForOnline() { }

void SolidsForOnline::analyze( const edm::Event& iEvent,  const edm::EventSetup& iSetup ) { 
  std::cout << "analyze does nothing" << std::endl;
}

void SolidsForOnline::beginRun( const edm::Run&, const edm::EventSetup& iSetup ) {

// TRD1 and Trapezoid can be in the same files.

  std::string solidsFileName("SOLIDS.dat"); //added by Jie Chen
  std::string boxFileName("BOXES.dat");

  std::string tubesFileName("TUBES.dat");
  std::string polyHedraFileName("POLYHEDRAS.dat");
  std::string polyConeFileName("POLYCONES.dat");
  std::string conesFileName("CONES.dat");
  std::string pseudoTrapFileName("PSEUDOTRAPEZOIDS.dat");
  std::string trapFileName("TRAPEZOIDS.dat");
  std::string boolSolidsFileName("BOOLEANSOLIDS.dat");
  std::string reflectionSolidsFileName("REFLECTIONSOLIDS.dat");
  std::string torusFileName("TORUS.dat");

  std::ofstream solidsOS(solidsFileName.c_str());
  std::ofstream boxOS(boxFileName.c_str());
  std::ofstream tubeOS(tubesFileName.c_str());
  std::ofstream polyHOS(polyHedraFileName.c_str());
  std::ofstream polyCOS(polyConeFileName.c_str());
  std::ofstream coneOS(conesFileName.c_str());
  std::ofstream ptrapOS(pseudoTrapFileName.c_str());
  std::ofstream trapOS(trapFileName.c_str());
  std::ofstream boolOS(boolSolidsFileName.c_str());
  std::ofstream reflectionOS(reflectionSolidsFileName.c_str());
  std::ofstream torusOS(torusFileName.c_str());

  std::cout << "SolidsForOnline Analyzer..." << std::endl;

  edm::ESTransientHandle<DDCompactView> pDD;

  iSetup.get<IdealGeometryRecord>().get( "", pDD );

  const DDCompactView & cpv = *pDD;
  DDCompactView::graph_type gra = cpv.graph();

  DDSolid::iterator<DDSolid> sit(DDSolid::begin()), sed(DDSolid::end());
  for (; sit != sed; ++sit) {
    if (! sit->isDefined().second) continue;  
    const DDSolid& solid = *sit;
    ///
    solidsOS<<solid.name()<<","<<DDSolidShapesName::name(solid.shape())
	    << std::endl;
    

    switch (solid.shape()) 
      {
      case ddunion:
	{
	  //for bool solid
	  DDBooleanSolid boolSolid(solid);
	  boolOS << boolSolid.name() << "," ;
	  boolOS << "U";
	  boolOS << "," << boolSolid.translation().x()
		 << "," << boolSolid.translation().y()
		 << "," << boolSolid.translation().z()
		 << "," << boolSolid.solidA().name() 
		 << "," << boolSolid.solidB().name()
		 << "," << boolSolid.rotation().name()
		 << std::endl;

	  break;
	}
      case ddsubtraction:
	{
	  DDBooleanSolid boolSolid(solid);
	  boolOS << boolSolid.name() << "," ;
	  boolOS << "S";
	  boolOS << "," << boolSolid.translation().x()
		 << "," << boolSolid.translation().y()
		 << "," << boolSolid.translation().z()
		 << "," << boolSolid.solidA().name() 
		 << "," << boolSolid.solidB().name()
		 << "," << boolSolid.rotation().name()
		 << std::endl;
	  break;
	}
      case ddintersection: 
	{      	  

	  //for bool solid
	  DDBooleanSolid boolSolid(solid);
	  // if translation is == identity there are no parameters.
	  // if there is no rotation the name will be ":"
	  //std::string rotName = boolSolid.rotation().toString();
	  //if (rotName == ":") {
	  //  rotName = "rotations:UNIT";
	  //}
	  boolOS << boolSolid.name() << "," ;
	  boolOS << "I";
	  boolOS << "," << boolSolid.translation().x()
		 << "," << boolSolid.translation().y()
		 << "," << boolSolid.translation().z()
		 << "," << boolSolid.solidA().name() 
		 << "," << boolSolid.solidB().name()
		 << "," << boolSolid.rotation().name()
		 << std::endl;
	  
	  break;
	  
	}
      case ddreflected:
	{ 
	  DDReflectionSolid reflection(solid);
	  reflectionOS<<reflection.name()<<","
		      <<reflection.unreflected().name()
		      <<std::endl;
	  break;
	}
      case ddbox:
	{
	  DDBox box(solid);
	  //std::cout<<"box shape is"<<solid.shape()<<std::endl;
	  boxOS<<box.name()<<",";
	  boxOS<<2.0*box.halfX()<<","<<2.0*box.halfY()<<","<<2.0*box.halfZ()<<std::endl;
	  break;
	}
	
      case ddpseudotrap:
	{
	  DDPseudoTrap pseudoTrap(solid);
	  
	  ptrapOS<<pseudoTrap.name() <<",";
	  ptrapOS<<pseudoTrap.x1() <<","<<pseudoTrap.x2() <<","
		 <<pseudoTrap.halfZ()*2. <<","<<pseudoTrap.y1() <<","
		 <<pseudoTrap.y2() <<","<<pseudoTrap.radius()<<","
		 <<pseudoTrap.atMinusZ()
		 << std::endl;
	  
	  break;
	}
	
      case ddtubs:
	{
	  //same as ddtrunctubs, Tube element is simply a Tube Section 
	  //with the angle going from zero to 360 degrees. 
	  //cutAtStart, cutAtDelta,  and  cutInside. are all zero 
	  //then they are tubes
	  DDTubs tubs(solid);
	  tubeOS<<tubs.name()<<","<<tubs.rIn()<<","
		<<tubs.rOut()<<","<<tubs.zhalf()*2.0<<","
		<<tubs.startPhi()<<","<<tubs.deltaPhi()<<"," 
		<<"0"<<","<<"0"<<","
		<<"0"
		<<std::endl;

	  break;
	}
      case ddtrap: // trd1s is included into this case
	{    
	  DDTrap trap(solid);	  
	  trapOS<<trap.name()<<",";
	  trapOS<<trap.alpha1()<<","<<trap.alpha2()<<","
		<<trap.x1()<<","<<trap.x2()<<","
		<<trap.halfZ()*2.0<<","<<trap.y1()<<","
		<<trap.y2()<<","
		<<trap.phi()<<","<<trap.theta()<<","
		<<trap.x3()<<","<<trap.x4()
		<<std::endl;
	  break;
	}
      case ddcons:
	{
	  DDCons cons(solid);
	  coneOS<<cons.name()<<","<<cons.zhalf()*2.0<<","
		<<cons.rInMinusZ()<<","
		<<cons.rOutMinusZ()<<","<<cons.rInPlusZ()<<","
		<<cons.rOutPlusZ()<<","<<cons.phiFrom()<<","
		<<cons.deltaPhi()
		<<std::endl;
	  
	  break;
	}

      case ddpolycone_rz:
	{
	  //rz_zs=rz if it's ddpolycone_rz
	  DDPolycone polyCone(solid);
	  polyCOS<<polyCone.name()<<","<<polyCone.startPhi()
		 <<","<<polyCone.deltaPhi()<<","<<"RZ"
		 <<std::endl;
	  break;
	}
      case ddpolyhedra_rz:
	{
	  DDPolyhedra  polyHedra(solid);
	  polyHOS<<polyHedra.name()<<",";
	  polyHOS<<polyHedra.sides()<<","<<polyHedra.startPhi()
		 <<","<<polyHedra.deltaPhi()<<","<<"RZ"
		 <<std::endl;
	  break;
	}
      case ddpolycone_rrz:{
	  //rz_zs=zs if it's ddpolycone_rrz
	  DDPolycone polyCone(solid);
	  polyCOS<<polyCone.name()<<","<<polyCone.startPhi()
		 <<","<<polyCone.deltaPhi()<<","<<"ZS"
		 <<std::endl;	
	break;
      }
      case ddpolyhedra_rrz:
	{

	  DDPolyhedra  polyHedra(solid);
	  polyHOS<<polyHedra.name()<<",";
	  polyHOS<<polyHedra.sides()<<","<<polyHedra.startPhi()
		 <<","<<polyHedra.deltaPhi()<<","<<"ZS"
		 <<std::endl;
	  break;
	}
       
      case ddtrunctubs:{

	  DDTruncTubs tubs(solid);
	  tubeOS<<tubs.name()<<","<<tubs.rIn()<<","
		<<tubs.rOut()<<","<<tubs.zHalf()*2.0<<","
		<<tubs.startPhi()<<","<<tubs.deltaPhi()<<"," 
		<<tubs.cutAtStart()<<","<<tubs.cutAtDelta()<<","
		<<tubs.cutInside()
		<<std::endl;
	  break;
      }

      case ddtorus: {
	DDTorus torus(solid);
	torusOS<<torus.name()<<","<<torus.rMin()<<","
	       <<torus.rMax()<<","<<torus.rTorus()<<","
	       <<torus.startPhi()<<","<<torus.deltaPhi()
	       <<std::endl;
	break;
      }

      case ddshapeless:{
	//	DDShapelessSolid shapeless(solid);
	//shapelessOS<<shapeless.name()
	// <<std::endl;

// 	return new PSolid( solid.toString(), solid.parameters()
// 			   , solid.shape() );
	break;
      }      
      case dd_not_init:
      default:
	throw cms::Exception("DDException") << "DDDToPersFactory::solid(...) either not inited or no such solid.";
	break;
      }
  

  }
  solidsOS.close();
  boxOS.close();

  tubeOS.close();
  polyHOS.close();
  polyCOS.close();
  coneOS.close();
  ptrapOS.close();
  trapOS.close();
  boolOS.close();
  reflectionOS.close();
}

//define this as a plug-in
DEFINE_FWK_MODULE(SolidsForOnline);
