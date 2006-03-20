/** Implementation of the RPC Geometry Builder from DDD
 *
 *  \author Port of: MuDDDRPCBuilder (ORCA)
 *  \author M. Maggi - INFN Bari
 */
#include "Geometry/RPCGeometryBuilder/src/RPCGeometryBuilderFromDDD.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
//#include "Geometry/RPCSimAlgo/interface/RPCChamber.h"

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>

#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/RPCNumberingScheme.h"

#include "Geometry/Surface/interface/RectangularPlaneBounds.h"
#include "Geometry/Surface/interface/TrapezoidalPlaneBounds.h"

#include "CLHEP/Units/SystemOfUnits.h"

#include <iostream>
#include <algorithm>

RPCGeometryBuilderFromDDD::RPCGeometryBuilderFromDDD()
{ }

RPCGeometryBuilderFromDDD::~RPCGeometryBuilderFromDDD() 
{ }

RPCGeometry* RPCGeometryBuilderFromDDD::build(const DDCompactView* cview)
{

  try {


    std::string attribute = "ReadOutName"; // could come from .orcarc
    std::string value     = "MuonRPCHits";    // could come from .orcarc
    DDValue val(attribute, value, 0.0);

    // Asking only for the MuonRPC's
    DDSpecificsFilter filter;
    filter.setCriteria(val, // name & value of a variable 
		       DDSpecificsFilter::matches,
		       DDSpecificsFilter::AND, 
		       true, // compare strings otherwise doubles
		       true // use merged-specifics or simple-specifics
		       );
    DDFilteredView fview(*cview);
    fview.addFilter(filter);

    return this->buildGeometry(fview);
  }
  catch (const DDException & e ) {
    std::cerr <<"RPCGeometryBuilderFromDDD::build() : "
	      <<"DDD Exception: something went wrong during XML parsing!" 
	      << std::endl
	      << "  Message: " << e << std::endl
	      << "  Terminating execution ... " << std::endl;
    throw;
  }
  catch (const cms::Exception& e){  
    std::cerr <<"RPCGeometryBuilderFromDDD::build() : "
	      <<"an unexpected exception occured: " 
	      << e << std::endl;   
    throw;
  }
  catch (const std::exception& e) {
    std::cerr <<"RPCGeometryBuilderFromDDD::build() : "
	      <<"an unexpected exception occured: " 
	      << e.what() << std::endl; 
    throw;
  }
  catch (...) {
    std::cerr <<"RPCGeometryBuilderFromDDD::build() : "
	      <<"An unexpected exception occured!" << std::endl
	      << "  Terminating execution ... " << std::endl;
    std::unexpected();           
  }
}



RPCGeometry* RPCGeometryBuilderFromDDD::buildGeometry(DDFilteredView& fview)
{
#ifdef DEBUG  
  std::cout <<"Building the geometry service"<<std::endl;
#endif
  RPCGeometry* geometry = new RPCGeometry();
  

#ifdef DEBUG  
  std::cout << "About to run through the RPC structure" << std::endl;
  std::cout <<" First logical part "
  	    <<fview.logicalPart().name().name()<<std::endl;
#endif
  bool doSubDets = fview.firstChild();

#ifdef DEBUG  
  std::cout << "doSubDets = " << doSubDets << std::endl;
#endif
  while (doSubDets){

#ifdef DEBUG  
    std::cout <<"start the loop"<<std::endl; 
#endif

    // Get the Base Muon Number
    MuonDDDNumbering mdddnum;
#ifdef DEBUG  
    std::cout <<"Getting the Muon base Number"<<std::endl;
#endif
    MuonBaseNumber   mbn=mdddnum.geoHistoryToBaseNumber(fview.geoHistory());

#ifdef DEBUG  
    std::cout <<"Start the Rpc Numbering Schema"<<std::endl;
#endif
    // Get the The Rpc det Id 
    RPCNumberingScheme rpcnum;
    int detid = 0;

#ifdef DEBUG  
    std::cout <<"Getting the Unit Number"<<std::endl;
#endif
    detid = rpcnum.baseNumberToUnitNumber(mbn);
#ifdef DEBUG  
    std::cout <<"Getting the RPC det Id "<<detid <<std::endl;
#endif
    RPCDetId rpcid(detid);
    //    rpcid.buildfromTrIndex(detid);
#ifdef DEBUG  
    std::cout <<"The RPCDEtid is "<<detid<<std::endl;
#endif

    DDValue numbOfStrips("nStrips");

    std::vector<const DDsvalues_type* > specs(fview.specifics());
    std::vector<const DDsvalues_type* >::iterator is=specs.begin();
    int nStrips=0;
    for (;is!=specs.end(); is++){
      if (DDfetch( *is, numbOfStrips)){
	nStrips=int(numbOfStrips.doubles()[0]);	
      }
    }
#ifdef DEBUG  
    if (nStrips == 0 )
      std::cout <<"No strip found!!"<<std::endl;
#endif
    
    std::vector<double> dpar=fview.logicalPart().solid().parameters();
    std::vector<float> pars;
    RPCRollSpecs* rollspecs= 0;
    Bounds* bounds = 0;

    if (dpar.size()==3){
      float width     = dpar[0]/cm;
      float length    = dpar[1]/cm;
      float thickness = dpar[2]/cm;
      //RectangularPlaneBounds* 
      bounds = 
	new RectangularPlaneBounds(width,length,thickness);
      pars.push_back(dpar[0]);
      pars.push_back(dpar[1]);
      pars.push_back(numbOfStrips.doubles()[0]); //h/2;
      rollspecs = new RPCRollSpecs(GeomDetType::RPCBarrel,pars);

#ifdef DEBUG  
      std::cout <<"Barrel "<<fview.logicalPart().name().name()
		<<" par "<<dpar[0]
		<<" "<<dpar[1]<<" "<<dpar[2];
#endif
    }else{
      float be = dpar[4]/cm;
      float te = dpar[8]/cm;
      float ap = dpar[0]/cm;
      float ti = 0.4/cm;
      //  TrapezoidalPlaneBounds* 
      bounds = 
	new TrapezoidalPlaneBounds(be,te,ap,ti);
      pars.push_back(dpar[4]); //b/2;
      pars.push_back(dpar[8]); //B/2;
      pars.push_back(dpar[0]); //h/2;
      pars.push_back(numbOfStrips.doubles()[0]); //h/2;
      
#ifdef DEBUG  
      std::cout <<"Forward "<<fview.logicalPart().name().name()<<" par "<<dpar[4]
		<<" "<<dpar[8]<<" "<<dpar[3]<<" "
		<<dpar[0];
#endif      

      rollspecs = new RPCRollSpecs(GeomDetType::RPCEndcap,pars);
    }
#ifdef DEBUG  
    std::cout <<"   Number of strips "<<nStrips<<std::endl;
#endif  

    DDTranslation tran    = fview.translation();
    DDRotationMatrix rota = fview.rotation().inverse();

    Surface::PositionType pos(tran.x()/cm,tran.y()/cm, tran.z()/cm);
    Surface::RotationType rot(rota.xx(),rota.xy(),rota.xz(),
			      rota.yx(),rota.yy(),rota.yz(),
			      rota.zx(),rota.zy(),rota.zz());

    BoundPlane* bp = new BoundPlane(pos,rot,bounds);

    

    RPCRoll* r=new RPCRoll(bp,rollspecs,rpcid);
    geometry->add(r);
    
    doSubDets = fview.nextSibling(); // go to next layer
  }
  return geometry;
}
