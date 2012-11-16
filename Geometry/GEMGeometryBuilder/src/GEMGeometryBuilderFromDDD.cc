/** Implementation of the GEM Geometry Builder from DDD
 *
 *  \author Port of: MuDDDGEMBuilder (ORCA)
 *  \author M. Maggi - INFN Bari
 */
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryBuilderFromDDD.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>

#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/GEMNumberingScheme.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>
#include <algorithm>

GEMGeometryBuilderFromDDD::GEMGeometryBuilderFromDDD(bool comp11) : theComp11Flag(comp11)
{ }

GEMGeometryBuilderFromDDD::~GEMGeometryBuilderFromDDD() 
{ }

GEMGeometry* GEMGeometryBuilderFromDDD::build(const DDCompactView* cview, const MuonDDDConstants& muonConstants)
{
  std::string attribute = "ReadOutName"; // could come from .orcarc
  std::string value     = "MuonGEMHits";    // could come from .orcarc
  DDValue val(attribute, value, 0.0);

  // Asking only for the MuonGEM's
  DDSpecificsFilter filter;
  filter.setCriteria(val, // name & value of a variable 
		     DDSpecificsFilter::matches,
		     DDSpecificsFilter::AND, 
		     true, // compare strings otherwise doubles
		     true // use merged-specifics or simple-specifics
		     );
  DDFilteredView fview(*cview);
  fview.addFilter(filter);

  return this->buildGeometry(fview, muonConstants);
}

GEMGeometry* GEMGeometryBuilderFromDDD::buildGeometry(DDFilteredView& fview, const MuonDDDConstants& muonConstants)
{
  std::cout <<" GEM Builder Start "<<std::endl;
  std::cout << "GEMGeometryBuilderFromDDD" <<"Building the geometry service"<<std::endl;
  GEMGeometry* geometry = new GEMGeometry();

  std::cout << "GEMGeometryBuilderFromDDD" << "About to run through the GEM structure\n" 
	    <<" First logical part "
	    <<fview.logicalPart().name().name()<<std::endl;
  bool doSubDets = fview.firstChild();
  
  std::cout << "GEMGeometryBuilderFromDDD" << " doSubDets = " << doSubDets<<std::endl;
  while (doSubDets){
    std::cout << "GEMGeometryBuilderFromDDD" <<"start the loop"<<std::endl;

    // Get the Base Muon Number
    MuonDDDNumbering mdddnum(muonConstants);
    std::cout << "GEMGeometryBuilderFromDDD" <<"Getting the Muon base Number"<<std::endl;
    MuonBaseNumber   mbn=mdddnum.geoHistoryToBaseNumber(fview.geoHistory());
    std::cout << "GEMGeometryBuilderFromDDD" <<"Start the GEM Numbering Schema"<<std::endl;
    // Get the The GEM det Id 
    GEMNumberingScheme gemnum(muonConstants);
    int detid = 0;

    std::cout << "GEMGeometryBuilderFromDDD" <<"Getting the Unit Number"<<std::endl;
    detid = gemnum.baseNumberToUnitNumber(mbn);
    std::cout << "GEMGeometryBuilderFromDDD" <<"Getting the GEM det Id "<<detid<<std::endl;

    GEMDetId gemmid(detid);
    //    GEMDetId chid(gemmid.region(),gemmid.ring(),gemmid.station(),gemmid.sector(),gemmid.layer(),gemmid.subsector(),0);

    std::cout << "GEMGeometryBuilderFromDDD" <<"The GEMDetid is "<<gemmid<<std::endl;

    DDValue numbOfStrips("nStrips");

    std::vector<const DDsvalues_type* > specs(fview.specifics());
    std::vector<const DDsvalues_type* >::iterator is=specs.begin();
    int nStrips=0;
    for (;is!=specs.end(); is++){
      if (DDfetch( *is, numbOfStrips)){
	nStrips=int(numbOfStrips.doubles()[0]);	
      }
    }

    std::cout << "GEMGeometryBuilderFromDDD" << ((nStrips == 0 ) ? ("No strip found!!") : (""))<<std::endl;
    
    std::vector<double> dpar=fview.logicalPart().solid().parameters();
    std::string name=fview.logicalPart().name().name();
    DDTranslation tran    = fview.translation();
    //removed .Inverse after comparing to DT...
    DDRotationMatrix rota = fview.rotation();//.Inverse();
    Surface::PositionType pos(tran.x()/cm,tran.y()/cm, tran.z()/cm);
    // CLHEP way
//     Surface::RotationType rot(rota.xx(),rota.xy(),rota.xz(),
// 			      rota.yx(),rota.yy(),rota.yz(),
// 			      rota.zx(),rota.zy(),rota.zz());

//ROOT::Math way
    DD3Vector x, y, z;
    rota.GetComponents(x,y,z);
    // doesn't this just re-inverse???
    Surface::RotationType rot (float(x.X()),float(x.Y()),float(x.Z()),
			       float(y.X()),float(y.Y()),float(y.Z()),
			       float(z.X()),float(z.Y()),float(z.Z())); 
    
    std::vector<float> pars;
    GEMEtaPartitionSpecs* etapartitionspecs= 0;
    Bounds* bounds = 0;

    float be = dpar[4]/cm;
    float te = dpar[8]/cm;
    float ap = dpar[0]/cm;
    float ti = 0.4/cm;
    //  TrapezoidalPlaneBounds* 
    bounds = 
      new TrapezoidalPlaneBounds(be,te,ap,ti);
    pars.push_back(dpar[4]/cm); //b/2;
    pars.push_back(dpar[8]/cm); //B/2;
    pars.push_back(dpar[0]/cm); //h/2;
    pars.push_back(numbOfStrips.doubles()[0]); //h/2;      
    std::cout << "GEMGeometryBuilderFromDDD" <<"GEM "<<name
					  <<" par "<<dpar[4]/cm
					  <<" "<<dpar[8]/cm<<" "<<dpar[3]/cm<<" "
					  <<dpar[0]<<std::endl;
    
    etapartitionspecs = new GEMEtaPartitionSpecs(GeomDetEnumerators::GEM,name,pars);

      //Change of axes for the forward
    Basic3DVector<float> newX(1.,0.,0.);
    Basic3DVector<float> newY(0.,0.,1.);
    //      if (tran.z() > 0. )
    newY *= -1;
    Basic3DVector<float> newZ(0.,1.,0.);
    rot.rotateAxes (newX, newY,newZ);
    
    std::cout << "GEMGeometryBuilderFromDDD" <<"   Number of strips "<<nStrips<<std::endl;
    BoundPlane* bp = new BoundPlane(pos,rot,bounds);
    ReferenceCountingPointer<BoundPlane> surf(bp);
    GEMEtaPartition* gep=new GEMEtaPartition(gemmid,surf,etapartitionspecs);
    geometry->add(gep);

    std::list<GEMEtaPartition *> gepls;
    /*
    if (chids.find(chid)!=chids.end()){
      gepls = chids[chid];
    }
    */
    gepls.push_back(gep);
    //chids[chid]=gepls;
    
    doSubDets = fview.nextSibling(); // go to next layer
  }
  /*
  // Create the GEMChambers and store them on the Geometry 
  for( std::map<GEMDetId, std::list<GEMEtaPartition *> >::iterator ich=chids.begin();
       ich != chids.end(); ich++){
    GEMDetId chid = ich->first;
    std::list<GEMEtaPartition * > gepls = ich->second;5D
    
    // compute the overall boundplane. At the moment we use just the last
    // surface
    BoundPlane* bp=0;
    for(std::list<GEMEtaPartition *>::iterator gepl=gepls.begin();
	gepl!=gepls.end(); gepl++){
      const BoundPlane& bps = (*gepl)->surface();
      bp = const_cast<BoundPlane *>(&bps);
    }
    
    ReferenceCountingPointer<BoundPlane> surf(bp);
    // Create the chamber 
    GEMChamber* ch = new GEMChamber (chid, surf); 
    // Add the etapartitions to the chamber
    for(std::list<GEMEtaPartition *>::iterator gepl=gepls.begin();
	gepl!=gepls.end(); rl++){
      ch->add(*gepl);
    }
    // Add the chamber to the geometry
    geometry->add(ch);
  } 
  */
  return geometry;
}
