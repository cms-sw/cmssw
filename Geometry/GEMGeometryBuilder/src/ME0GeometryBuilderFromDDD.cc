/** Implementation of the ME0 Geometry Builder from DDD
 *
 *  \author Port of: MuDDDME0Builder (ORCA)
 *  \author M. Maggi - INFN Bari
 */
#include "Geometry/GEMGeometryBuilder/src/ME0GeometryBuilderFromDDD.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>

#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/ME0NumberingScheme.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>
#include <algorithm>
#include <boost/lexical_cast.hpp>

ME0GeometryBuilderFromDDD::ME0GeometryBuilderFromDDD(bool comp11) : theComp11Flag(comp11)
{ }

ME0GeometryBuilderFromDDD::~ME0GeometryBuilderFromDDD() 
{ }

ME0Geometry* ME0GeometryBuilderFromDDD::build(const DDCompactView* cview, const MuonDDDConstants& muonConstants)
{
  std::string attribute = "ReadOutName"; // could come from .orcarc
  std::string value     = "MuonME0Hits";    // could come from .orcarc
  DDValue val(attribute, value, 0.0);

  // Asking only for the MuonME0's
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

ME0Geometry* ME0GeometryBuilderFromDDD::buildGeometry(DDFilteredView& fview, const MuonDDDConstants& muonConstants)
{
  LogDebug("ME0GeometryBuilderFromDDD") <<"Building the geometry service";
  ME0Geometry* geometry = new ME0Geometry();

  LogDebug("ME0GeometryBuilderFromDDD") << "About to run through the ME0 structure\n" 
					<<" First logical part "
					<<fview.logicalPart().name().name();


  bool doSubDets = fview.firstChild();
 
  LogDebug("ME0GeometryBuilderFromDDD") << "doSubDets = " << doSubDets;

   LogDebug("ME0GeometryBuilderFromDDD") <<"start the loop"; 
  while (doSubDets)
  {
    // Get the Base Muon Number
    MuonDDDNumbering mdddnum(muonConstants);
    LogDebug("ME0GeometryBuilderFromDDD") <<"Getting the Muon base Number";
    MuonBaseNumber mbn = mdddnum.geoHistoryToBaseNumber(fview.geoHistory());

    LogDebug("ME0GeometryBuilderFromDDD") <<"Start the ME0 Numbering Schema";
    ME0NumberingScheme me0num(muonConstants);

    ME0DetId rollDetId(me0num.baseNumberToUnitNumber(mbn));
    LogDebug("ME0GeometryBuilderFromDDD") << "ME0 eta partition rawId: " << rollDetId.rawId() << ", detId: " << rollDetId;

    std::vector<double> dpar=fview.logicalPart().solid().parameters();
    std::string name = fview.logicalPart().name().name();
    DDTranslation tran = fview.translation();
    DDRotationMatrix rota = fview.rotation();
    Surface::PositionType pos(tran.x()/cm, tran.y()/cm, tran.z()/cm);
    // CLHEP way
    // Surface::RotationType rot(rota.xx(),rota.xy(),rota.xz(),
    //           	      rota.yx(),rota.yy(),rota.yz(),
    // 			      rota.zx(),rota.zy(),rota.zz());

    //ROOT::Math way
    DD3Vector x, y, z;
    rota.GetComponents(x,y,z);
    // doesn't this just re-inverse???
    Surface::RotationType rot(float(x.X()), float(x.Y()), float(x.Z()),
			      float(y.X()), float(y.Y()), float(y.Z()),
			      float(z.X()), float(z.Y()), float(z.Z())); 
    
    float be = dpar[4]/cm; // half bottom edge
    float te = dpar[8]/cm; // half top edge
    float ap = dpar[0]/cm; // half apothem
    float ti = 0.4/cm;     // half thickness

    //  TrapezoidalPlaneBounds* 
    Bounds* bounds = new TrapezoidalPlaneBounds(be, te, ap, ti);

    std::vector<float> pars;
    pars.push_back(be); 
    pars.push_back(te); 
    pars.push_back(ap); 
    //    pars.push_back(nStrips);
    // pars.push_back(nPads);

    LogDebug("ME0GeometryBuilderFromDDD") 
      << "ME0 " << name << " par " << be << " " << te << " " << ap << " " << dpar[0];
    
    ME0EtaPartitionSpecs* e_p_specs = new ME0EtaPartitionSpecs(GeomDetEnumerators::ME0, name, pars);

      //Change of axes for the forward
    Basic3DVector<float> newX(1.,0.,0.);
    Basic3DVector<float> newY(0.,0.,1.);
    //      if (tran.z() > 0. )
    newY *= -1;
    Basic3DVector<float> newZ(0.,1.,0.);
    rot.rotateAxes (newX, newY, newZ);
    
    BoundPlane* bp = new BoundPlane(pos, rot, bounds);
    ReferenceCountingPointer<BoundPlane> surf(bp);
    ME0EtaPartition* mep = new ME0EtaPartition(rollDetId, surf, e_p_specs);

    // Add the eta partition to the geometry
    geometry->add(mep);
    // go to next layer
    doSubDets = fview.nextSibling(); 
  }
  
  /*
  auto& partitions(geometry->etaPartitions());
  // build the chambers and add them to the geometry
  std::vector<ME0DetId> vDetId;
  vDetId.clear();
  int oldRollNumber = 1;
  for (unsigned i=1; i<=partitions.size(); ++i){
    ME0DetId detId(partitions.at(i-1)->id());
    const int rollNumber(detId.roll());
    // new batch of eta partitions --> new chamber
    if (rollNumber < oldRollNumber || i == partitions.size()) {
      // don't forget the last partition for the last chamber
      if (i == partitions.size()) vDetId.push_back(detId);

      ME0DetId fId(vDetId.front());
      ME0DetId chamberId(fId.chamberId());
      // compute the overall boundplane using the first eta partition
      const ME0EtaPartition* p(geometry->etaPartition(fId));
      const BoundPlane& bps = p->surface();
      BoundPlane* bp = const_cast<BoundPlane*>(&bps);
      ReferenceCountingPointer<BoundPlane> surf(bp);
      
      ME0Chamber* ch = new ME0Chamber(chamberId, surf); 
      LogDebug("ME0GeometryBuilderFromDDD")  << "Creating chamber " << chamberId << " with " << vDetId.size() << " eta partitions" << std::endl;
      
      for(auto id : vDetId){
	LogDebug("ME0GeometryBuilderFromDDD") << "Adding eta partition " << id << " to ME0 chamber" << std::endl;
	ch->add(const_cast<ME0EtaPartition*>(geometry->etaPartition(id)));
      }

      LogDebug("ME0GeometryBuilderFromDDD") << "Adding the chamber to the geometry" << std::endl;
      geometry->add(ch);
      vDetId.clear();
    }
    vDetId.push_back(detId);
    oldRollNumber = rollNumber;
  }
  
  auto& chambers(geometry->chambers());
  // construct super chambers
  for (unsigned i=0; i<chambers.size(); ++i){
    const BoundPlane& bps = chambers.at(i)->surface();
    BoundPlane* bp = const_cast<BoundPlane*>(&bps);
    ReferenceCountingPointer<BoundPlane> surf(bp);
    ME0DetId detIdL1(chambers.at(i)->id());
    if (detIdL1.layer()==2) continue;
    ME0DetId detIdL2(detIdL1.region(),detIdL1.ring(),detIdL1.station(),2,detIdL1.chamber(),0);
    auto ch2 = geometry->chamber(detIdL2);

    LogDebug("ME0GeometryBuilderFromDDD") << "First chamber for super chamber: " << detIdL1 << std::endl;
    LogDebug("ME0GeometryBuilderFromDDD") << "Second chamber for super chamber: " << detIdL2 << std::endl;

    LogDebug("ME0GeometryBuilderFromDDD") << "Creating new ME0 super chamber out of chambers." << std::endl;
    ME0SuperChamber* sch = new ME0SuperChamber(detIdL1, surf); 
    sch->add(const_cast<ME0Chamber*>(chambers.at(i)));
    sch->add(const_cast<ME0Chamber*>(ch2));

    LogDebug("ME0GeometryBuilderFromDDD") << "Adding the super chamber to the geometry." << std::endl;
    geometry->add(sch);
  }

  auto& superChambers(geometry->superChambers());
  // construct the regions, stations and rings. 
  std::cout << "maxStation " << maxStation << std::endl;
  for (int re = -1; re <= 1; re = re+2) {
    ME0Region* region = new ME0Region(re); 
    for (int st=1; st<=maxStation; ++st) {
      ME0Station* station = new ME0Station(re, st); 
      std::string name("ME+ std::to_string(re) + "/" + std::to_string(st));
      // Closest (furthest) super chambers in GE2/1 are called GE2/1s (GE2/1l)
      if (st==2) name = "GE" + std::to_string(re) + "/" + std::to_string(st) + "s";
      if (st==3) name = "GE" + std::to_string(re) + "/" + std::to_string(st-1) + "l";
      station->setName(name); 
      for (int ri=1; ri<=1; ++ri) {
	ME0Ring* ring = new ME0Ring(re, st, ri); 
	for (unsigned sch=0; sch<superChambers.size(); ++sch){
	  const ME0DetId detId(superChambers.at(sch)->id());
	  if (detId.region() != re || detId.station() != st || detId.ring() != ri) continue;
	  ring->add(superChambers.at(sch));
	  LogDebug("ME0GeometryBuilderFromDDD") << "Adding super chamber " << detId << " to ring: " 
						<< "re " << re << " st " << st << " ri " << ri << std::endl;
 	}
	LogDebug("ME0GeometryBuilderFromDDD") << "Adding ring " <<  ri << " to station " << "re " << re << " st " << st << std::endl;
	station->add(ring);
	geometry->add(ring);
      }
      LogDebug("ME0GeometryBuilderFromDDD") << "Adding station " << st << " to region " << re << std::endl;
      region->add(station);
      geometry->add(station);
    }
    LogDebug("ME0GeometryBuilderFromDDD") << "Adding region " << re << " to the geometry " << std::endl;
    geometry->add(region);
  }
  */
  return geometry;
}
