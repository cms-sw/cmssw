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
#include <boost/lexical_cast.hpp>

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
  std::cout << "Building the geometry service" << std::endl;
  LogDebug("GEMGeometryBuilderFromDDD") <<"Building the geometry service";
  GEMGeometry* geometry = new GEMGeometry();

  LogDebug("GEMGeometryBuilderFromDDD") << "About to run through the GEM structure\n" 
					<<" First logical part "
					<<fview.logicalPart().name().name();
  bool doSubDets = fview.firstChild();
  LogDebug("GEMGeometryBuilderFromDDD") << "doSubDets = " << doSubDets;

  LogDebug("GEMGeometryBuilderFromDDD") <<"start the loop"; 
  int nChambers(0);
  int maxStation(1);
  while (doSubDets)
  {
    // Get the Base Muon Number
    MuonDDDNumbering mdddnum(muonConstants);
    LogDebug("GEMGeometryBuilderFromDDD") <<"Getting the Muon base Number";
    MuonBaseNumber mbn = mdddnum.geoHistoryToBaseNumber(fview.geoHistory());

    LogDebug("GEMGeometryBuilderFromDDD") <<"Start the GEM Numbering Schema";
    GEMNumberingScheme gemnum(muonConstants);

    GEMDetId rollDetId(gemnum.baseNumberToUnitNumber(mbn));
    LogDebug("GEMGeometryBuilderFromDDD") << "GEM eta partition rawId: " << rollDetId.rawId() << ", detId: " << rollDetId;

    // chamber id for this partition. everything is the same; but partition number is 0.
    GEMDetId chamberId(rollDetId.chamberId());
    LogDebug("GEMGeometryBuilderFromDDD") << "GEM chamber rawId: " << chamberId.rawId() << ", detId: " << chamberId;
    const int stationId(rollDetId.station());
    if (stationId > maxStation) maxStation = stationId;
    
    if (rollDetId.roll()==1) ++nChambers;

    DDValue numbOfStrips("nStrips");
    DDValue numbOfPads("nPads");

    std::vector<const DDsvalues_type* > specs(fview.specifics());
    std::vector<const DDsvalues_type* >::iterator is = specs.begin();
    double nStrips = 0., nPads = 0.;
    for (;is != specs.end(); is++)
    {
      if (DDfetch( *is, numbOfStrips)) nStrips = numbOfStrips.doubles()[0];
      if (DDfetch( *is, numbOfPads))   nPads = numbOfPads.doubles()[0];
    }
    LogDebug("GEMGeometryBuilderFromDDD") 
      << ((nStrips == 0. ) ? ("No nStrips found!!") : ("Number of strips: " + boost::lexical_cast<std::string>(nStrips))); 
    LogDebug("GEMGeometryBuilderFromDDD") 
      << ((nPads == 0. ) ? ("No nPads found!!") : ("Number of pads: " + boost::lexical_cast<std::string>(nPads)));

    std::vector<double> dpar=fview.logicalPart().solid().parameters();
    std::string name = fview.logicalPart().name().name();
    DDTranslation tran = fview.translation();
   //removed .Inverse after comparing to DT...
    DDRotationMatrix rota = fview.rotation();//.Inverse();
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
    pars.push_back(nStrips);
    pars.push_back(nPads);

    LogDebug("GEMGeometryBuilderFromDDD") 
      << "GEM " << name << " par " << be << " " << te << " " << ap << " " << dpar[0];
    
    GEMEtaPartitionSpecs* e_p_specs = new GEMEtaPartitionSpecs(GeomDetEnumerators::GEM, name, pars);

      //Change of axes for the forward
    Basic3DVector<float> newX(1.,0.,0.);
    Basic3DVector<float> newY(0.,0.,1.);
    //      if (tran.z() > 0. )
    newY *= -1;
    Basic3DVector<float> newZ(0.,1.,0.);
    rot.rotateAxes (newX, newY, newZ);
    
    BoundPlane* bp = new BoundPlane(pos, rot, bounds);
    ReferenceCountingPointer<BoundPlane> surf(bp);
    GEMEtaPartition* gep = new GEMEtaPartition(rollDetId, surf, e_p_specs);

    // Add the eta partition to the geometry
    geometry->add(gep);
    // go to next layer
    doSubDets = fview.nextSibling(); 
  }
  
  auto& partitions(geometry->etaPartitions());
  // build the chambers and add them to the geometry
  std::vector<GEMDetId> vDetId;
  vDetId.clear();
  int oldRollNumber = 1;
  for (unsigned i=1; i<=partitions.size(); ++i){
    GEMDetId detId(partitions.at(i-1)->id());
    const int rollNumber(detId.roll());
    // new batch of eta partitions --> new chamber
    if (rollNumber < oldRollNumber || i == partitions.size()) {
      // don't forget the last partition for the last chamber
      if (i == partitions.size()) vDetId.push_back(detId);

      GEMDetId fId(vDetId.front());
      GEMDetId chamberId(fId.chamberId());
      // compute the overall boundplane using the first eta partition
      const GEMEtaPartition* p(geometry->etaPartition(fId));
      const BoundPlane& bps = p->surface();
      BoundPlane* bp = const_cast<BoundPlane*>(&bps);
      ReferenceCountingPointer<BoundPlane> surf(bp);
      
      GEMChamber* ch = new GEMChamber(chamberId, surf); 
      LogDebug("GEMGeometryBuilderFromDDD")  << "Creating chamber " << chamberId << " with " << vDetId.size() << " eta partitions" << std::endl;
      
      for(auto id : vDetId){
	LogDebug("GEMGeometryBuilderFromDDD") << "Adding eta partition " << id << " to GEM chamber" << std::endl;
	ch->add(const_cast<GEMEtaPartition*>(geometry->etaPartition(id)));
      }

      LogDebug("GEMGeometryBuilderFromDDD") << "Adding the chamber to the geometry" << std::endl;
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
    GEMDetId detIdL1(chambers.at(i)->id());
    if (detIdL1.layer()==2) continue;
    GEMDetId detIdL2(detIdL1.region(),detIdL1.ring(),detIdL1.station(),2,detIdL1.chamber(),0);
    auto ch2 = geometry->chamber(detIdL2);

    LogDebug("GEMGeometryBuilderFromDDD") << "First chamber for super chamber: " << detIdL1 << std::endl;
    LogDebug("GEMGeometryBuilderFromDDD") << "Second chamber for super chamber: " << detIdL2 << std::endl;

    LogDebug("GEMGeometryBuilderFromDDD") << "Creating new GEM super chamber out of chambers." << std::endl;
    GEMSuperChamber* sch = new GEMSuperChamber(detIdL1, surf); 
    sch->add(const_cast<GEMChamber*>(chambers.at(i)));
    sch->add(const_cast<GEMChamber*>(ch2));

    LogDebug("GEMGeometryBuilderFromDDD") << "Adding the super chamber to the geometry." << std::endl;
    geometry->add(sch);
  }

  auto& superChambers(geometry->superChambers());
  // construct the regions, stations and rings. 
  for (int re = -1; re <= 1; re = re+2) {
    GEMRegion* region = new GEMRegion(re); 
    for (int st=1; st<=maxStation; ++st) {
      GEMStation* station = new GEMStation(re, st);
      std::string sign( re==-1 ? "-" : "");
      std::string name("GE" + sign + std::to_string(st) + "/1");
      // Closest (furthest) super chambers in GE2/1 are called GE2/1s (GE2/1l)
      if (st==2) name = "GE" + sign + std::to_string(st) + "/1s";
      if (st==3) name = "GE" + sign + std::to_string(st-1) + "/1l";
      station->setName(name); 
      for (int ri=1; ri<=1; ++ri) {
	GEMRing* ring = new GEMRing(re, st, ri); 
	for (unsigned sch=0; sch<superChambers.size(); ++sch){
	  const GEMDetId detId(superChambers.at(sch)->id());
	  if (detId.region() != re || detId.station() != st || detId.ring() != ri) continue;
	  ring->add(superChambers.at(sch));
	  LogDebug("GEMGeometryBuilderFromDDD") << "Adding super chamber " << detId << " to ring: " 
						<< "re " << re << " st " << st << " ri " << ri << std::endl;
 	}
	LogDebug("GEMGeometryBuilderFromDDD") << "Adding ring " <<  ri << " to station " << "re " << re << " st " << st << std::endl;
	station->add(ring);
	geometry->add(ring);
      }
      LogDebug("GEMGeometryBuilderFromDDD") << "Adding station " << st << " to region " << re << std::endl;
      region->add(station);
      geometry->add(station);
    }
    LogDebug("GEMGeometryBuilderFromDDD") << "Adding region " << re << " to the geometry " << std::endl;
    geometry->add(region);
  }
  return geometry;
}
