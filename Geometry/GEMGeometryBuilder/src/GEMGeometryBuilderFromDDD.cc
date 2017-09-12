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

#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>
#include <algorithm>
#include <boost/lexical_cast.hpp>

GEMGeometryBuilderFromDDD::GEMGeometryBuilderFromDDD()
{ }

GEMGeometryBuilderFromDDD::~GEMGeometryBuilderFromDDD() 
{ }

GEMGeometry* GEMGeometryBuilderFromDDD::build(const DDCompactView* cview, const MuonDDDConstants& muonConstants)
{
  std::string attribute = "MuStructure"; // "ReadOutName"; // could come from .orcarc
  std::string value     = "MuonEndCapGEM"; // "MuonGEMHits"; // could come from .orcarc

  
  // Asking only for the MuonGEM's
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0.0)};
  DDFilteredView fview(*cview,filter);

  return this->buildGeometry(fview, muonConstants);
}

GEMGeometry* GEMGeometryBuilderFromDDD::buildGeometry(DDFilteredView& fv, const MuonDDDConstants& muonConstants)
{
  LogDebug("GEMGeometryBuilderFromDDD") <<"Building the geometry service";
  GEMGeometry* geometry = new GEMGeometry();
  LogDebug("GEMGeometryBuilderFromDDD") << "About to run through the GEM structure\n" 
					<<" First logical part "
					<<fv.logicalPart().name().name();
 

  
  bool doSuper = fv.firstChild();
  LogDebug("GEMGeometryBuilderFromDDD") << "doSuperChamber = " << doSuper;
  // loop over superchambers
  while (doSuper){

    // getting chamber id from eta partitions
    fv.firstChild();fv.firstChild();
    MuonDDDNumbering mdddnumCh(muonConstants);
    GEMNumberingScheme gemNumCh(muonConstants);
    int rawidCh = gemNumCh.baseNumberToUnitNumber(mdddnumCh.geoHistoryToBaseNumber(fv.geoHistory()));
    GEMDetId detIdCh = GEMDetId(rawidCh);
    // back to chambers
    fv.parent();fv.parent();

    // currently there is no superchamber in the geometry
    // only 2 chambers are present separated by a gap.
    // making superchamber out of the first chamber layer including the gap between chambers
    if (detIdCh.layer() == 1){// only make superChambers when doing layer 1
      GEMSuperChamber *gemSuperChamber = buildSuperChamber(fv, detIdCh);
      geometry->add(gemSuperChamber);
    }
    GEMChamber *gemChamber = buildChamber(fv, detIdCh);
    
    // loop over chambers
    // only 1 chamber
    bool doChambers = fv.firstChild();
    while (doChambers){
      
    // loop over GEMEtaPartitions
      bool doEtaPart = fv.firstChild();
      while (doEtaPart){

	MuonDDDNumbering mdddnum(muonConstants);
	GEMNumberingScheme gemNum(muonConstants);
	int rawid = gemNum.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
	GEMDetId detId = GEMDetId(rawid);

	GEMEtaPartition *etaPart = buildEtaPartition(fv, detId);
	gemChamber->add(etaPart);
	geometry->add(etaPart);
	doEtaPart = fv.nextSibling();
      }
      fv.parent();

      geometry->add(gemChamber);
      
      doChambers = fv.nextSibling();
    }
    fv.parent();

    doSuper = fv.nextSibling();
  }
  
  auto& superChambers(geometry->superChambers());
  // construct the regions, stations and rings. 
  for (int re = -1; re <= 1; re = re+2) {
    GEMRegion* region = new GEMRegion(re);
    for (int st=1; st<=GEMDetId::maxStationId; ++st) {
      GEMStation* station = new GEMStation(re, st);
      std::string sign( re==-1 ? "-" : "");
      std::string name("GE" + sign + std::to_string(st) + "/1");
      station->setName(name);
      for (int ri=1; ri<=1; ++ri) {
	GEMRing* ring = new GEMRing(re, st, ri);
	for (auto sch : superChambers){
	  GEMSuperChamber* superChamber = const_cast<GEMSuperChamber*>(sch);
	  const GEMDetId detId(superChamber->id());
	  if (detId.region() != re || detId.station() != st || detId.ring() != ri) continue;
	  
	  auto ch1 = geometry->chamber(GEMDetId(detId.region(),detId.ring(),detId.station(),1,detId.chamber(),0));
	  auto ch2 = geometry->chamber(GEMDetId(detId.region(),detId.ring(),detId.station(),2,detId.chamber(),0));
	  superChamber->add(const_cast<GEMChamber*>(ch1));
	  superChamber->add(const_cast<GEMChamber*>(ch2));
	  
	  ring->add(superChamber);
	  LogDebug("GEMGeometryBuilderFromDDD") << "Adding super chamber " << detId << " to ring: " 
						<< "re " << re << " st " << st << " ri " << ri << std::endl;
 	}
	LogDebug("GEMGeometryBuilderFromDDD") << "Adding ring " <<  ri << " to station " << "re " << re << " st " << st << std::endl;
	station->add(const_cast<GEMRing*>(ring));
	geometry->add(const_cast<GEMRing*>(ring));
      }
      LogDebug("GEMGeometryBuilderFromDDD") << "Adding station " << st << " to region " << re << std::endl;
      region->add(const_cast<GEMStation*>(station));
      geometry->add(const_cast<GEMStation*>(station));
    }
    LogDebug("GEMGeometryBuilderFromDDD") << "Adding region " << re << " to the geometry " << std::endl;
    geometry->add(const_cast<GEMRegion*>(region));
  }
  
  return geometry;
}

GEMSuperChamber* GEMGeometryBuilderFromDDD::buildSuperChamber(DDFilteredView& fv, GEMDetId detId) const {
  LogDebug("GEMGeometryBuilderFromDDD") << "buildSuperChamber "<<fv.logicalPart().name().name() <<" "<< detId <<std::endl;
  
  DDBooleanSolid solid = (DDBooleanSolid)(fv.logicalPart().solid());
  std::vector<double> dpar = solid.solidA().parameters();
  
  double dy = dpar[0]/cm;//length is along local Y
  double dz = dpar[3]/cm;// thickness is long local Z
  double dx1= dpar[4]/cm;// bottom width is along local X
  double dx2= dpar[8]/cm;// top width is along local X
  dpar = solid.solidB().parameters();
  dz += dpar[3]/cm;// chamber thickness
  dz *=2; // 2 chambers in superchamber
  dz += 2.105;// gap between chambers
  
  bool isOdd = detId.chamber()%2;
  RCPBoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(dx1,dx2,dy,dz), isOdd ));
  LogDebug("GEMGeometryBuilderFromDDD") << "size "<< dx1 << " " << dx2 << " " << dy << " " << dz <<std::endl;
  
  GEMSuperChamber* superChamber = new GEMSuperChamber(detId.superChamberId(), surf);
  return superChamber;
}

GEMChamber* GEMGeometryBuilderFromDDD::buildChamber(DDFilteredView& fv, GEMDetId detId) const {
  LogDebug("GEMGeometryBuilderFromDDD") << "buildChamber "<<fv.logicalPart().name().name() <<" "<< detId <<std::endl;
  
  DDBooleanSolid solid = (DDBooleanSolid)(fv.logicalPart().solid());
  std::vector<double> dpar = solid.solidA().parameters();
  
  double dy = dpar[0]/cm;//length is along local Y
  double dz = dpar[3]/cm;// thickness is long local Z
  double dx1= dpar[4]/cm;// bottom width is along local X
  double dx2= dpar[8]/cm;// top width is along local X
  dpar = solid.solidB().parameters();
  dz += dpar[3]/cm;// chamber thickness
  
  bool isOdd = detId.chamber()%2;
  RCPBoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(dx1,dx2,dy,dz), isOdd ));
  LogDebug("GEMGeometryBuilderFromDDD") << "size "<< dx1 << " " << dx2 << " " << dy << " " << dz <<std::endl;
  
  GEMChamber* chamber = new GEMChamber(detId.chamberId(), surf);
  return chamber;
}

GEMEtaPartition* GEMGeometryBuilderFromDDD::buildEtaPartition(DDFilteredView& fv, GEMDetId detId) const {
  LogDebug("GEMGeometryBuilderFromDDD") << "buildEtaPartition "<<fv.logicalPart().name().name() <<" "<< detId <<std::endl;
  
  // EtaPartition specific parameter (nstrips and npads) 
  DDValue numbOfStrips("nStrips");
  DDValue numbOfPads("nPads");
  std::vector<const DDsvalues_type* > specs(fv.specifics());
  std::vector<const DDsvalues_type* >::iterator is = specs.begin();
  double nStrips = 0., nPads = 0.;
  for (;is != specs.end(); is++){
    if (DDfetch( *is, numbOfStrips)) nStrips = numbOfStrips.doubles()[0];
    if (DDfetch( *is, numbOfPads))   nPads = numbOfPads.doubles()[0];
  }
  LogDebug("GEMGeometryBuilderFromDDD") 
    << ((nStrips == 0. ) ? ("No nStrips found!!") : ("Number of strips: " + boost::lexical_cast<std::string>(nStrips))); 
  LogDebug("GEMGeometryBuilderFromDDD") 
    << ((nPads == 0. ) ? ("No nPads found!!") : ("Number of pads: " + boost::lexical_cast<std::string>(nPads)));
  
  // EtaPartition specific parameter (size) 
  std::vector<double> dpar = fv.logicalPart().solid().parameters();

  double be = dpar[4]/cm; // half bottom edge
  double te = dpar[8]/cm; // half top edge
  double ap = dpar[0]/cm; // half apothem
  double ti = 0.4/cm;     // half thickness
  
  std::vector<float> pars;
  pars.emplace_back(be); 
  pars.emplace_back(te); 
  pars.emplace_back(ap); 
  pars.emplace_back(nStrips);
  pars.emplace_back(nPads);
  
  bool isOdd = detId.chamber()%2;
  RCPBoundPlane surf(boundPlane(fv, new TrapezoidalPlaneBounds(be, te, ap, ti), isOdd ));
  std::string name = fv.logicalPart().name().name();
  GEMEtaPartitionSpecs* e_p_specs = new GEMEtaPartitionSpecs(GeomDetEnumerators::GEM, name, pars);
  
  LogDebug("GEMGeometryBuilderFromDDD") << "size "<< be << " " << te << " " << ap << " " << ti <<std::endl;
  GEMEtaPartition* etaPartition = new GEMEtaPartition(detId, surf, e_p_specs);
  return etaPartition;
}

GEMGeometryBuilderFromDDD::RCPBoundPlane 
GEMGeometryBuilderFromDDD::boundPlane(const DDFilteredView& fv,
				      Bounds* bounds, bool isOddChamber) const {
  // extract the position
  const DDTranslation & trans(fv.translation());
  const Surface::PositionType posResult(float(trans.x()/cm), 
                                        float(trans.y()/cm), 
                                        float(trans.z()/cm));
  
  // now the rotation
  //  DDRotationMatrix tmp = fv.rotation();
  // === DDD uses 'active' rotations - see CLHEP user guide ===
  //     ORCA uses 'passive' rotation. 
  //     'active' and 'passive' rotations are inverse to each other
  //  DDRotationMatrix tmp = fv.rotation();
  const DDRotationMatrix& rotation = fv.rotation();//REMOVED .Inverse();
  DD3Vector x, y, z;
  rotation.GetComponents(x,y,z);
  // LogDebug("GEMGeometryBuilderFromDDD") << "translation: "<< fv.translation() << std::endl;
  // LogDebug("GEMGeometryBuilderFromDDD") << "rotation   : "<< fv.rotation() << std::endl;
  // LogDebug("GEMGeometryBuilderFromDDD") << "INVERSE rotation manually: \n"
  // 	    << x.X() << ", " << x.Y() << ", " << x.Z() << std::endl
  // 	    << y.X() << ", " << y.Y() << ", " << y.Z() << std::endl
  // 	    << z.X() << ", " << z.Y() << ", " << z.Z() << std::endl;

  Surface::RotationType rotResult(float(x.X()),float(x.Y()),float(x.Z()),
  				  float(y.X()),float(y.Y()),float(y.Z()),
  				  float(z.X()),float(z.Y()),float(z.Z()));
  
  //Change of axes for the forward
  Basic3DVector<float> newX(1.,0.,0.);
  Basic3DVector<float> newY(0.,0.,1.);
  Basic3DVector<float> newZ(0.,1.,0.);
  // Odd chambers are inverted in gem.xml
  if (isOddChamber) newY *= -1;
  
  rotResult.rotateAxes(newX, newY, newZ);

  return RCPBoundPlane( new BoundPlane( posResult, rotResult, bounds));
}
