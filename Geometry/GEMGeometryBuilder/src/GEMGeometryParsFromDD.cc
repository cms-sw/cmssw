/** Implementation of the GEM Geometry Builder from DDD
 *
 *  \author M. Maggi - INFN Bari
 */
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryParsFromDD.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/GEMNumberingScheme.h"

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>
#include <algorithm>
#include <boost/lexical_cast.hpp>

GEMGeometryParsFromDD::GEMGeometryParsFromDD() 
{ }

GEMGeometryParsFromDD::~GEMGeometryParsFromDD() 
{ }

void
GEMGeometryParsFromDD::build(const DDCompactView* cview, 
			     const MuonDDDConstants& muonConstants, RecoIdealGeometry& rgeo )
{
  std::string attribute = "MuStructure";
  std::string value     = "MuonEndCapGEM";

  // Asking only for the MuonGEM's
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0.0)};
  DDFilteredView fv(*cview,filter);
  
  this->buildGeometry(fv, muonConstants, rgeo);
}

void
GEMGeometryParsFromDD::buildGeometry(DDFilteredView& fv,
				     const MuonDDDConstants& muonConstants,
				     RecoIdealGeometry& rgeo)
{  
  LogDebug("GEMGeometryParsFromDD") <<"Building the geometry service";
  LogDebug("GEMGeometryParsFromDD") << "About to run through the GEM structure\n" 
				    <<" First logical part "
				    <<fv.logicalPart().name().name(); 
  
  bool doSuper = fv.firstChild();
  LogDebug("GEMGeometryParsFromDD") << "doSuperChamber = " << doSuper;
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
      buildSuperChamber(fv, detIdCh, rgeo);
    }
    buildChamber(fv, detIdCh, rgeo);

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

	buildEtaPartition(fv, detId, rgeo);
	
	doEtaPart = fv.nextSibling();
      }
      fv.parent();
      doChambers = fv.nextSibling();
    }
    fv.parent();
    doSuper = fv.nextSibling();
  }  
}

void
GEMGeometryParsFromDD::buildSuperChamber(DDFilteredView& fv, GEMDetId detId, RecoIdealGeometry& rgeo)
{
  LogDebug("GEMGeometryParsFromDD") << "buildSuperChamber "<<fv.logicalPart().name().name()
				    <<" "<< detId <<std::endl;
  
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

  GEMDetId gemid = detId.superChamberId();
  std::vector<std::string> strpars;
  std::string name = fv.logicalPart().name().name();
  strpars.emplace_back(name);
  
  std::vector<double> pars;  
  pars.emplace_back(dx1); //b/2; bottom width is along local X
  pars.emplace_back(dx2); //B/2; top width is along local X
  pars.emplace_back(dy); //h/2; length is along local Y
  pars.emplace_back(dz); // thickness is long local Z

  std::vector<double> vtra = getTranslation(fv);
  std::vector<double> vrot = getRotation(fv);
  
  rgeo.insert(gemid.rawId(), vtra, vrot, pars, strpars);
}

void
GEMGeometryParsFromDD::buildChamber(DDFilteredView& fv, GEMDetId detId, RecoIdealGeometry& rgeo)
{
  LogDebug("GEMGeometryParsFromDD") << "buildChamber "<<fv.logicalPart().name().name()
				    <<" "<< detId <<std::endl;
  
  DDBooleanSolid solid = (DDBooleanSolid)(fv.logicalPart().solid());
  std::vector<double> dpar = solid.solidA().parameters();
  
  double dy = dpar[0]/cm;//length is along local Y
  double dz = dpar[3]/cm;// thickness is long local Z
  double dx1= dpar[4]/cm;// bottom width is along local X
  double dx2= dpar[8]/cm;// top width is along local X
  dpar = solid.solidB().parameters();
  dz += dpar[3]/cm;// chamber thickness

  GEMDetId gemid = detId.chamberId();
  std::vector<std::string> strpars;
  std::string name = fv.logicalPart().name().name();
  strpars.emplace_back(name);
  
  std::vector<double> pars;  
  pars.emplace_back(dx1); //b/2; bottom width is along local X
  pars.emplace_back(dx2); //B/2; top width is along local X
  pars.emplace_back(dy); //h/2; length is along local Y
  pars.emplace_back(dz); // thickness is long local Z

  std::vector<double> vtra = getTranslation(fv);
  std::vector<double> vrot = getRotation(fv);
  
  rgeo.insert(gemid.rawId(), vtra, vrot, pars, strpars);
}

void
GEMGeometryParsFromDD::buildEtaPartition(DDFilteredView& fv, GEMDetId detId, RecoIdealGeometry& rgeo)
{
  LogDebug("GEMGeometryParsFromDD") << "buildEtaPartition "<<fv.logicalPart().name().name()
				    <<" "<< detId <<std::endl;
  
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
  LogDebug("GEMGeometryParsFromDD") 
    << ((nStrips == 0. ) ? ("No nStrips found!!") : ("Number of strips: " + boost::lexical_cast<std::string>(nStrips))); 
  LogDebug("GEMGeometryParsFromDD") 
    << ((nPads == 0. ) ? ("No nPads found!!") : ("Number of pads: " + boost::lexical_cast<std::string>(nPads)));
  
  // EtaPartition specific parameter (size) 
  std::vector<double> dpar = fv.logicalPart().solid().parameters();

  double dy = dpar[0]/cm;//length is along local Y
  double dz = 0.4/cm;// thickness is long local Z
  double dx1= dpar[4]/cm;// bottom width is along local X
  double dx2= dpar[8]/cm;// top width is along local X

  std::vector<std::string> strpars;
  std::string name = fv.logicalPart().name().name();
  strpars.emplace_back(name);
  
  std::vector<double> pars;  
  pars.emplace_back(dx1); //b/2; bottom width is along local X
  pars.emplace_back(dx2); //B/2; top width is along local X
  pars.emplace_back(dy); //h/2; length is along local Y
  pars.emplace_back(dz); // thickness is long local Z
  pars.emplace_back(nStrips);
  pars.emplace_back(nPads);

  std::vector<double> vtra = getTranslation(fv);
  std::vector<double> vrot = getRotation(fv);
  
  rgeo.insert(detId.rawId(), vtra, vrot, pars, strpars);
}

std::vector<double> GEMGeometryParsFromDD::getTranslation(DDFilteredView& fv)
{
  DDTranslation tran = fv.translation();
  std::vector<double> vtra(3);
  vtra[0]=(float) 1.0 * (tran.x());
  vtra[1]=(float) 1.0 * (tran.y());
  vtra[2]=(float) 1.0 * (tran.z());
  return vtra;  
}

std::vector<double> GEMGeometryParsFromDD::getRotation(DDFilteredView& fv)
{
  DDRotationMatrix rota = fv.rotation();//.Inverse();
  DD3Vector x, y, z;
  rota.GetComponents(x,y,z);  
  std::vector<double> vrot(9);
  vrot[0]=(float) 1.0 * x.X();
  vrot[1]=(float) 1.0 * x.Y();
  vrot[2]=(float) 1.0 * x.Z();
  vrot[3]=(float) 1.0 * y.X();
  vrot[4]=(float) 1.0 * y.Y();
  vrot[5]=(float) 1.0 * y.Z();
  vrot[6]=(float) 1.0 * z.X();
  vrot[7]=(float) 1.0 * z.Y();
  vrot[8]=(float) 1.0 * z.Z();
  return vrot;
}
