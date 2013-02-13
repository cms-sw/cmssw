/** Implementation of the GEM Geometry Builder from DDD
 *
 *  \author M. Maggi - INFN Bari
 */
#include "Geometry/GEMGeometryBuilder/src/GEMGeometryParsFromDD.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

#include <CondFormats/GeometryObjects/interface/RecoIdealGeometry.h>
#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>

#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/GEMNumberingScheme.h"

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>
#include <algorithm>

GEMGeometryParsFromDD::GEMGeometryParsFromDD() 
{ }

GEMGeometryParsFromDD::~GEMGeometryParsFromDD() 
{ }

void
GEMGeometryParsFromDD::build(const DDCompactView* cview, 
      const MuonDDDConstants& muonConstants, RecoIdealGeometry& rgeo )
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

  this->buildGeometry(fview, muonConstants, rgeo);
}

void
GEMGeometryParsFromDD::buildGeometry(DDFilteredView& fview, const MuonDDDConstants& muonConstants, RecoIdealGeometry& rgeo)
{

  bool doSubDets = fview.firstChild();

  while (doSubDets){

    // Get the Base Muon Number
    MuonDDDNumbering mdddnum(muonConstants);
    MuonBaseNumber   mbn=mdddnum.geoHistoryToBaseNumber(fview.geoHistory());

    // Get the The GEM det Id 
    GEMNumberingScheme gemnum(muonConstants);
    int detid = 0;

    detid = gemnum.baseNumberToUnitNumber(mbn);
    GEMDetId gemid(detid);

    DDValue numbOfStrips("nStrips");

    std::vector<const DDsvalues_type* > specs(fview.specifics());
    std::vector<const DDsvalues_type* >::iterator is=specs.begin();
    int nStrips=0;
    for (;is!=specs.end(); is++){
      if (DDfetch( *is, numbOfStrips)){
	nStrips=int(numbOfStrips.doubles()[0]);	
      }
    }
    if (nStrips == 0 )
      std::cout <<"No strip found!!"<<std::endl;
    
    std::vector<double> dpar=fview.logicalPart().solid().parameters();

    std::vector<std::string> strpars;
    std::string name=fview.logicalPart().name().name();
    strpars.push_back(name);
    DDTranslation tran    = fview.translation();

    DDRotationMatrix rota = fview.rotation();//.Inverse();
    DD3Vector x, y, z;
    rota.GetComponents(x,y,z);
    std::vector<double> pars;    
    pars.push_back(dpar[4]); //b/2;
    pars.push_back(dpar[8]); //B/2;
    pars.push_back(dpar[0]); //h/2;
    pars.push_back(0.4);
    pars.push_back(numbOfStrips.doubles()[0]); //h/2;
    
    std::vector<double> vtra(3);
    std::vector<double> vrot(9);
    vtra[0]=(float) 1.0 * (tran.x());
    vtra[1]=(float) 1.0 * (tran.y());
    vtra[2]=(float) 1.0 * (tran.z());
    vrot[0]=(float) 1.0 * x.X();
    vrot[1]=(float) 1.0 * x.Y();
    vrot[2]=(float) 1.0 * x.Z();
    vrot[3]=(float) 1.0 * y.X();
    vrot[4]=(float) 1.0 * y.Y();
    vrot[5]=(float) 1.0 * y.Z();
    vrot[6]=(float) 1.0 * z.X();
    vrot[7]=(float) 1.0 * z.Y();
    vrot[8]=(float) 1.0 * z.Z();
    rgeo.insert(gemid.rawId(),vtra,vrot, pars,strpars);
    doSubDets = fview.nextSibling(); // go to next layer
  }
  
}
