/** Implementation of the RPC Geometry Builder from DDD
 *
 *  \author Port of: MuDDDRPCBuilder (ORCA)
 *  \author M. Maggi - INFN Bari
 */
#include "Geometry/RPCGeometryBuilder/src/RPCGeometryParsFromDD.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include <CondFormats/GeometryObjects/interface/RecoIdealGeometry.h>
#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>

#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/RPCNumberingScheme.h"

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>
#include <algorithm>

RPCGeometryParsFromDD::RPCGeometryParsFromDD() 
{ }

RPCGeometryParsFromDD::~RPCGeometryParsFromDD() 
{ }

void
RPCGeometryParsFromDD::build(const DDCompactView* cview, 
      const MuonDDDConstants& muonConstants, RecoIdealGeometry& rgeo )
{
  std::string attribute = "ReadOutName";
  std::string value     = "MuonRPCHits";
  DDValue val(attribute, value, 0.0);

  // Asking only for the MuonRPC's
  DDSpecificsFilter filter;
  filter.setCriteria(val, // name & value of a variable 
		     DDCompOp::matches,
		     DDLogOp::AND, 
		     true, // compare strings otherwise doubles
		     true // use merged-specifics or simple-specifics
		     );
  DDFilteredView fview(*cview);
  fview.addFilter(filter);

  this->buildGeometry(fview, muonConstants, rgeo);
}

void
RPCGeometryParsFromDD::buildGeometry(DDFilteredView& fview, const MuonDDDConstants& muonConstants, RecoIdealGeometry& rgeo)
{

  bool doSubDets = fview.firstChild();

  while (doSubDets){

    // Get the Base Muon Number
    MuonDDDNumbering mdddnum(muonConstants);
    MuonBaseNumber   mbn=mdddnum.geoHistoryToBaseNumber(fview.geoHistory());

    // Get the The Rpc det Id 
    RPCNumberingScheme rpcnum(muonConstants);
    int detid = 0;

    detid = rpcnum.baseNumberToUnitNumber(mbn);
    RPCDetId rpcid(detid);

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
    if (dpar.size()==3){
      double width     = dpar[0];
      double length    = dpar[1];
      double thickness = dpar[2];
      pars.push_back(width);
      pars.push_back(length);
      pars.push_back(thickness);
      pars.push_back(numbOfStrips.doubles()[0]); 
    }else{
      pars.push_back(dpar[4]); //b/2;
      pars.push_back(dpar[8]); //B/2;
      pars.push_back(dpar[0]); //h/2;
      pars.push_back(0.4);
      pars.push_back(numbOfStrips.doubles()[0]); //h/2;
    }

    
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
    rgeo.insert(rpcid.rawId(),vtra,vrot, pars,strpars);
    doSubDets = fview.nextSibling(); // go to next layer
  }
  
}
