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

void RPCGeometryParsFromDD::build(const DDCompactView* cview,
                                  const MuonDDDConstants& muonConstants, RecoIdealGeometry& rgeo )
{
  const std::string attribute = "ReadOutName";
  const std::string value     = "MuonRPCHits";

  // Asking only for the MuonRPC's
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0.0)};
  DDFilteredView fview(*cview,filter);

  this->buildGeometry(fview, muonConstants, rgeo);
}

void RPCGeometryParsFromDD::buildGeometry(DDFilteredView& fview,
                                          const MuonDDDConstants& muonConstants, RecoIdealGeometry& rgeo)
{
  for ( bool doSubDets = fview.firstChild(); doSubDets==true; doSubDets = fview.nextSibling() ) {

    // Get the Base Muon Number
    MuonDDDNumbering mdddnum(muonConstants);
    MuonBaseNumber   mbn=mdddnum.geoHistoryToBaseNumber(fview.geoHistory());

    // Get the The Rpc det Id
    RPCNumberingScheme rpcnum(muonConstants);
    const int detid = rpcnum.baseNumberToUnitNumber(mbn);
    RPCDetId rpcid(detid);

    DDValue numbOfStrips("nStrips");

    std::vector<const DDsvalues_type* > specs(fview.specifics());
    int nStrips=0;
    for (auto & spec : specs){
      if (DDfetch( spec, numbOfStrips)){
        nStrips=int(numbOfStrips.doubles()[0]);
      }
    }
    if (nStrips == 0 ) std::cout <<"No strip found!!"<<std::endl;

    const std::vector<double> dpar = fview.logicalPart().solid().parameters();

    const std::string name=fview.logicalPart().name().name();
    const std::vector<std::string> strpars = {name};
    const DDTranslation& tran = fview.translation();

    const DDRotationMatrix& rota = fview.rotation();//.Inverse();
    DD3Vector x, y, z;
    rota.GetComponents(x,y,z);
    std::vector<double> pars;
    if (dpar.size()==3){
      const double width     = dpar[0];
      const double length    = dpar[1];
      const double thickness = dpar[2];
      pars = {width, length, thickness, numbOfStrips.doubles()[0]};
    }
    else{
      pars = {
        dpar[4] /*b/2*/, dpar[8] /*B/2*/, dpar[0] /*h/2*/,
        0.4,
        numbOfStrips.doubles()[0] /*h/2*/
      };
    }

    const std::vector<double> vtra = {tran.x(), tran.y(), tran.z()};
    const std::vector<double> vrot = {x.X(), x.Y(), x.Z(),
                                      y.X(), y.Y(), y.Z(),
                                      z.X(), z.Y(), z.Z()};
    rgeo.insert(rpcid.rawId(),vtra,vrot, pars,strpars);
  }

}
