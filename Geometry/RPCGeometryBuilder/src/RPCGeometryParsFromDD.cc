/* Implementation of the  RPCGeometryParsFromDD Class
 *  Build the RPCGeometry from the DDD and DD4Hep description
 *  
 *  DD4hep part added to the original old file (DD version) made by M. Maggi (INFN Bari)
 *  Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) 
 *  Created:  Mon, 09 Nov 2020 
 *
 */
#include "Geometry/RPCGeometryBuilder/src/RPCGeometryParsFromDD.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include <CondFormats/GeometryObjects/interface/RecoIdealGeometry.h>
#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>
#include "Geometry/MuonNumbering/interface/MuonGeometryNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/RPCNumberingScheme.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>
#include <algorithm>
#include <DetectorDescription/DDCMS/interface/DDFilteredView.h>
#include <DetectorDescription/DDCMS/interface/DDCompactView.h>
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

RPCGeometryParsFromDD::RPCGeometryParsFromDD() {}

RPCGeometryParsFromDD::~RPCGeometryParsFromDD() {}

// DD
void RPCGeometryParsFromDD::build(const DDCompactView* cview,
                                  const MuonGeometryConstants& muonConstants,
                                  RecoIdealGeometry& rgeo) {
  const std::string attribute = "ReadOutName";
  const std::string value = "MuonRPCHits";

  // Asking only for the MuonRPC's
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0.0)};
  DDFilteredView fview(*cview, filter);

  this->buildGeometry(fview, muonConstants, rgeo);
}

// DD4Hep

void RPCGeometryParsFromDD::build(const cms::DDCompactView* cview,
                                  const MuonGeometryConstants& muonConstants,
                                  RecoIdealGeometry& rgeo) {
  const std::string attribute = "ReadOutName";
  const std::string value = "MuonRPCHits";
  const cms::DDFilter filter(attribute, value);
  cms::DDFilteredView fview(*cview, filter);
  this->buildGeometry(fview, muonConstants, rgeo);
}

// DD

void RPCGeometryParsFromDD::buildGeometry(DDFilteredView& fview,
                                          const MuonGeometryConstants& muonConstants,
                                          RecoIdealGeometry& rgeo) {
  for (bool doSubDets = fview.firstChild(); doSubDets == true; doSubDets = fview.nextSibling()) {
    // Get the Base Muon Number
    MuonGeometryNumbering mdddnum(muonConstants);
    MuonBaseNumber mbn = mdddnum.geoHistoryToBaseNumber(fview.geoHistory());

    // Get the The Rpc det Id
    RPCNumberingScheme rpcnum(muonConstants);
    const int detid = rpcnum.baseNumberToUnitNumber(mbn);

    RPCDetId rpcid(detid);

    DDValue numbOfStrips("nStrips");

    std::vector<const DDsvalues_type*> specs(fview.specifics());
    int nStrips = 0;
    for (auto& spec : specs) {
      if (DDfetch(spec, numbOfStrips)) {
        nStrips = int(numbOfStrips.doubles()[0]);
      }
    }
    if (nStrips == 0)
      std::cout << "No strip found!!" << std::endl;

    const std::vector<double> dpar = fview.logicalPart().solid().parameters();

    const std::string name = fview.logicalPart().name().name();

    edm::LogVerbatim("RPCGeometryParsFromDD")
        << " (1) "
        << "detid: " << detid << " name: " << name << " number of Strips: " << nStrips;

    const std::vector<std::string> strpars = {name};
    const DDTranslation& tran = fview.translation();

    const DDRotationMatrix& rota = fview.rotation();  //.Inverse();
    DD3Vector x, y, z;
    rota.GetComponents(x, y, z);
    std::vector<double> pars;
    if (dpar.size() == 3) {
      const double width = dpar[0];
      const double length = dpar[1];
      const double thickness = dpar[2];
      edm::LogVerbatim("RPCGeometryParsFromDD")
          << " (2) dpar.size() == 3, width: " << width << " length: " << length << " thickness: " << thickness;
      pars = {width, length, thickness, numbOfStrips.doubles()[0]};
    } else {
      const double dpar4 = dpar[4];
      const double dpar8 = dpar[8];
      const double dpar0 = dpar[0];
      pars = {dpar4, dpar8, dpar0, 0.4, numbOfStrips.doubles()[0]};
      edm::LogVerbatim("RPCGeometryParsFromDD")
          << " (3), else, dpar[4]: " << dpar4 << " dpar[8]: " << dpar8 << " dpar[0]: " << dpar0;
    }

    const std::vector<double> vtra = {tran.x(), tran.y(), tran.z()};

    edm::LogVerbatim("RPCGeometryParsFromDD")
        << " (4), tran.x() " << tran.x() << " tran.y(): " << tran.y() << " tran.z(): " << tran.z();

    const std::vector<double> vrot = {x.X(), x.Y(), x.Z(), y.X(), y.Y(), y.Z(), z.X(), z.Y(), z.Z()};
    edm::LogVerbatim("RPCGeometryParsFromDD")
        << " (5), x.X(), x.Y(), x.Z(), y.X(), y.Y(), y.Z(), z.X(), z.Y(), z.Z() " << x.X() << ", " << x.Y() << ", "
        << x.Z() << ", " << y.X() << ", " << y.Y() << ", " << y.Z() << ", " << z.X() << ", " << z.Y() << ", " << z.Z();
    rgeo.insert(rpcid.rawId(), vtra, vrot, pars, strpars);
  }
}

// DD4Hep

void RPCGeometryParsFromDD::buildGeometry(cms::DDFilteredView& fview,
                                          const MuonGeometryConstants& muonConstants,
                                          RecoIdealGeometry& rgeo) {
  while (fview.firstChild()) {
    MuonGeometryNumbering mdddnum(muonConstants);
    RPCNumberingScheme rpcnum(muonConstants);
    int rawidCh = rpcnum.baseNumberToUnitNumber(mdddnum.geoHistoryToBaseNumber(fview.history()));
    RPCDetId rpcid = RPCDetId(rawidCh);

    auto nStrips = fview.get<double>("nStrips");

    std::vector<double> dpar = fview.parameters();

    std::string_view name = fview.name();

    edm::LogVerbatim("RPCGeometryParsFromDD")
        << " (1), detid: " << rawidCh << " name: " << std::string(name) << " number of Strips: " << nStrips;

    const std::vector<std::string> strpars = {std::string(name)};

    std::vector<double> tran(3);
    tran[0] = static_cast<double>(fview.translation().X()) / dd4hep::mm;
    tran[1] = static_cast<double>(fview.translation().Y()) / dd4hep::mm;
    tran[2] = static_cast<double>(fview.translation().Z()) / dd4hep::mm;

    DDRotationMatrix rota;
    fview.rot(rota);
    DD3Vector x, y, z;
    rota.GetComponents(x, y, z);
    const std::vector<double> rot = {x.X(), x.Y(), x.Z(), y.X(), y.Y(), y.Z(), z.X(), z.Y(), z.Z()};

    if (dd4hep::isA<dd4hep::Box>(fview.solid())) {
      const std::vector<double> pars = {
          dpar[0] / dd4hep::mm, dpar[1] / dd4hep::mm, dpar[2] / dd4hep::mm, double(nStrips)};
      edm::LogVerbatim("RPCGeometryParsFromDD")
          << " (2), dd4hep::Box, width: " << dpar[0] / dd4hep::mm << " length: " << dpar[1] / dd4hep::mm
          << " thickness: " << dpar[2] / dd4hep::mm;
      rgeo.insert(rpcid, tran, rot, pars, strpars);
    } else {
      const double ti = 0.4;
      const std::vector<double> pars = {
          dpar[0] / dd4hep::mm, dpar[1] / dd4hep::mm, dpar[3] / dd4hep::mm, ti, double(nStrips)};
      edm::LogVerbatim("RPCGeometryParsFromDD") << " (3), else, dpar[0] (i.e. dpar[4] for DD): " << dpar[0] / dd4hep::mm
                                                << " dpar[1] (i.e. dpar[8] for DD): " << dpar[1] / dd4hep::mm
                                                << " dpar[3] (i.e. dpar[0] for DD): " << dpar[3] / dd4hep::mm;
      rgeo.insert(rpcid, tran, rot, pars, strpars);
    }
    edm::LogVerbatim("RPCGeometryParsFromDD")
        << " (4), tran.x(): " << tran[0] << " tran.y(): " << tran[1] << " tran.z(): " << tran[2];
    edm::LogVerbatim("RPCGeometryParsFromDD")
        << " (5), x.X(), x.Y(), x.Z(), y.X(), y.Y(), y.Z(), z.X(), z.Y(), z.Z(): " << x.X() << ", " << x.Y() << ", "
        << x.Z() << ", " << y.X() << ", " << y.Y() << ", " << y.Z() << ", " << z.X() << ", " << z.Y() << ", " << z.Z();
  }
}
