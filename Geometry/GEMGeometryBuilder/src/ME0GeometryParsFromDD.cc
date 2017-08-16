#include "Geometry/GEMGeometryBuilder/src/ME0GeometryParsFromDD.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/ME0NumberingScheme.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

void
ME0GeometryParsFromDD::build( const DDCompactView* cview, 
			      const MuonDDDConstants& muonConstants,
			      RecoIdealGeometry& rgeo )
{
  std::string attribute = "ReadOutName";
  std::string value     = "MuonME0Hits";

  // Asking only for the MuonME0's
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0.0)};
  DDFilteredView fview( *cview, filter );

  this->buildGeometry( fview, muonConstants, rgeo );
}

void
ME0GeometryParsFromDD::buildGeometry( DDFilteredView& fview,
				      const MuonDDDConstants& muonConstants,
				      RecoIdealGeometry& rgeo )
{
  bool doSubDets = fview.firstChild();

  while( doSubDets )
  {
    // Get the Base Muon Number
    MuonDDDNumbering mdddnum( muonConstants );
    MuonBaseNumber mbn = mdddnum.geoHistoryToBaseNumber( fview.geoHistory());

    // Get the The ME0 det Id 
    ME0NumberingScheme me0num( muonConstants );
    ME0DetId rollDetId( me0num.baseNumberToUnitNumber( mbn ));

    std::vector<double> dpar = fview.logicalPart().solid().parameters();
    std::vector<double> pars;
    pars.emplace_back( dpar[4]); // half bottom edge
    pars.emplace_back( dpar[8]); // half top edge
    pars.emplace_back( dpar[0]); // half apothem
    pars.emplace_back( 0.4 ); // half thickness
    pars.emplace_back( 0.0 ); // nStrips
    pars.emplace_back( 0.0 ); // nPads

    std::string name = fview.logicalPart().name().name();
    std::vector<std::string> strpars;
    strpars.emplace_back( name );

    DDRotationMatrix rota = fview.rotation();

    // ROOT::Math way
    DD3Vector x, y, z;
    rota.GetComponents( x, y, z );
    std::vector<double> vrot(9);
    vrot[0] = x.X();
    vrot[1] = x.Y();
    vrot[2] = x.Z();
    vrot[3] = y.X();
    vrot[4] = y.Y();
    vrot[5] = y.Z();
    vrot[6] = z.X();
    vrot[7] = z.Y();
    vrot[8] = z.Z();
 
    DDTranslation tran = fview.translation();
    std::vector<double> vtra(3);
    vtra[0] = tran.x();
    vtra[1] = tran.y();
    vtra[2] = tran.z();

    rgeo.insert( rollDetId.rawId(), vtra, vrot, pars, strpars );
    doSubDets = fview.nextSibling(); // go to next layer
  }
}
