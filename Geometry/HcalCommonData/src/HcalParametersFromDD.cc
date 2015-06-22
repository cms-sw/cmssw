#include "Geometry/HcalCommonData/interface/HcalParametersFromDD.h"
#include "Geometry/CaloTopology/interface/HcalTopologyMode.h"
#include "CondFormats/GeometryObjects/interface/PHcalParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Base/interface/DDutils.h"

namespace
{
  double getTopologyMode( const char* s, const DDsvalues_type & sv )
  {
    DDValue val( s );
    if( DDfetch( &sv, val ))
    {
      const std::vector<std::string> & fvec = val.strings();
      if( fvec.size() == 0 )
      {
	throw cms::Exception( "HcalParametersFromDD" ) << "Failed to get " << s << " tag.";
      }

      StringToEnumParser<HcalTopologyMode::Mode> eparser;
      HcalTopologyMode::Mode mode = (HcalTopologyMode::Mode) eparser.parseString( fvec[0] );

      return double( mode );
    }
    else
      throw cms::Exception( "HcalParametersFromDD" ) << "Failed to get "<< s << " tag.";
  }
}

bool
HcalParametersFromDD::build( const DDCompactView* cpv,
			     PHcalParameters& php )
{
  php.phioff = DDVectorGetter::get( "phioff" );
  php.etaTable = DDVectorGetter::get( "etaTable" );
  php.rTable = DDVectorGetter::get( "rTable" );
  php.phibin = DDVectorGetter::get( "phibin" );
  php.phitable = DDVectorGetter::get( "phitable" );  
  php.etaRange = DDVectorGetter::get( "etaRange" );
  php.gparHF = DDVectorGetter::get( "gparHF" );
  php.noff = dbl_to_int( DDVectorGetter::get( "noff" ));
  php.Layer0Wt = DDVectorGetter::get( "Layer0Wt" );  
  php.HBGains = DDVectorGetter::get( "HBGains" );
  php.HEGains = DDVectorGetter::get( "HEGains" );
  php.HFGains = DDVectorGetter::get( "HFGains" );
  php.etaMin = dbl_to_int( DDVectorGetter::get( "etaMin" ));
  php.etaMax = dbl_to_int( DDVectorGetter::get( "etaMax" ));
  php.HBShift = dbl_to_int( DDVectorGetter::get( "HBShift" ));
  php.HEShift = dbl_to_int( DDVectorGetter::get( "HEShift" ));
  php.HFShift = dbl_to_int( DDVectorGetter::get( "HFShift" ));

  php.etagroup = dbl_to_int( DDVectorGetter::get( "etagroup" ));
  php.phigroup = dbl_to_int( DDVectorGetter::get( "phigroup" ));
			      
  PHcalParameters::LayerItem layerGroupEta;
  for( unsigned int i = 0; i < 27; ++i )
  {
    std::stringstream sstm;
    sstm << "layerGroupEta" << i;
    std::string tempName = sstm.str();

    if( DDVectorGetter::check( tempName ))
    {
      PHcalParameters::LayerItem layerGroupEta;
      layerGroupEta.layer = i;
      layerGroupEta.layerGroup = dbl_to_int( DDVectorGetter::get( tempName ));
      php.layerGroupEta.push_back( layerGroupEta );
    }
  }

  // FIXME: HcalTopology mode can be defined as double.
  //        This is for consistency with SLHC releases.
  //
  std::string attribute = "OnlyForHcalRecNumbering"; 
  std::string value     = "any";
  DDValue val( attribute, value, 0.0 );
  
  DDSpecificsFilter filter;
  filter.setCriteria( val,
		      DDCompOp::not_equals,
		      DDLogOp::AND, true, // compare strings 
		      true  // use merged-specifics or simple-specifics
		     );
  DDFilteredView fv( *cpv );
  fv.addFilter( filter );
  bool ok = fv.firstChild();
  
  if( !ok ) throw cms::Exception( "HcalParametersFromDD" ) << "Not found "<< attribute.c_str() << " but needed.";
  
  DDsvalues_type sv( fv.mergedSpecifics());
  
  php.topologyMode = getTopologyMode( "TopologyMode", sv );

  return true;
}
