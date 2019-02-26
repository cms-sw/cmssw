#include "Geometry/MTDGeometryBuilder/interface/MTDParametersFromDD.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"
#include "CondFormats/GeometryObjects/interface/PMTDParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Core/interface/DDutils.h"

using namespace MTDTopologyMode;

namespace {
  int getMTDTopologyMode(const char* s, const DDsvalues_type & sv) {
    DDValue val( s );
    if (DDfetch( &sv, val )) {
      const std::vector<std::string> & fvec = val.strings();
      if (fvec.empty()) {
        throw cms::Exception( "MTDParametersFromDD" ) << "Failed to get " << s << " tag.";
      }
 
      int result(-1);
      MTDTopologyMode::Mode eparser = MTDTopologyMode::MTDStringToEnumParser(fvec[0]);
      result = static_cast<int>(eparser);
      return result;
    } else {
      throw cms::Exception( "MTDParametersFromDD" ) << "Failed to get "<< s << " tag.";
    }
  }
}

bool
MTDParametersFromDD::build( const DDCompactView* cvp,
                            PMTDParameters& ptp)
{
  std::array<std::string,2> mtdSubdet { { "BTL", "ETL" } };
  int subdet(0);
  for( const auto& name : mtdSubdet )
    {
      if( DDVectorGetter::check( name ))
        {
          subdet += 1;
          std::vector<int> subdetPars = dbl_to_int( DDVectorGetter::get( name ));
          putOne( subdet, subdetPars, ptp );
        }
    }
  
  ptp.vpars_ = dbl_to_int( DDVectorGetter::get( "vPars" ));

  std::string attribute = "OnlyForMTDRecNumbering"; 
  DDSpecificsHasNamedValueFilter filter1{attribute};
  DDFilteredView fv1(*cvp,filter1);
  bool ok = fv1.firstChild();
  if (ok) {
    DDsvalues_type sv(fv1.mergedSpecifics());
    int topoMode = getMTDTopologyMode("TopologyMode", sv);
    ptp.topologyMode_ = topoMode;
  } else {                
    throw cms::Exception( "MTDParametersFromDD" ) << "Not found "<< attribute.c_str() << " but needed.";  }

  return true;
}

void
MTDParametersFromDD::putOne( int subdet, std::vector<int> & vpars, PMTDParameters& ptp )
{
  PMTDParameters::Item item;
  item.id_ = subdet;
  item.vpars_ = vpars;
  ptp.vitems_.emplace_back( item );
}
