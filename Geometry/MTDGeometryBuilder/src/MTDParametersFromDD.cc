#include "Geometry/MTDGeometryBuilder/interface/MTDParametersFromDD.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"
#include "CondFormats/GeometryObjects/interface/PMTDParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Core/interface/DDutils.h"

namespace {
  int getMTDTopologyMode(const char* s, const DDsvalues_type & sv) {
    DDValue val( s );
    if (DDfetch( &sv, val )) {
      const std::vector<std::string> & fvec = val.strings();
      if (fvec.empty()) {
        throw cms::Exception( "MTDParametersFromDD" ) << "Failed to get " << s << " tag.";
      }
 
      int result(-1);
      MTDStringToEnumParser<MTDTopologyMode::Mode> eparser;
      MTDTopologyMode::Mode mode = (MTDTopologyMode::Mode) eparser.parseString(fvec[0]);
      result = (int)(mode);
      return result;
    } else {
      throw cms::Exception( "MTDParametersFromDD" ) << "Failed to get "<< s << " tag.";
    }
  }
}

MTDParametersFromDD::MTDParametersFromDD(const edm::ParameterSet& pset) {
  const edm::VParameterSet& items = 
    pset.getParameterSetVector("vitems");
  pars_ = pset.getParameter<std::vector<int32_t> >("vpars");

  items_.resize(items.size());
  for( unsigned i = 0; i < items.size(); ++i) {
    auto& item = items_[i];
    item.id_ = i+1;
    item.vpars_ = items[i].getParameter<std::vector<int32_t> >("subdetPars");    
  }
}

bool
MTDParametersFromDD::build( const DDCompactView* cvp,
                            PMTDParameters& ptp)
{
  if( items_.empty() ) {
    for( int subdet = 1; subdet <= 6; ++subdet )
      {
        std::stringstream sstm;
        sstm << "Subdetector" << subdet;
        std::string name = sstm.str();
	
        if( DDVectorGetter::check( name ))
          {
            std::vector<int> subdetPars = dbl_to_int( DDVectorGetter::get( name ));
            putOne( subdet, subdetPars, ptp );
          }
      }
  } else {
    ptp.vitems_ = items_;
  }

  if( pars_.empty() ) {
    ptp.vpars_ = dbl_to_int( DDVectorGetter::get( "vPars" ));
  } else {
    ptp.vpars_ = pars_;
  }

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
