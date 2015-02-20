#include "Geometry/TrackerGeometryBuilder/interface/TrackerParametersFromDD.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Base/interface/DDutils.h"

bool
TrackerParametersFromDD::build( const DDCompactView* cvp,
				PTrackerParameters& ptp)
{
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

  ptp.vpars = dbl_to_int( DDVectorGetter::get( "vPars" ));

  return true;
}

void
TrackerParametersFromDD::putOne( int subdet, std::vector<int> & vpars, PTrackerParameters& ptp )
{
  PTrackerParameters::Item item;
  item.id = subdet;
  item.vpars = vpars;
  ptp.vitems.push_back( item );
}
