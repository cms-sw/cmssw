#include "Geometry/TrackerGeometryBuilder/interface/TrackerParametersFromDD.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Base/interface/DDutils.h"

bool
TrackerParametersFromDD::build( const DDCompactView* cvp,
				PTrackerParameters& ptp)
{
  std::vector<int> pxbPars = dbl_to_int( DDVectorGetter::get( "pxbPars" ));
  putOne( GeometricDet::PixelBarrel, pxbPars, ptp );

  std::vector<int> pxfPars = dbl_to_int( DDVectorGetter::get( "pxfPars" ));
  putOne( GeometricDet::PixelEndCap, pxfPars, ptp );

  std::vector<int> tecPars = dbl_to_int( DDVectorGetter::get( "tecPars" ));
  putOne( GeometricDet::TEC, tecPars, ptp );

  std::vector<int> tibPars = dbl_to_int( DDVectorGetter::get( "tibPars" ));
  putOne( GeometricDet::TIB, tibPars, ptp );

  std::vector<int> tidPars = dbl_to_int( DDVectorGetter::get( "tidPars" ));
  putOne( GeometricDet::TID, tidPars, ptp );

  std::vector<int> tobPars = dbl_to_int( DDVectorGetter::get( "tobPars" ));
  putOne( GeometricDet::TOB, tobPars, ptp );

  std::vector<int> vpars  = dbl_to_int( DDVectorGetter::get( "vPars" ));
  putOne( GeometricDet::Tracker, vpars, ptp );

  return true;
}

void
TrackerParametersFromDD::putOne( GeometricDet::GeometricEnumType det, std::vector<int> & vpars, PTrackerParameters& ptp )
{
  PTrackerParameters::Item item;
  item.id = det;
  item.vpars = vpars;
  ptp.vitems.push_back( item );
}
