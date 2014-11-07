/*! \class DTMatchBasePtMethods
 *  \author Ignazio Lazzizzera
 *  \author Nicola Pozzobon
 *  \brief DT local triggers matched together, base class
 *         for DTMatchBase, which is abstract base class for
 *         DTMatch. For a given object, several ways of getting its
 *         Pt are available. Each is accessed through a string.
 *         Various ways of obtaining muon Pt using stubs and tracks.
 *         NOTE: this is just a container class. Nothing else.
 *  \date 2010, Apr 10
 */

#include "DataFormats/L1DTPlusTrackTrigger/interface/DTMatchBasePtMethods.h"

/// Trivial constructor
DTMatchBasePtMethods::DTMatchBasePtMethods()
{
  thePtMethodsMap = std::map< std::string, DTMatchPt* >();
}

/// Copy constructor
DTMatchBasePtMethods::DTMatchBasePtMethods( const DTMatchBasePtMethods& aDTMBPt )
{
  thePtMethodsMap = aDTMBPt.getPtMethodsMap();
}

/// Assignment operator
DTMatchBasePtMethods& DTMatchBasePtMethods::operator = ( const DTMatchBasePtMethods& aDTMBPt )
{
  if ( this == &aDTMBPt ) /// Same object?
  {
    return *this;
  }
  thePtMethodsMap = aDTMBPt.getPtMethodsMap();
  return *this;
}

/// Method to assign the Pt Method to the map
void DTMatchBasePtMethods::addPtMethod( std::string const aLabel, DTMatchPt* aDTMPt )
{
  /// Store only non-NAN values of Pt
  if ( isnan(aDTMPt->getPt()) )
  {
    return;
  }

  if ( thePtMethodsMap.find( aLabel ) == thePtMethodsMap.end() )
  {
    thePtMethodsMap.insert( std::make_pair( aLabel, aDTMPt ) );
  }
  else
  {
    thePtMethodsMap[ aLabel ] = aDTMPt;
  }
  return;
}

/// Methods to get quantities
float const DTMatchBasePtMethods::getPt( std::string const aLabel ) const
{
  if ( thePtMethodsMap.find( aLabel ) != thePtMethodsMap.end() )
  {
    return thePtMethodsMap.find( aLabel )->second->getPt();
  }
  return NAN;
} 

float const DTMatchBasePtMethods::getAlpha0( std::string const aLabel ) const
{
  if ( thePtMethodsMap.find( aLabel ) != thePtMethodsMap.end() )
  {
    return thePtMethodsMap.find( aLabel )->second->getAlpha0();
  }
  return NAN;
} 

float const DTMatchBasePtMethods::getD( std::string const aLabel ) const
{
  if ( thePtMethodsMap.find( aLabel ) != thePtMethodsMap.end() )
  {
    return thePtMethodsMap.find( aLabel )->second->getD();

  }
  return NAN;
}
 
