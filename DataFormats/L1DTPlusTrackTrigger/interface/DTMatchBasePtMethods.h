#ifndef DTMatchBasePtMethods_h
#define DTMatchBasePtMethods_h

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

#include "DataFormats/L1DTPlusTrackTrigger/interface/DTMatchPt.h"

#include <map>
#include <string>

/// Class implementation
class DTMatchBasePtMethods
{
  public :

    /// Trivial default constructor
    DTMatchBasePtMethods();

    /// Copy constructor
    DTMatchBasePtMethods( const DTMatchBasePtMethods& aDTMBPt );

    /// Assignment operator
    DTMatchBasePtMethods& operator = ( const DTMatchBasePtMethods& aDTMBPt );

    /// Destructor
    virtual ~DTMatchBasePtMethods(){};

    /// Methods to get quantities
    float const getPt( std::string const aLabel ) const;
    float const getAlpha0( std::string const aLabel ) const;
    float const getD( std::string const aLabel ) const;

    /// Method to retrieve the map
    inline std::map< std::string, DTMatchPt* > getPtMethodsMap() const
    {
      return thePtMethodsMap;
    }

    /// Method to assign the Pt Method to the map
    void addPtMethod( std::string const aLabel, DTMatchPt* aDTMPt );

  protected :

    /// Data members
    std::map< std::string, DTMatchPt* > thePtMethodsMap;
};

#endif

