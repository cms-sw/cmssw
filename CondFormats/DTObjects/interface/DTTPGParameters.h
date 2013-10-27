#ifndef DTTPGParameters_H
#define DTTPGParameters_H
/** \class DTTPGParameters
 *
 *  Description:
 *       Class to hold drift tubes TPG parameters
 *
 *  $Date: 2009/03/06 14:09:14 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/DTObjects/interface/DTTimeUnits.h"
#include "CondFormats/DTObjects/interface/DTBufferTree.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTTPGParametersId   {

 public:

  DTTPGParametersId();
  ~DTTPGParametersId();

  int   wheelId;
  int stationId;
  int  sectorId;


 COND_SERIALIZABLE;
};


class DTTPGParametersData {

 public:

  DTTPGParametersData();
  ~DTTPGParametersData();

  int   nClock;
  float tPhase;


 COND_SERIALIZABLE;
};


class DTTPGParameters {

 public:

  /** Constructor
   */
  DTTPGParameters();
  DTTPGParameters( const std::string& version );

  /** Destructor
   */
  ~DTTPGParameters();

  /** Operations
   */
  /// get content
  int get( int   wheelId,
           int stationId,
           int  sectorId,
           int&    nc,
           float&  ph,
           DTTimeUnits::type unit ) const;
  int get( const DTChamberId& id,
           int&    nc,
           float&  ph,
           DTTimeUnits::type unit ) const;
  float totalTime( int   wheelId,
                   int stationId,
                   int  sectorId,
                   DTTimeUnits::type unit ) const;
  float totalTime( const DTChamberId& id,
                   DTTimeUnits::type unit ) const;
  int   clock() const;
  float unit() const;

  /// access version
  const
  std::string& version() const;
  std::string& version();

  /// reset content
  void clear();

  int set( int   wheelId,
           int stationId,
           int  sectorId,
           int    nc,
           float  ph,
           DTTimeUnits::type unit );
  int set( const DTChamberId& id,
           int    nc,
           float  ph,
           DTTimeUnits::type unit );
  void setClock( int clock );
  void setUnit( float unit );

  /// Access methods to data
  typedef std::vector< std::pair<DTTPGParametersId,
                                 DTTPGParametersData> >::const_iterator
                                                         const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  std::string dataVersion;
  float nsPerCount;
  int   clockLength;

  std::vector< std::pair<DTTPGParametersId,DTTPGParametersData> > dataList;

  DTBufferTree<int,int>* dBuf COND_TRANSIENT;

  /// read and store full content
  void cacheMap() const;
  std::string mapName() const;


 COND_SERIALIZABLE;
};


#endif // DTTPGParameters_H

