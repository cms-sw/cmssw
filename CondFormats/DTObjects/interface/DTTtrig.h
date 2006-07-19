#ifndef DTTtrig_H
#define DTTtrig_H
/** \class DTTtrig
 *
 *  Description:
 *       Class to hold drift tubes TTrigs
 *             ( SL by SL time offsets )
 *
 *  $Date: 2006/05/04 06:54:02 $
 *  $Revision: 1.4 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "CondFormats/DTObjects/interface/DTTimeUnits.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"

//---------------
// C++ Headers --
//---------------
#include <string>
//#include <vector>
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTTtrigId   {

 public:

  DTTtrigId();
  ~DTTtrigId();

  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;

};


class DTTtrigData {

 public:

  DTTtrigData();
  ~DTTtrigData();

  float tTrig;
  float tTrms;

};


class DTTtrigCompare {
 public:
  bool operator()( const DTTtrigId& idl,
                   const DTTtrigId& idr ) const;
};


class DTTtrig {

 public:

  /** Constructor
   */
  DTTtrig();
  DTTtrig( const std::string& version );

  /** Destructor
   */
  ~DTTtrig();

  /** Operations
   */
  /// get content
  int slTtrig( int   wheelId,
               int stationId,
               int  sectorId,
               int      slId,
               float&  tTrig,
               float&  tTrms,
               DTTimeUnits::type unit = DTTimeUnits::counts ) const;
  int slTtrig( const DTSuperLayerId& id,
               float&  tTrig,
               float&  tTrms,
               DTTimeUnits::type unit = DTTimeUnits::counts ) const;
  float unit() const;

  /// access version
  const
  std::string& version() const;
  std::string& version();

  /// reset content
  void clear();

  int setSLTtrig( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  float   tTrig,
                  float   tTrms,
                  DTTimeUnits::type unit = DTTimeUnits::counts );
  int setSLTtrig( const DTSuperLayerId& id,
                  float   tTrig,
                  float   tTrms,
                  DTTimeUnits::type unit = DTTimeUnits::counts );
  void setUnit( float unit );

  /// Access methods to data
  typedef std::map<DTTtrigId,
                   DTTtrigData,
                   DTTtrigCompare>::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  std::string dataVersion;
  float nsPerCount;

  std::map<DTTtrigId,DTTtrigData,DTTtrigCompare> slData;

};


#endif // DTTtrig_H

