#ifndef DTMtime_H
#define DTMtime_H
/** \class DTMtime
 *
 *  Description:
 *       Class to hold drift tubes mean-times
 *             ( SL by SL mean-time calculation )
 *
 *  $Date: 2006/05/17 10:33:51 $
 *  $Revision: 1.5 $
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
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTMtimeId {

 public:

  DTMtimeId();
  ~DTMtimeId();

  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;

};


class DTMtimeData {

 public:

  DTMtimeData();
  ~DTMtimeData();

  float mTime;
  float mTrms;

};


class DTMtimeCompare {
 public:
  bool operator()( const DTMtimeId& idl,
                   const DTMtimeId& idr ) const;
};


class DTMtime {

 public:

  /** Constructor
   */
  DTMtime();
  DTMtime( const std::string& version );

  /** Destructor
   */
  ~DTMtime();

  /** Operations
   */
  /// get content
  int slMtime( int   wheelId,
               int stationId,
               int  sectorId,
               int      slId,
               float&  mTime,
               float&  mTrms,
               DTTimeUnits::type unit = DTTimeUnits::counts ) const;
  int slMtime( const DTSuperLayerId& id,
               float&  mTime,
               float&  mTrms,
               DTTimeUnits::type unit = DTTimeUnits::counts ) const;
  float unit() const;

  /// access version
  const
  std::string& version() const;
  std::string& version();

  /// reset content
  void clear();

  int setSLMtime( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  float   mTime,
                  float   mTrms,
                  DTTimeUnits::type unit = DTTimeUnits::counts );
  int setSLMtime( const DTSuperLayerId& id,
                  float   mTime,
                  float   mTrms,
                  DTTimeUnits::type unit = DTTimeUnits::counts );
  void setUnit( float unit );

  /// Access methods to data
  typedef std::map<DTMtimeId,
                   DTMtimeData,
                   DTMtimeCompare>::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  std::string dataVersion;
  float nsPerCount;

  std::map<DTMtimeId,DTMtimeData,DTMtimeCompare> slData;

};


#endif // DTMtime_H

