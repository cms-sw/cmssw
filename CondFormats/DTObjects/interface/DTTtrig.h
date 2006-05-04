#ifndef DTTtrig_H
#define DTTtrig_H
/** \class DTTtrig
 *
 *  Description:
 *       Class to hold drift tubes TTrigs
 *             ( SL by SL time offsets )
 *
 *  $Date: 2006/02/28 18:06:29 $
 *  $Revision: 1.3 $
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
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTSLTtrigData {

 public:

  DTSLTtrigData();
  ~DTSLTtrigData();

  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  float   tTrig;
  float   tTrms;

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
  typedef std::vector<DTSLTtrigData>::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  /// read and store full content
  void initSetup() const;

  std::string dataVersion;
  float nsPerCount;

  std::vector<DTSLTtrigData> slData;

};


#endif // DTTtrig_H

