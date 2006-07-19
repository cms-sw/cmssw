#ifndef DTRangeT0_H
#define DTRangeT0_H
/** \class DTRangeT0
 *
 *  Description:
 *       Class to hold drift tubes T0 range
 *             ( SL by SL time offsets )
 *
 *  $Date: 2006/05/11 08:31:30 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTRangeT0Id {

 public:

  DTRangeT0Id();
  ~DTRangeT0Id();

  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;

};


class DTRangeT0Data {

 public:

  DTRangeT0Data();
  ~DTRangeT0Data();

  int t0min;
  int t0max;

};


class DTRangeT0Compare {
 public:
  bool operator()( const DTRangeT0Id& idl,
                   const DTRangeT0Id& idr ) const;
};


class DTRangeT0 {

 public:

  /** Constructor
   */
  DTRangeT0();
  DTRangeT0( const std::string& version );

  /** Destructor
   */
  ~DTRangeT0();

  /** Operations
   */
  /// get content
  int slRangeT0( int   wheelId,
                 int stationId,
                 int  sectorId,
                 int      slId,
                 int&    t0min,
                 int&    t0max ) const;
  int slRangeT0( const DTSuperLayerId& id,
                 int&    t0min,
                 int&    t0max ) const;

  /// access version
  const
  std::string& version() const;
  std::string& version();

  /// reset content
  void clear();

  int setSLRangeT0( int   wheelId,
                    int stationId,
                    int  sectorId,
                    int      slId,
                    int     t0min,
                    int     t0max );
  int setSLRangeT0( const DTSuperLayerId& id,
                    int     t0min,
                    int     t0max );

  /// Access methods to data
  typedef std::map<DTRangeT0Id,
                   DTRangeT0Data,
                   DTRangeT0Compare>::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  std::string dataVersion;

  std::map<DTRangeT0Id,DTRangeT0Data,DTRangeT0Compare> slData;

};


#endif // DTRangeT0_H

