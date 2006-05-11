#ifndef DTRangeT0_H
#define DTRangeT0_H
/** \class DTRangeT0
 *
 *  Description:
 *       Class to hold drift tubes T0 range
 *             ( SL by SL time offsets )
 *
 *  $Date: 2006/02/28 16:00:00 $
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
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTSLRangeT0Data {

 public:

  DTSLRangeT0Data();
  ~DTSLRangeT0Data();

  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int     t0min;
  int     t0max;

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
  typedef std::vector<DTSLRangeT0Data>::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  /// read and store full content
  void initSetup() const;

  std::string dataVersion;

  std::vector<DTSLRangeT0Data> slData;

};


#endif // DTRangeT0_H

