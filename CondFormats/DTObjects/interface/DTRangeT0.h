#ifndef DTRangeT0_H
#define DTRangeT0_H
/** \class DTRangeT0
 *
 *  Description:
 *       Class to hold drift tubes T0 range
 *             ( SL by SL min - max T0 )
 *
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
#include "FWCore/Utilities/interface/ConstRespectingPtr.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <utility>

template <class Key, class Content> class DTBufferTree;

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
                 int&    t0max ) const
      { return get( wheelId, stationId, sectorId, slId,
                    t0min, t0max ); };
  int slRangeT0( const DTSuperLayerId& id,
                 int&    t0min,
                 int&    t0max ) const
      { return get( id, t0min, t0max ); };
  int get( int   wheelId,
           int stationId,
           int  sectorId,
           int      slId,
           int&    t0min,
           int&    t0max ) const;
  int get( const DTSuperLayerId& id,
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
                    int     t0max )
      { return set( wheelId, stationId, sectorId, slId, t0min, t0max ); };
  int setSLRangeT0( const DTSuperLayerId& id,
                    int     t0min,
                    int     t0max )
      { return set( id, t0min, t0max ); };
  int set( int   wheelId,
           int stationId,
           int  sectorId,
           int      slId,
           int     t0min,
           int     t0max );
  int set( const DTSuperLayerId& id,
           int     t0min,
           int     t0max );

  /// Access methods to data
  typedef std::vector< std::pair<DTRangeT0Id,
                                 DTRangeT0Data> >::const_iterator
                                                   const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

  void initialize();

 private:

  DTRangeT0(DTRangeT0 const&);
  DTRangeT0& operator=(DTRangeT0 const&);

  std::string dataVersion;

  std::vector< std::pair<DTRangeT0Id,DTRangeT0Data> > dataList;

  edm::ConstRespectingPtr<DTBufferTree<int,int> > dBuf;

  /// read and store full content
  std::string mapName() const;

};
#endif // DTRangeT0_H
