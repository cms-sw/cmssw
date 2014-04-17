#ifndef DTTPGParameters_H
#define DTTPGParameters_H
/** \class DTTPGParameters
 *
 *  Description:
 *       Class to hold drift tubes TPG parameters
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
#include "CondFormats/DTObjects/interface/DTTimeUnits.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
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

class DTTPGParametersId   {

 public:

  DTTPGParametersId();
  ~DTTPGParametersId();

  int   wheelId;
  int stationId;
  int  sectorId;

};


class DTTPGParametersData {

 public:

  DTTPGParametersData();
  ~DTTPGParametersData();

  int   nClock;
  float tPhase;

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

  void initialize();

  DTTPGParameters& operator=(DTTPGParameters const&);

 private:

  DTTPGParameters(DTTPGParameters const&);

  std::string dataVersion;
  float nsPerCount;
  int   clockLength;

  std::vector< std::pair<DTTPGParametersId,DTTPGParametersData> > dataList;

  edm::ConstRespectingPtr<DTBufferTree<int,int> > dBuf;

  std::string mapName() const;

};
#endif // DTTPGParameters_H
