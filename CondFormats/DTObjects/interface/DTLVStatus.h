#ifndef DTLVStatus_H
#define DTLVStatus_H
/** \class DTLVStatus
 *
 *  Description:
 *       Class to hold CCB status
 *
 *  $Date: 2008/11/20 12:00:00 $
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
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTLVStatusId {

 public:

  DTLVStatusId();
  ~DTLVStatusId();

  int   wheelId;
  int stationId;
  int  sectorId;

};


class DTLVStatusData {

 public:

  DTLVStatusData();
  ~DTLVStatusData();

  int flag;

};


class DTLVStatus {

 public:

  /** Constructor
   */
  DTLVStatus();
  DTLVStatus( const std::string& version );

  /** Destructor
   */
  ~DTLVStatus();

  /** Operations
   */
  /// get content
  int get( int   wheelId,
           int stationId,
           int  sectorId,
           int&     flag ) const;
  int get( const DTChamberId& id,
           int&     flag ) const;
  /// access version
  const
  std::string& version() const;
  std::string& version();

  /// reset content
  void clear();

  int set( int   wheelId,
           int stationId,
           int  sectorId,
           int      flag );
  int set( const DTChamberId& id,
           int      flag );


  /// Access methods to data
  typedef std::vector< std::pair<DTLVStatusId,
                                 DTLVStatusData> >::const_iterator
                                                     const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  std::string dataVersion;

  std::vector< std::pair<DTLVStatusId,DTLVStatusData> > dataList;

  /// read and store full content
  void cacheMap() const;
  std::string mapName() const;

};


#endif // DTLVStatus_H

