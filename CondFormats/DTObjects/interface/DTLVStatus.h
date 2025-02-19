#ifndef DTLVStatus_H
#define DTLVStatus_H
/** \class DTLVStatus
 *
 *  Description:
 *       Class to hold CCB status
 *
 *  $Date: 2010/01/20 18:20:08 $
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
#include "CondFormats/DTObjects/interface/DTBufferTree.h"
class DTChamberId;

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

  int flagCFE;
  int flagDFE;
  int flagCMC;
  int flagDMC;

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
           int&  flagCFE,
           int&  flagDFE,
           int&  flagCMC,
           int&  flagDMC ) const;
  int get( const DTChamberId& id,
           int&  flagCFE,
           int&  flagDFE,
           int&  flagCMC,
           int&  flagDMC ) const;
  /// access version
  const
  std::string& version() const;
  std::string& version();

  /// reset content
  void clear();

  int set( int   wheelId,
           int stationId,
           int  sectorId,
           int   flagCFE,
           int   flagDFE,
           int   flagCMC,
           int   flagDMC );
  int set( const DTChamberId& id,
           int   flagCFE,
           int   flagDFE,
           int   flagCMC,
           int   flagDMC );
  int setFlagCFE( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      flag );
  int setFlagCFE( const DTChamberId& id,
                  int   flag );
  int setFlagDFE( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      flag );
  int setFlagDFE( const DTChamberId& id,
                  int   flag );
  int setFlagCMC( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      flag );
  int setFlagCMC( const DTChamberId& id,
                  int   flag );
  int setFlagDMC( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      flag );
  int setFlagDMC( const DTChamberId& id,
                  int   flag );

  /// Access methods to data
  typedef std::vector< std::pair<DTLVStatusId,
                                 DTLVStatusData> >::const_iterator
                                                    const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  std::string dataVersion;

  std::vector< std::pair<DTLVStatusId,DTLVStatusData> > dataList;

  DTBufferTree<int,int>* dBuf;

  /// read and store full content
  void cacheMap() const;
  std::string mapName() const;

};


#endif // DTLVStatus_H

