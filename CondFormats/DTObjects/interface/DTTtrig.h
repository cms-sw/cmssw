#ifndef DTTtrig_H
#define DTTtrig_H
/** \class DTTtrig
 *
 *  Description:
 *       Class to hold drift tubes TTrigs
 *             ( SL by SL time offsets )
 *
 *  $Date: 2005/12/01 12:48:05 $
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
  int     tTrig;

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
  /// read and store full content
  void initSetup() const;

  /// get content
  int slTtrig( int   wheelId,
               int stationId,
               int  sectorId,
               int      slId,
               int&    tTrig ) const;

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
                  int     tTrig );

  /// Access methods to data
  typedef std::vector<DTSLTtrigData>::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  std::string dataVersion;

  std::vector<DTSLTtrigData> slData;
  void getIdNumbers( int& minWheel,  int& minStation,
                     int& minSector, int& minSL,
                     int& maxWheel,  int& maxStation,
                     int& maxSector, int& maxSL      ) const;

};


#endif // DTTtrig_H

