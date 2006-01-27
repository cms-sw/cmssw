#ifndef DTMtime_H
#define DTMtime_H
/** \class DTMtime
 *
 *  Description:
 *       Class to hold drift tubes mean-times
 *             ( SL by SL mean-time calculation )
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

class DTSLMtimeData {

 public:

  DTSLMtimeData();
  ~DTSLMtimeData();

  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int     mTime;
  int     mTrms;

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
  /// read and store full content
  void initSetup() const;

  /// get content
  int slMtime( int   wheelId,
               int stationId,
               int  sectorId,
               int      slId,
               int&    mTime,
               float&  mTrms ) const;

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
                  int     mTime,
                  float   mTrms );

  /// Access methods to data
  typedef std::vector<DTSLMtimeData>::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  std::string dataVersion;

  std::vector<DTSLMtimeData> slData;
  void getIdNumbers( int& minWheel,  int& minStation,
                     int& minSector, int& minSL,
                     int& maxWheel,  int& maxStation,
                     int& maxSector, int& maxSL      ) const;

  static int rmsFactor;

};


#endif // DTMtime_H

