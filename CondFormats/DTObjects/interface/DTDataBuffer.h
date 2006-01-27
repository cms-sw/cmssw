#ifndef DTDataBuffer_H
#define DTDataBuffer_H
/** \class DTDataBuffer
 *
 *  Description:
 *       Class to hold drift tubes T0s
 *
 *  $Date: 2005/11/14 19:17:18 $
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

template <class T>
class DTDataBuffer {

 public:

  /** Constructor
   */
  DTDataBuffer();

  /** Destructor
   */
  ~DTDataBuffer();

  /** Operations
   */
  /// access internal buffer
  static
  bool openBuffer( const std::string& bType,
                   const std::string& bName,
                   int min1, int min2, int min3, int min4, int min5, int min6, 
                   int max1, int max2, int max3, int max4, int max5, int max6, 
                   const T& init );
  static
  bool findBuffer( const std::string& bType,
                   const std::string& bName );

  /// set content
  static
  bool insertCellData( const std::string& name,
                       int   wheelId,
                       int stationId,
                       int  sectorId,
                       int      slId,
                       int   layerId,
                       int    cellId,
                       const T& current );
  bool insertLayerData( const std::string& name,
                        int   wheelId,
                        int stationId,
                        int  sectorId,
                        int      slId,
                        int   layerId,
                        const T& current );
  static
  bool insertSLData( const std::string& name,
                     int   wheelId,
                     int stationId,
                     int  sectorId,
                     int      slId,
                     const T& current );
  static
  bool insertChamberData( const std::string& name,
                          int   wheelId,
                          int stationId,
                          int  sectorId,
                          const T& current );
  static
  bool insertTDCChannelData( const std::string& name,
                             int        dduId,
                             int        rosId,
                             int        robId,
                             int        tdcId,
                             int tdcChannelId,
                             const T& current );

  /// get content
  static
  const T& getCellData( const std::string& name,
                        int   wheelId,
                        int stationId,
                        int  sectorId,
                        int      slId,
                        int   layerId,
                        int    cellId );
  static
  const T& getLayerData( const std::string& name,
                         int   wheelId,
                         int stationId,
                         int  sectorId,
                         int      slId,
                         int   layerId );
  static
  const T& getSLData( const std::string& name,
                      int   wheelId,
                      int stationId,
                      int  sectorId,
                      int      slId );
  static
  const T& getChamberData( const std::string& name,
                           int   wheelId,
                           int stationId,
                           int  sectorId );
  static
  const T& getTDCChannelData( const std::string& name,
                              int        dduId,
                              int        rosId,
                              int        robId,
                              int        tdcId,
                              int tdcChannelId );

 private:

  class NamedBuffer {

   public:

    NamedBuffer( const std::string& bType,
                 const std::string& bName,
                 int min1, int min2, int min3, int min4, int min5, int min6, 
                 int num1, int num2, int num3, int num4, int num5, int num6, 
                 const T& init );
    ~NamedBuffer();

    const std::string& type() const;
    const std::string& name() const;

    std::string bufferType;
    std::string bufferName;
    T* data;

    int nMaxWheel;
    int nMaxStation;
    int nMaxSector;
    int nMaxSL;
    int nMaxLayer;
    int nMaxCell;
    int nMaxDDU;
    int nMaxROS;
    int nMaxROB;
    int nMaxTDC;
    int nMaxTDCChannel;

    int nMinWheel;
    int nMinStation;
    int nMinSector;
    int nMinSL;
    int nMinLayer;
    int nMinCell;
    int nMinDDU;
    int nMinROS;
    int nMinROB;
    int nMinTDC;
    int nMinTDCChannel;

    int geometryGlobalId( int   wheelId,
                          int stationId,
                          int  sectorId,
                          int      slId,
                          int   layerId,
                          int    cellId );
    int geometryGlobalId( int   wheelId,
                          int stationId,
                          int  sectorId,
                          int      slId,
                          int   layerId );
    int geometryGlobalId( int   wheelId,
                          int stationId,
                          int  sectorId,
                          int      slId );
    int geometryGlobalId( int   wheelId,
                          int stationId,
                          int  sectorId );
    int  readoutGlobalId( int        dduId,
                          int        rosId,
                          int        robId,
                          int        tdcId,
                          int tdcChannelId );


   private:

  };

  typedef typename std::vector<NamedBuffer*> buffer_type;
  typedef typename std::vector<NamedBuffer*>::iterator buf_iter;
  typedef typename std::vector<NamedBuffer*>::const_iterator const_iter;

  static std::vector<NamedBuffer*> dataBuffer;
  static T defaultObject;

  static NamedBuffer* findNamedBuffer( const std::string& bType,
                                       const std::string& bName );

};

#include "DTDataBuffer.icc"

#endif // DTDataBuffer_H

