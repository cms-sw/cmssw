#ifndef DTDataBuffer_H
#define DTDataBuffer_H
/** \class DTDataBuffer
 *
 *  Description:
 *       Class to hold drift tubes T0s
 *
 *  $Date: 2005/10/14 16:00:00 $
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
  T* openBuffer( const std::string& bType,
                 const std::string& bName,
                 const T& init );
  static const
  T* findBuffer( const std::string& bType,
                 const std::string& bName );

  /// set content
  static
  void insertCellData( const std::string& name,
                       int   wheelId,
                       int stationId,
                       int  sectorId,
                       int      slId,
                       int   layerId,
                       int    cellId,
                       const T& current,
                       const T& init );
  static
  void insertLayerData( const std::string& name,
                        int   wheelId,
                        int stationId,
                        int  sectorId,
                        int      slId,
                        int   layerId,
                        const T& current,
                        const T& init );
  static
  void insertSLData( const std::string& name,
                     int   wheelId,
                     int stationId,
                     int  sectorId,
                     int      slId,
                     const T& current,
                     const T& init );
  static
  void insertChamberData( const std::string& name,
                          int   wheelId,
                          int stationId,
                          int  sectorId,
                          const T& current,
                          const T& init );
  static
  void insertTDCChannelData( const std::string& name,
                             int        dduId,
                             int        rosId,
                             int        robId,
                             int        tdcId,
                             int tdcChannelId,
                             const T& current,
                             const T& init );

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
                 const T& init );
    ~NamedBuffer();

    const std::string& type() const;
    const std::string& name() const;
    T* data() const;

   private:

    std::string bufferType;
    std::string bufferName;
    T* dataBufPtr;

  };

  typedef typename std::vector<NamedBuffer*> buffer_type;
  typedef typename std::vector<NamedBuffer*>::iterator buf_iter;
  typedef typename std::vector<NamedBuffer*>::const_iterator const_iter;

  static int geometryGlobalId( int   wheelId,
                               int stationId,
                               int  sectorId,
                               int      slId,
                               int   layerId,
                               int    cellId );
  static int geometryGlobalId( int   wheelId,
                               int stationId,
                               int  sectorId,
                               int      slId,
                               int   layerId );
  static int geometryGlobalId( int   wheelId,
                               int stationId,
                               int  sectorId,
                               int      slId );
  static int geometryGlobalId( int   wheelId,
                               int stationId,
                               int  sectorId );
  static int readoutGlobalId( int        dduId,
                              int        rosId,
                              int        robId,
                              int        tdcId,
                              int tdcChannelId );

  static std::vector<NamedBuffer*> dataBuffer;
  static T defaultObject;

  static int nMaxWheel;
  static int nMaxStation;
  static int nMaxSector;
  static int nMaxSL;
  static int nMaxLayer;
  static int nMaxCell;
  static int nMaxDDU;
  static int nMaxROS;
  static int nMaxROB;
  static int nMaxTDC;
  static int nMaxTDCChannel;

  static int nMinWheel;
  static int nMinStation;
  static int nMinSector;
  static int nMinSL;
  static int nMinLayer;
  static int nMinCell;
  static int nMinDDU;
  static int nMinROS;
  static int nMinROB;
  static int nMinTDC;
  static int nMinTDCChannel;

};

#include "DTDataBuffer.icc"

#endif // DTDataBuffer_H

