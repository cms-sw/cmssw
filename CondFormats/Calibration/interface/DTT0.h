#ifndef DTT0_H
#define DTT0_H
/** \class DTT0
 *
 *  Description:
 *       Class to hold drift tubes T0s
 *
 *  $Date: 2004/08/04 12:00:00 $
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
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTT0 {

public:

  /** Constructor
   */
  DTT0();

  /** Destructor
   */
  ~DTT0();

  /** Operations
   */
  /// get content
  int     chamberPhaseByGeometryId( int   wheelId,
                                    int stationId,
                                    int  sectorId ) const;
  std::pair<float,float>
          slTimeOffsetByGeometryId( int   wheelId,
                                    int stationId,
                                    int  sectorId,
                                    int      slId ) const;
  float pulseEvenVsOddByGeometryId( int   wheelId,
                                    int stationId,
                                    int  sectorId,
                                    int      slId ) const;
  std::pair<float,float>
                cellT0ByGeometryId( int   wheelId,
                                    int stationId,
                                    int  sectorId,
                                    int      slId,
                                    int   layerId,
                                    int    cellId ) const;
  /// reset content
  void clear();
  /// update content
  void   setChamberPhaseByGeometryId( int   wheelId,
                                      int stationId,
                                      int  sectorId,
                                      int     phase );
  void   setSLTimeOffsetByGeometryId( int   wheelId,
                                      int stationId,
                                      int  sectorId,
                                      int      slId,
                                      float    mean,
                                      float   sigma );
  void setPulseEvenVsOddByGeometryId( int   wheelId,
                                      int stationId,
                                      int  sectorId,
                                      int      slId,
                                      float  offset );
  void         setCellT0ByGeometryId( int   wheelId,
                                      int stationId,
                                      int  sectorId,
                                      int      slId,
                                      int   layerId,
                                      int    cellId,
                                      float    mean,
                                      float   sigma );

 public:

  struct CellData {
    //CellData() {};
    //~CellData() {};
    float t0Mean;
    float t0Sigma;
  };
  /*
  struct LayerData {
    LayerData() {};
    ~LayerData() {};
  };
  */
  struct SLData {
    //SLData() {};
    //~SLData() {};
    float slTimeOffsetMean;
    float slTimeOffsetSigma;
    float pulseOffsetEvenVsOdd;
  };

  struct ChamberData {
    //ChamberData() {};
    //~ChamberData() {};
    int clockPhase;
  };

  std::vector<   CellData> cells;
  //std::vector<  LayerData> layers;
  std::vector<     SLData> superLayers;
  std::vector<ChamberData> chambers;

  int chamberPointerByGeometryId( int   wheelId,
                                  int stationId,
                                  int  sectorId ) const;

  int      slPointerByGeometryId( int   wheelId,
                                  int stationId,
                                  int  sectorId,
                                  int      slId ) const;

  int   layerPointerByGeometryId( int   wheelId,
                                  int stationId,
                                  int  sectorId,
                                  int      slId,
                                  int   layerId ) const;

  int    cellPointerByGeometryId( int   wheelId,
                                  int stationId,
                                  int  sectorId,
                                  int      slId,
                                  int   layerId,
                                  int    cellId ) const;

  static int nMinWheel;
  static int nMaxWheel;
  static int nMaxStation;
  static int nMaxSector;
  static int nMaxSL;
  static int nMaxLayer;
  static int nMaxCell;

};


#endif // DTT0_H

