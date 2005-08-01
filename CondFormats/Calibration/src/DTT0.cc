/*
 *  See header file for a description of this class.
 *
 *  $Date: 2004/08/04 12:00:00 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondFormats/Calibration/interface/DTT0.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------


//---------------
// C++ Headers --
//---------------


//-------------------
// Initializations --
//-------------------
int DTT0::nMinWheel   = -2;
int DTT0::nMaxWheel   =  5;
int DTT0::nMaxStation =  4;
int DTT0::nMaxSector  = 14;
int DTT0::nMaxSL      =  3;
int DTT0::nMaxLayer   =  4;
int DTT0::nMaxCell    = 99;

//----------------
// Constructors --
//----------------
DTT0::DTT0() {
  int nChambers = nMaxWheel * nMaxStation * nMaxSector;
  chambers.reserve( nChambers );
  std::vector<ChamberData>::iterator chamberIter = chambers.begin();
  std::vector<ChamberData>::iterator chamberIend = chambers.end();
  while ( chamberIter != chamberIend ) {
    ChamberData& chamberData = *chamberIter++;
    chamberData.clockPhase = 0;
  }
  int nSL = nMaxSL * nChambers;
  superLayers.reserve( nMaxSL );
  std::vector<SLData>::iterator slIter = superLayers.begin();
  std::vector<SLData>::iterator slIend = superLayers.end();
  while ( slIter != slIend ) {
    SLData& slData = *slIter++;
    slData.slTimeOffsetMean  = 0.0;
    slData.slTimeOffsetSigma = 0.0;
    slData.pulseOffsetEvenVsOdd = 0.0;
  }
  int nLayers = nMaxLayer * nSL;
  //layers.reserve( nLayers );
  /*
  std::vector<LayerData>::iterator layerIter = layers.begin();
  std::vector<LayerData>::iterator layerIend = layers.end();
  while ( layerIter != layerIend ) {
    LayerData& layerData = *layerIter++;
  }
  */
  int nCells = nMaxCell * nLayers;
  cells.reserve( nCells );
  std::vector<CellData>::iterator cellIter = cells.begin();
  std::vector<CellData>::iterator cellIend = cells.end();
  while ( cellIter != cellIend ) {
    CellData& cellData = *cellIter++;
    cellData.t0Mean  = 0.0;
    cellData.t0Sigma = 0.0;
  }
}

//--------------
// Destructor --
//--------------
DTT0::~DTT0() {
}

//--------------
// Operations --
//--------------
int DTT0::chamberPhaseByGeometryId( int   wheelId,
                                    int stationId,
                                    int  sectorId ) const {
  return chambers[ chamberPointerByGeometryId( wheelId, 
                                             stationId,
                                              sectorId ) ].clockPhase;
}


std::pair<float,float>
    DTT0::slTimeOffsetByGeometryId( int   wheelId,
                                    int stationId,
                                    int  sectorId,
                                    int      slId ) const {
  const SLData& slData = superLayers[
                slPointerByGeometryId( wheelId,
                                     stationId,
                                      sectorId,
                                          slId ) ];
  return std::pair<float,float>( slData.slTimeOffsetMean,
                                 slData.slTimeOffsetSigma );
}


float DTT0::pulseEvenVsOddByGeometryId( int   wheelId,
                                        int stationId,
                                        int  sectorId,
                                        int      slId ) const {
  return superLayers[ slPointerByGeometryId( wheelId,
                                           stationId,
                                            sectorId,
                                                slId ) ].pulseOffsetEvenVsOdd;
}


std::pair<float,float>
    DTT0::cellT0ByGeometryId( int   wheelId,
                              int stationId,
                              int  sectorId,
                              int      slId,
                              int   layerId,
                              int    cellId ) const {
  const CellData& cellData = cells[
                  cellPointerByGeometryId( wheelId, 
                                         stationId,
                                          sectorId,
                                              slId,
                                           layerId,
                                            cellId ) ];
  return std::pair<float,float>( cellData.t0Mean,
                                 cellData.t0Sigma );
}


void DTT0::clear() {
  std::vector<ChamberData>::iterator chamberIter = chambers.begin();
  std::vector<ChamberData>::iterator chamberIend = chambers.end();
  while ( chamberIter != chamberIend ) {
    ChamberData& chamberData = *chamberIter++;
    chamberData.clockPhase = 0;
  }
  std::vector<SLData>::iterator slIter = superLayers.begin();
  std::vector<SLData>::iterator slIend = superLayers.end();
  while ( slIter != slIend ) {
    SLData& slData = *slIter++;
    slData.slTimeOffsetMean  = 0.0;
    slData.slTimeOffsetSigma = 0.0;
    slData.pulseOffsetEvenVsOdd = 0.0;
  }
  /*
  std::vector<LayerData>::iterator layerIter = layers.begin();
  std::vector<LayerData>::iterator layerIend = layers.end();
  while ( layerIter != layerIend ) {
    LayerData& layerData = *layerIter++;
  }
  */
  std::vector<CellData>::iterator cellIter = cells.begin();
  std::vector<CellData>::iterator cellIend = cells.end();
  while ( cellIter != cellIend ) {
    CellData& cellData = *cellIter++;
    cellData.t0Mean  = 0.0;
    cellData.t0Sigma = 0.0;
  }
  return;
}


void DTT0::setChamberPhaseByGeometryId( int   wheelId,
                                        int stationId,
                                        int  sectorId,
                                        int     phase ) {
  chambers[ chamberPointerByGeometryId( wheelId, 
                                        stationId,
                                        sectorId ) ].clockPhase = phase;
}



void DTT0::setSLTimeOffsetByGeometryId( int   wheelId,
                                        int stationId,
                                        int  sectorId,
                                        int      slId,
                                        float    mean,
                                        float   sigma ) {
  ChamberData& chamberData =
               chambers[ chamberPointerByGeometryId( wheelId, 
                                                   stationId,
                                                    sectorId ) ];
  SLData& slData = superLayers[
          slPointerByGeometryId( wheelId,
                               stationId,
                                sectorId,
                                    slId ) ];
  slData.slTimeOffsetMean  = mean;
  slData.slTimeOffsetSigma = sigma;
  return;
}


void DTT0::setPulseEvenVsOddByGeometryId( int   wheelId,
                                          int stationId,
                                          int  sectorId,
                                          int      slId,
                                          float  offset ) {
  SLData& slData = superLayers[
          slPointerByGeometryId( wheelId,
                               stationId,
                                sectorId,
                                    slId ) ];
  slData.pulseOffsetEvenVsOdd = offset;
  return;
}


void DTT0::setCellT0ByGeometryId( int   wheelId,
                                  int stationId,
                                  int  sectorId,
                                  int      slId,
                                  int   layerId,
                                  int    cellId,
                                  float    mean,
                                  float   sigma ) {
  CellData& cellData = cells[
            cellPointerByGeometryId( wheelId, 
                                   stationId,
                                    sectorId,
                                        slId,
                                     layerId,
                                      cellId ) ];
  cellData.t0Mean  =  mean;
  cellData.t0Sigma = sigma;
  return;
}


int DTT0::chamberPointerByGeometryId( int   wheelId,
                                      int stationId,
                                      int  sectorId ) const {
  wheelId -= nMinWheel;
  return -- sectorId +
     ( ( --stationId +
       ( --  wheelId * nMaxStation ) )
                     * nMaxSector  );
}


int DTT0::slPointerByGeometryId( int   wheelId,
                                 int stationId,
                                 int  sectorId,
                                 int      slId ) const {
  wheelId -= nMinWheel;
  return --     slId +
     ( ( -- sectorId +
     ( ( --stationId +
       ( --  wheelId * nMaxStation ) )
                     * nMaxSector  ) )
                     * nMaxSL      );
}


int DTT0::layerPointerByGeometryId( int   wheelId,
                                    int stationId,
                                    int  sectorId,
                                    int      slId,
                                    int   layerId ) const {
  wheelId -= nMinWheel;
  return --  layerId +
     ( ( --     slId +
     ( ( -- sectorId +
     ( ( --stationId +
       ( --  wheelId * nMaxStation ) )
                     * nMaxSector  ) )
                     * nMaxSL      ) )
                     * nMaxLayer   );
}


int DTT0::cellPointerByGeometryId( int   wheelId,
                                   int stationId,
                                   int  sectorId,
                                   int      slId,
                                   int   layerId,
                                   int    cellId ) const {
  wheelId -= nMinWheel;
  return --   cellId +
     ( ( --  layerId +
     ( ( --     slId +
     ( ( -- sectorId +
     ( ( --stationId +
       ( --  wheelId * nMaxStation ) )
                     * nMaxSector  ) )
                     * nMaxSL      ) )
                     * nMaxLayer   ) )
                     * nMaxCell    );
}


