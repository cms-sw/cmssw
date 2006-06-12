/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/05/17 18:04:57 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "CondFormats/DTObjects/interface/DTDataBuffer.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------


//---------------
// C++ Headers --
//---------------
#include <iostream>

//-------------------
// Initializations --
//-------------------

//----------------
// Constructors --
//----------------
DTStatusFlag::DTStatusFlag():
  dataVersion( " " ) {
}

DTStatusFlag::DTStatusFlag( const std::string& version ):
  dataVersion( version ) {
}

DTCellStatusFlagData::DTCellStatusFlagData() :
    wheelId( 0 ),
  stationId( 0 ),
   sectorId( 0 ),
       slId( 0 ),
    layerId( 0 ),
     cellId( 0 ),
  noiseFlag( false ),
     feMask( false ),
    tdcMask( false ),
   trigMask( false ),
   deadFlag( false ),
   nohvFlag( false ) {
}

//--------------
// Destructor --
//--------------
DTStatusFlag::~DTStatusFlag() {
  std::string statusVersionN = dataVersion + "_StatusN";
  std::string statusVersionF = dataVersion + "_StatusF";
  std::string statusVersionT = dataVersion + "_StatusT";
  std::string statusVersionR = dataVersion + "_StatusR";
  std::string statusVersionD = dataVersion + "_StatusD";
  std::string statusVersionH = dataVersion + "_StatusH";
  DTDataBuffer<int,bool>::dropBuffer( statusVersionN );
  DTDataBuffer<int,bool>::dropBuffer( statusVersionF );
  DTDataBuffer<int,bool>::dropBuffer( statusVersionT );
  DTDataBuffer<int,bool>::dropBuffer( statusVersionR );
  DTDataBuffer<int,bool>::dropBuffer( statusVersionD );
  DTDataBuffer<int,bool>::dropBuffer( statusVersionH );
}

DTCellStatusFlagData::~DTCellStatusFlagData() {
}

//--------------
// Operations --
//--------------
int DTStatusFlag::cellStatus( int   wheelId,
                              int stationId,
                              int  sectorId,
                              int      slId,
                              int   layerId,
                              int    cellId,
                              bool& noiseFlag,
                              bool&    feMask,
                              bool&   tdcMask,
                              bool&  trigMask,
                              bool&  deadFlag,
                              bool&  nohvFlag ) const {

  noiseFlag = false;
     feMask = false;
    tdcMask = false;
   deadFlag = false;
   nohvFlag = false;

  std::string statusVersionN = dataVersion + "_StatusN";
  std::string statusVersionF = dataVersion + "_StatusF";
  std::string statusVersionT = dataVersion + "_StatusT";
  std::string statusVersionR = dataVersion + "_StatusR";
  std::string statusVersionD = dataVersion + "_StatusD";
  std::string statusVersionH = dataVersion + "_StatusH";
  DTBufferTree<int,bool>* dataNBuf =
  DTDataBuffer<int,bool>::findBuffer( statusVersionN );
  DTBufferTree<int,bool>* dataFBuf =
  DTDataBuffer<int,bool>::findBuffer( statusVersionF );
  DTBufferTree<int,bool>* dataTBuf =
  DTDataBuffer<int,bool>::findBuffer( statusVersionT );
  DTBufferTree<int,bool>* dataRBuf =
  DTDataBuffer<int,bool>::findBuffer( statusVersionR );
  DTBufferTree<int,bool>* dataDBuf =
  DTDataBuffer<int,bool>::findBuffer( statusVersionD );
  DTBufferTree<int,bool>* dataHBuf =
  DTDataBuffer<int,bool>::findBuffer( statusVersionH );

  if ( dataNBuf == 0 ) {
    initSetup();
    dataNBuf = DTDataBuffer<int,bool>::findBuffer( statusVersionN );
  }

  if ( dataFBuf == 0 ) {
    initSetup();
    dataFBuf = DTDataBuffer<int,bool>::findBuffer( statusVersionF );
  }

  if ( dataTBuf == 0 ) {
    initSetup();
    dataTBuf = DTDataBuffer<int,bool>::findBuffer( statusVersionT );
  }

  if ( dataRBuf == 0 ) {
    initSetup();
    dataRBuf = DTDataBuffer<int,bool>::findBuffer( statusVersionR );
  }

  if ( dataDBuf == 0 ) {
    initSetup();
    dataDBuf = DTDataBuffer<int,bool>::findBuffer( statusVersionD );
  }

  if ( dataHBuf == 0 ) {
    initSetup();
    dataHBuf = DTDataBuffer<int,bool>::findBuffer( statusVersionH );
  }

  std::vector<int> cellKey;
  cellKey.push_back(   wheelId );
  cellKey.push_back( stationId );
  cellKey.push_back(  sectorId );
  cellKey.push_back(      slId );
  cellKey.push_back(   layerId );
  cellKey.push_back(    cellId );
//  noiseFlag = dataNBuf->find( cellKey.begin(), cellKey.end() );
//     feMask = dataFBuf->find( cellKey.begin(), cellKey.end() );
//    tdcMask = dataTBuf->find( cellKey.begin(), cellKey.end() );
  int searchStatusN =
      dataNBuf->find( cellKey.begin(), cellKey.end(), noiseFlag );
  int searchStatusF =
      dataFBuf->find( cellKey.begin(), cellKey.end(),    feMask );
  int searchStatusT =
      dataTBuf->find( cellKey.begin(), cellKey.end(),   tdcMask );
  int searchStatusR =
      dataTBuf->find( cellKey.begin(), cellKey.end(),  trigMask );
  int searchStatusD =
      dataFBuf->find( cellKey.begin(), cellKey.end(),  deadFlag );
  int searchStatusH =
      dataTBuf->find( cellKey.begin(), cellKey.end(),  nohvFlag );

//  return 1;
  return ( searchStatusN ||
           searchStatusF ||
           searchStatusT ||
           searchStatusR ||
           searchStatusD ||
           searchStatusH );

}


int DTStatusFlag::cellStatus( const DTWireId& id,
                              bool& noiseFlag,
                              bool&    feMask,
                              bool&   tdcMask,
                              bool&  trigMask,
                              bool&  deadFlag,
                              bool&  nohvFlag ) const {
  return cellStatus( id.wheel(),
                     id.station(),
                     id.sector(),
                     id.superLayer(),
                     id.layer(),
                     id.wire(),
                     noiseFlag,   feMask,  tdcMask,
                      trigMask, deadFlag, nohvFlag );
}


const
std::string& DTStatusFlag::version() const {
  return dataVersion;
}


std::string& DTStatusFlag::version() {
  return dataVersion;
}


void DTStatusFlag::clear() {
  cellData.clear();
  return;
}


int DTStatusFlag::setCellStatus( int   wheelId,
                                 int stationId,
                                 int  sectorId,
                                 int      slId,
                                 int   layerId,
                                 int    cellId,
                                 bool noiseFlag,
                                 bool    feMask,
                                 bool   tdcMask,
                                 bool  trigMask,
                                 bool  deadFlag,
                                 bool  nohvFlag ) {

  DTCellStatusFlagData data;
  data.  wheelId =   wheelId;
  data.stationId = stationId;
  data. sectorId =  sectorId;
  data.     slId =      slId;
  data.  layerId =   layerId;
  data.   cellId =    cellId;
  data.noiseFlag = noiseFlag;
  data.   feMask =    feMask;
  data.  tdcMask =   tdcMask;
  data. trigMask =  trigMask;
  data. deadFlag =  deadFlag;
  data. nohvFlag =  nohvFlag;

  cellData.push_back( data );

  std::string statusVersionN = dataVersion + "_StatusN";
  std::string statusVersionF = dataVersion + "_StatusF";
  std::string statusVersionT = dataVersion + "_StatusT";
  std::string statusVersionR = dataVersion + "_StatusR";
  std::string statusVersionD = dataVersion + "_StatusD";
  std::string statusVersionH = dataVersion + "_StatusH";

  DTBufferTree<int,bool>* dataNBuf =
  DTDataBuffer<int,bool>::openBuffer( statusVersionN );
  DTBufferTree<int,bool>* dataFBuf =
  DTDataBuffer<int,bool>::openBuffer( statusVersionF );
  DTBufferTree<int,bool>* dataTBuf =
  DTDataBuffer<int,bool>::openBuffer( statusVersionT );
  DTBufferTree<int,bool>* dataRBuf =
  DTDataBuffer<int,bool>::openBuffer( statusVersionR );
  DTBufferTree<int,bool>* dataDBuf =
  DTDataBuffer<int,bool>::openBuffer( statusVersionD );
  DTBufferTree<int,bool>* dataHBuf =
  DTDataBuffer<int,bool>::openBuffer( statusVersionH );

  std::vector<int> cellKey;
  cellKey.push_back(   wheelId );
  cellKey.push_back( stationId );
  cellKey.push_back(  sectorId );
  cellKey.push_back(      slId );
  cellKey.push_back(   layerId );
  cellKey.push_back(    cellId );

  dataNBuf->insert( cellKey.begin(), cellKey.end(), noiseFlag );
  dataFBuf->insert( cellKey.begin(), cellKey.end(),    feMask );
  dataTBuf->insert( cellKey.begin(), cellKey.end(),   tdcMask );
  dataRBuf->insert( cellKey.begin(), cellKey.end(),  trigMask );
  dataDBuf->insert( cellKey.begin(), cellKey.end(),  deadFlag );
  dataHBuf->insert( cellKey.begin(), cellKey.end(),  nohvFlag );

  return 0;

}


int DTStatusFlag::setCellStatus( const DTWireId& id,
                                 bool noiseFlag,
                                 bool    feMask,
                                 bool   tdcMask,
                                 bool  trigMask,
                                 bool  deadFlag,
                                 bool  nohvFlag  ) {
  return setCellStatus( id.wheel(),
                        id.station(),
                        id.sector(),
                        id.superLayer(),
                        id.layer(),
                        id.wire(),
                        noiseFlag,   feMask,  tdcMask,
                         trigMask, deadFlag, nohvFlag );
}


int DTStatusFlag::setCellNoise( int   wheelId,
                                int stationId,
                                int  sectorId,
                                int      slId,
                                int   layerId,
                                int    cellId,
                                bool flag ) {

  bool noiseFlag;
  bool    feMask;
  bool   tdcMask;
  bool  trigMask;
  bool  deadFlag;
  bool  nohvFlag;
  int status = cellStatus(   wheelId,
                           stationId,
                            sectorId,
                                slId,
                             layerId,
                              cellId,
                           noiseFlag,
                              feMask,
                             tdcMask,
                            trigMask,
                            deadFlag,
                            nohvFlag );
  setCellStatus(   wheelId,
                 stationId,
                  sectorId,
                      slId,
                   layerId,
                    cellId,
                      flag,
                    feMask,
                   tdcMask,
                  trigMask,
                  deadFlag,
                  nohvFlag );
  return status;

}


int DTStatusFlag::setCellNoise( const DTWireId& id,
                                bool flag ) {
  return setCellNoise( id.wheel(),
                       id.station(),
                       id.sector(),
                       id.superLayer(),
                       id.layer(),
                       id.wire(),
                       flag );
}


int DTStatusFlag::setCellFEMask( int   wheelId,
                                 int stationId,
                                 int  sectorId,
                                 int      slId,
                                 int   layerId,
                                 int    cellId,
                                 bool mask ) {

  bool noiseFlag;
  bool    feMask;
  bool   tdcMask;
  bool  trigMask;
  bool  deadFlag;
  bool  nohvFlag;
  int status = cellStatus(   wheelId,
                           stationId,
                            sectorId,
                                slId,
                             layerId,
                              cellId,
                           noiseFlag,
                              feMask,
                             tdcMask,
                            trigMask,
                            deadFlag,
                            nohvFlag );
  setCellStatus(   wheelId,
                 stationId,
                  sectorId,
                      slId,
                   layerId,
                    cellId,
                 noiseFlag,
                      mask,
                   tdcMask,
                  trigMask,
                  deadFlag,
                  nohvFlag );
  return status;

}


int DTStatusFlag::setCellFEMask( const DTWireId& id,
                                 bool mask ) {
  return setCellFEMask( id.wheel(),
                        id.station(),
                        id.sector(),
                        id.superLayer(),
                        id.layer(),
                        id.wire(),
                        mask );
}


int DTStatusFlag::setCellTDCMask( int   wheelId,
                                  int stationId,
                                  int  sectorId,
                                  int      slId,
                                  int   layerId,
                                  int    cellId,
                                  bool mask ) {

  bool noiseFlag;
  bool    feMask;
  bool   tdcMask;
  bool  trigMask;
  bool  deadFlag;
  bool  nohvFlag;
  int status = cellStatus(   wheelId,
                           stationId,
                            sectorId,
                                slId,
                             layerId,
                              cellId,
                           noiseFlag,
                              feMask,
                             tdcMask,
                            trigMask,
                            deadFlag,
                            nohvFlag );
  setCellStatus(   wheelId,
                 stationId,
                  sectorId,
                      slId,
                   layerId,
                    cellId,
                 noiseFlag,
                    feMask,
                      mask,
                  trigMask,
                  deadFlag,
                  nohvFlag );
  return status;

}


int DTStatusFlag::setCellTDCMask( const DTWireId& id,
                                  bool mask ) {
  return setCellTDCMask( id.wheel(),
                         id.station(),
                         id.sector(),
                         id.superLayer(),
                         id.layer(),
                         id.wire(),
                         mask );
}


int DTStatusFlag::setCellTrigMask( int   wheelId,
                                   int stationId,
                                   int  sectorId,
                                   int      slId,
                                   int   layerId,
                                   int    cellId,
                                   bool mask ) {

  bool noiseFlag;
  bool    feMask;
  bool   tdcMask;
  bool  trigMask;
  bool  deadFlag;
  bool  nohvFlag;
  int status = cellStatus(   wheelId,
                           stationId,
                            sectorId,
                                slId,
                             layerId,
                              cellId,
                           noiseFlag,
                              feMask,
                             tdcMask,
                            trigMask,
                            deadFlag,
                            nohvFlag );
  setCellStatus(   wheelId,
                 stationId,
                  sectorId,
                      slId,
                   layerId,
                    cellId,
                 noiseFlag,
                    feMask,
                   tdcMask,
                      mask,
                  deadFlag,
                  nohvFlag );
  return status;

}


int DTStatusFlag::setCellTrigMask( const DTWireId& id,
                                   bool mask ) {
  return setCellTrigMask( id.wheel(),
                          id.station(),
                          id.sector(),
                          id.superLayer(),
                          id.layer(),
                          id.wire(),
                          mask );
}


int DTStatusFlag::setCellDead( int   wheelId,
                               int stationId,
                               int  sectorId,
                               int      slId,
                               int   layerId,
                               int    cellId,
                               bool flag ) {

  bool noiseFlag;
  bool    feMask;
  bool   tdcMask;
  bool  trigMask;
  bool  deadFlag;
  bool  nohvFlag;
  int status = cellStatus(   wheelId,
                           stationId,
                            sectorId,
                                slId,
                             layerId,
                              cellId,
                           noiseFlag,
                              feMask,
                             tdcMask,
                            trigMask,
                            deadFlag,
                            nohvFlag );
  setCellStatus(   wheelId,
                 stationId,
                  sectorId,
                      slId,
                   layerId,
                    cellId,
                 noiseFlag,
                    feMask,
                   tdcMask,
                  trigMask,
                      flag,
                  nohvFlag );
  return status;

}


int DTStatusFlag::setCellDead( const DTWireId& id,
                               bool flag ) {
  return setCellDead( id.wheel(),
                      id.station(),
                      id.sector(),
                      id.superLayer(),
                      id.layer(),
                      id.wire(),
                      flag );
}


int DTStatusFlag::setCellNoHV( int   wheelId,
                               int stationId,
                               int  sectorId,
                               int      slId,
                               int   layerId,
                               int    cellId,
                               bool flag ) {

  bool noiseFlag;
  bool    feMask;
  bool   tdcMask;
  bool  trigMask;
  bool  deadFlag;
  bool  nohvFlag;
  int status = cellStatus(   wheelId,
                           stationId,
                            sectorId,
                                slId,
                             layerId,
                              cellId,
                           noiseFlag,
                              feMask,
                             tdcMask,
                            trigMask,
                            deadFlag,
                            nohvFlag );
  setCellStatus(   wheelId,
                 stationId,
                  sectorId,
                      slId,
                   layerId,
                    cellId,
                 noiseFlag,
                    feMask,
                   tdcMask,
                  trigMask,
                  deadFlag,
                      flag );
  return status;

}


int DTStatusFlag::setCellNoHV( const DTWireId& id,
                               bool flag ) {
  return setCellNoHV( id.wheel(),
                      id.station(),
                      id.sector(),
                      id.superLayer(),
                      id.layer(),
                      id.wire(),
                      flag );
}


DTStatusFlag::const_iterator DTStatusFlag::begin() const {
  return cellData.begin();
}


DTStatusFlag::const_iterator DTStatusFlag::end() const {
  return cellData.end();
}


void DTStatusFlag::initSetup() const {

  std::string statusVersionN = dataVersion + "_StatusN";
  std::string statusVersionF = dataVersion + "_StatusF";
  std::string statusVersionT = dataVersion + "_StatusT";
  std::string statusVersionD = dataVersion + "_StatusD";
  std::string statusVersionH = dataVersion + "_StatusH";

  DTBufferTree<int,bool>* dataNBuf =
  DTDataBuffer<int,bool>::openBuffer( statusVersionN );
  DTBufferTree<int,bool>* dataFBuf =
  DTDataBuffer<int,bool>::openBuffer( statusVersionF );
  DTBufferTree<int,bool>* dataTBuf =
  DTDataBuffer<int,bool>::openBuffer( statusVersionT );
  DTBufferTree<int,bool>* dataDBuf =
  DTDataBuffer<int,bool>::openBuffer( statusVersionD );
  DTBufferTree<int,bool>* dataHBuf =
  DTDataBuffer<int,bool>::openBuffer( statusVersionH );

  std::vector<DTCellStatusFlagData>::const_iterator iter = cellData.begin();
  std::vector<DTCellStatusFlagData>::const_iterator iend = cellData.end();
  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int   layerId;
  int    cellId;
  bool noiseFlag;
  bool    feMask;
  bool   tdcMask;
  bool  deadFlag;
  bool  nohvFlag;
  while ( iter != iend ) {

    const DTCellStatusFlagData& data = *iter++;
      wheelId = data.  wheelId;
    stationId = data.stationId;
     sectorId = data. sectorId;
         slId = data.     slId;
      layerId = data.  layerId;
       cellId = data.   cellId;

    std::vector<int> cellKey;
    cellKey.push_back(   wheelId );
    cellKey.push_back( stationId );
    cellKey.push_back(  sectorId );
    cellKey.push_back(      slId );
    cellKey.push_back(   layerId );
    cellKey.push_back(    cellId );

    noiseFlag = data.noiseFlag;
    dataNBuf->insert( cellKey.begin(), cellKey.end(), noiseFlag );
       feMask = data.   feMask;
    dataFBuf->insert( cellKey.begin(), cellKey.end(),    feMask );
      tdcMask = data.  tdcMask;
    dataTBuf->insert( cellKey.begin(), cellKey.end(),   tdcMask );
     deadFlag = data. deadFlag;
    dataDBuf->insert( cellKey.begin(), cellKey.end(),  deadFlag );
     nohvFlag = data. nohvFlag;
    dataHBuf->insert( cellKey.begin(), cellKey.end(),  nohvFlag );

  }

  return;

}

