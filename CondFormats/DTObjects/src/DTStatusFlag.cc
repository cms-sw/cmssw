/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/05/17 10:34:24 $
 *  $Revision: 1.1 $
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
    tdcMask( false ) {
}

//--------------
// Destructor --
//--------------
DTStatusFlag::~DTStatusFlag() {
  std::string statusVersionN = dataVersion + "_StatusN";
  std::string statusVersionF = dataVersion + "_StatusF";
  std::string statusVersionT = dataVersion + "_StatusT";
  DTDataBuffer<int,bool>::dropBuffer( statusVersionN );
  DTDataBuffer<int,bool>::dropBuffer( statusVersionF );
  DTDataBuffer<int,bool>::dropBuffer( statusVersionT );
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
                              bool&   tdcMask ) const {

  noiseFlag = false;
     feMask = false;
    tdcMask = false;

  std::string statusVersionN = dataVersion + "_StatusN";
  std::string statusVersionF = dataVersion + "_StatusF";
  std::string statusVersionT = dataVersion + "_StatusT";
  DTBufferTree<int,bool>* dataNBuf =
  DTDataBuffer<int,bool>::findBuffer( statusVersionN );
  DTBufferTree<int,bool>* dataFBuf =
  DTDataBuffer<int,bool>::findBuffer( statusVersionF );
  DTBufferTree<int,bool>* dataTBuf =
  DTDataBuffer<int,bool>::findBuffer( statusVersionT );

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

//  return 1;
  return ( searchStatusN || searchStatusF || searchStatusT );

}


int DTStatusFlag::cellStatus( const DTWireId& id,
                              bool& noiseFlag,
                              bool&    feMask,
                              bool&   tdcMask ) const {
  return cellStatus( id.wheel(),
                     id.station(),
                     id.sector(),
                     id.superLayer(),
                     id.layer(),
                     id.wire(),
                     noiseFlag, feMask, tdcMask );
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
                                 bool   tdcMask ) {

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

  cellData.push_back( data );

  std::string statusVersionN = dataVersion + "_StatusN";
  std::string statusVersionF = dataVersion + "_StatusF";
  std::string statusVersionT = dataVersion + "_StatusT";

  DTBufferTree<int,bool>* dataNBuf =
  DTDataBuffer<int,bool>::openBuffer( statusVersionN );
  DTBufferTree<int,bool>* dataFBuf =
  DTDataBuffer<int,bool>::openBuffer( statusVersionF );
  DTBufferTree<int,bool>* dataTBuf =
  DTDataBuffer<int,bool>::openBuffer( statusVersionT );

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

  return 0;

}


int DTStatusFlag::setCellStatus( const DTWireId& id,
                                 bool noiseFlag,
                                 bool    feMask,
                                 bool   tdcMask ) {
  return setCellStatus( id.wheel(),
                        id.station(),
                        id.sector(),
                        id.superLayer(),
                        id.layer(),
                        id.wire(),
                        noiseFlag, feMask, tdcMask );
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

  DTBufferTree<int,bool>* dataNBuf =
  DTDataBuffer<int,bool>::openBuffer( statusVersionN );
  DTBufferTree<int,bool>* dataFBuf =
  DTDataBuffer<int,bool>::openBuffer( statusVersionF );
  DTBufferTree<int,bool>* dataTBuf =
  DTDataBuffer<int,bool>::openBuffer( statusVersionT );

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

  }

  return;

}

