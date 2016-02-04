/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/03/26 14:11:04 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondTools/DT/interface/DTHVHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTHVStatus.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
//#include "Geometry/DTGeometry/interface/DTGeometry.h"
//#include "Geometry/Records/interface/MuonGeometryRecord.h"

//---------------
// C++ Headers --
//---------------


//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTHVHandler::DTHVHandler():
  objectPtr ( 0 ) {
}


/*
DTHVHandler::DTHVHandler( const DTHVStatus* dbObject,
                          const DTGeometry* geometry ):
  objectPtr( dbObject ),
  dtGeomPtr( geometry ) {
}
*/
DTHVHandler::DTHVHandler( const DTHVStatus* dbObject ):
  objectPtr( dbObject ) {
}


//--------------
// Destructor --
//--------------
DTHVHandler::~DTHVHandler() {
}


//--------------
// Operations --
//--------------
int DTHVHandler::get( const DTWireId& id,
                      int&         flagA,
                      int&         flagC,
                      int&         flagS ) const {
/*
  if ( objectPtr == 0 ) {
    flagA = flagC = flagS = 0;
    return 999;
  }
  return objectPtr->get( id.wheel(),
                         id.station(),
                         id.sector(),
                         id.superLayer(),
                         id.layer(),
                         findLayerPart( id ),
                         flagA, flagC, flagS );
*/
  flagA = flagC = flagS = 0;
  if ( objectPtr == 0 ) return 999;
  int iCell = id.wire();
  int fCell;
  int lCell;
  int
  fCheck = objectPtr->get( id.wheel(),
                           id.station(),
                           id.sector(),
                           id.superLayer(),
                           id.layer(),
                           0,     fCell, lCell,
                           flagA, flagC, flagS );
  if ( ( fCheck == 0 ) &&
       ( fCell <= iCell ) &&
       ( lCell >= iCell ) ) return 0;
  fCheck = objectPtr->get( id.wheel(),
                           id.station(),
                           id.sector(),
                           id.superLayer(),
                           id.layer(),
                           1,     fCell, lCell,
                           flagA, flagC, flagS );
  if ( ( fCheck == 0 ) &&
       ( fCell <= iCell ) &&
       ( lCell >= iCell ) ) return 0;
  flagA = flagC = flagS = 0;
  return 1;
}


int DTHVHandler::offChannelsNumber() const {
  int offNum = 0;
  DTHVStatus::const_iterator iter = objectPtr->begin();
  DTHVStatus::const_iterator iend = objectPtr->end();
  while ( iter != iend ) {
    const std::pair<DTHVStatusId,DTHVStatusData>& entry = *iter++;
    DTHVStatusId   hvId = entry.first;
    DTHVStatusData data = entry.second;
    if (   data.flagA || data.flagC || data.flagS   )
           offNum += ( 1 + data.lCell - data.fCell );
  }
  return offNum;
}


int DTHVHandler::offChannelsNumber( const DTChamberId& id ) const {
  int offNum = 0;
  DTHVStatus::const_iterator iter = objectPtr->begin();
  DTHVStatus::const_iterator iend = objectPtr->end();
/*
  while ( iter != iend ) {
    const std::pair<DTHVStatusId,DTHVStatusData>& entry = *iter++;
    DTHVStatusId   idCh = entry.first;
    DTHVStatusData data = entry.second;
    DTLayerId idL( idCh.wheelId, idCh.stationId, idCh.sectorId,
                   idCh.slId,    idCh.layerId );
    int fCell;
    int lCell;
    getLayerEdges( idL, idCh.partId, fCell, lCell );
    if ( data.flagA || data.flagC || data.flagS )
         offNum += ( 1 + lCell - fCell );
  }
*/
  while ( iter != iend ) {
    const std::pair<DTHVStatusId,DTHVStatusData>& entry = *iter++;
    DTHVStatusId   hvId = entry.first;
    DTHVStatusData data = entry.second;
    if ( ( hvId.  wheelId == id.  wheel() ) &&
         ( hvId.stationId == id.station() ) &&
         ( hvId. sectorId == id. sector() ) &&
         ( data.flagA || data.flagC || data.flagS ) )
           offNum += ( 1 + data.lCell - data.fCell );
  }
  return offNum;
//  return 0;
}


/*
int DTHVHandler::findLayerPart( const DTWireId& id ) const {
  int fc;
  int lc;
  getLayerEdges( id.layerId(), fc, lc );
  if ( id.wire() < ( lc / 2 ) ) return 1;
  return 2;
}


int DTHVHandler::getLayerEdges( const DTLayerId& id,
                                int& fCell, int& lCell  ) const {
  const DTLayer* layPtr = dtGeomPtr->layer( id );
  const DTTopology& topo = layPtr->specificTopology();
  fCell = topo.firstChannel();
  lCell = topo. lastChannel();
  return 1;
}


int DTHVHandler::getLayerEdges( const DTLayerId& id, int part,
                                int& fCell, int& lCell  ) const {
  int flc;
  int llc;
  getLayerEdges( id, flc, llc );
  int hlc = llc / 2;
  if ( part == 1 ) {
    fCell = flc;
    lCell = hlc;
  }
  else {
    fCell = hlc + 1;
    lCell = llc;
  }
  return 1;
}
*/

