/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/05/06 14:41:14 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondFormats/DTObjects/interface/DTSequentialCellNumber.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------


//---------------
// C++ Headers --
//---------------


//-------------------
// Initializations --
//-------------------
int DTSequentialCellNumber::cellsPerWheel    = 0;
int DTSequentialCellNumber::cellsPerSector   = 0;
int DTSequentialCellNumber::cellsIn13Sectors = 0;
int DTSequentialCellNumber::cellsInTheta     = 58;
int DTSequentialCellNumber::cellsInMB1       = 0;
int DTSequentialCellNumber::cellsInMB2       = 0;
int DTSequentialCellNumber::cellsInMB3       = 0;
int DTSequentialCellNumber::cellsInMB4       = 0;

int* DTSequentialCellNumber::offsetChamber = 0;
int* DTSequentialCellNumber::cellsPerLayer = 0;

DTSequentialCellNumber DTSequentialCell;


//----------------
// Constructors --
//----------------
DTSequentialCellNumber::DTSequentialCellNumber() {
  if ( cellsPerLayer == 0 ) {
    cellsPerLayer = new int[5];
    cellsPerLayer[1] = 50;
    cellsPerLayer[2] = 60;
    cellsPerLayer[3] = 72;
    cellsPerLayer[4] = 96;
  }
  if ( offsetChamber == 0 ) {
    offsetChamber = new int[5];
    cellsInMB1 = ( cellsPerLayer[1] * 8 ) + ( cellsInTheta * 4 );
    cellsInMB2 = ( cellsPerLayer[2] * 8 ) + ( cellsInTheta * 4 );
    cellsInMB3 = ( cellsPerLayer[3] * 8 ) + ( cellsInTheta * 4 );
    cellsInMB4 =   cellsPerLayer[4] * 8;
    offsetChamber[1] = 0;
    offsetChamber[2] = cellsInMB1;
    offsetChamber[3] = cellsInMB1 + cellsInMB2;
    offsetChamber[4] = cellsInMB1 + cellsInMB2 + cellsInMB3;
    cellsPerSector   = cellsInMB1 + cellsInMB2 + cellsInMB3 + cellsInMB4;
    cellsIn13Sectors = ( cellsPerSector * 12 ) + cellsInMB4;
    cellsPerWheel = cellsIn13Sectors + cellsInMB4;
  }
}


//--------------
// Destructor --
//--------------
DTSequentialCellNumber::~DTSequentialCellNumber() {
}


//--------------
// Operations --
//--------------
int DTSequentialCellNumber::id( int      wheel,
                                int    station,
                                int     sector,
                                int superlayer,
                                int      layer,
                                int       cell ) {

  wheel += 3;
  if ( wheel      <= 0 ) return -1;
  if ( station    <= 0 ) return -2;
  if ( sector     <= 0 ) return -3;
  if ( superlayer <= 0 ) return -4;
  if ( layer      <= 0 ) return -5;
  if ( cell       <= 0 ) return -6;

  int seqWireNum = 0;

  if ( wheel      >  5 ) return -1;
  seqWireNum += ( wheel - 1 ) * cellsPerWheel;

  if ( sector     > 14 ) return -2;
  if ( sector     > 12   &&
       station    <  4 ) return -2;
  if ( sector     > 13 ) seqWireNum += cellsIn13Sectors;
  else
  seqWireNum += ( sector - 1 ) * cellsPerSector;

  if ( station    >  4 ) return -3;
  if ( sector < 13 )
  seqWireNum += offsetChamber[station];

  if ( superlayer >  3 ) return -4;
  if ( layer      >  4 ) return -5;
  if ( superlayer != 2 ) {
    if ( cell     > cellsPerLayer[station] ) return -6;
    if ( superlayer == 3 ) layer += 4;
    seqWireNum += ( layer - 1 ) * cellsPerLayer[station];
  }
  else {
    if ( station  == 4 ) return -4;
    if ( cell     > cellsInTheta           ) return -6;
    seqWireNum += ( 8 * cellsPerLayer[station] ) +
                ( ( layer - 1 ) * cellsInTheta );
  }

  return seqWireNum + cell;

}


int DTSequentialCellNumber::max() {
  return 5 * cellsPerWheel;
}


