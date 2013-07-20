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
#include "CondFormats/DTObjects/interface/DTSequentialLayerNumber.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------


//---------------
// C++ Headers --
//---------------


//-------------------
// Initializations --
//-------------------
int DTSequentialLayerNumber::layersPerWheel    = 0;
int DTSequentialLayerNumber::layersPerSector   = 0;
int DTSequentialLayerNumber::layersIn13Sectors = 0;

int* DTSequentialLayerNumber::offsetChamber = 0;

DTSequentialLayerNumber DTSequentialLayer;


//----------------
// Constructors --
//----------------
DTSequentialLayerNumber::DTSequentialLayerNumber() {
  if ( offsetChamber == 0 ) {
    offsetChamber = new int[5];
    offsetChamber[1] = 0;
    offsetChamber[2] = 12;
    offsetChamber[3] = 24;
    offsetChamber[4] = 36;
    layersPerSector   = 44;
    layersIn13Sectors = ( layersPerSector * 12 ) + 8;
    layersPerWheel = layersIn13Sectors + 8;
  }
}


//--------------
// Destructor --
//--------------
DTSequentialLayerNumber::~DTSequentialLayerNumber() {
}


int DTSequentialLayerNumber::id( int      wheel,
                                 int    station,
                                 int     sector,
                                 int superlayer,
                                 int      layer ) {

  wheel += 3;
  if ( wheel      <= 0 ) return -1;
  if ( station    <= 0 ) return -2;
  if ( sector     <= 0 ) return -3;
  if ( superlayer <= 0 ) return -4;
  if ( layer      <= 0 ) return -5;

  int seqLayerNum = 0;

  if ( wheel      >  5 ) return -1;
  seqLayerNum += ( wheel - 1 ) * layersPerWheel;

  if ( sector     > 14 ) return -2;
  if ( sector     > 12   &&
       station    <  4 ) return -2;
  if ( sector     > 13 ) seqLayerNum += layersIn13Sectors;
  else
  seqLayerNum += ( sector - 1 ) * layersPerSector;

  if ( station    >  4 ) return -3;
  if ( sector < 13 )
  seqLayerNum += offsetChamber[station];

  if ( superlayer >  3 ) return -4;
  if ( layer      >  4 ) return -5;
  if ( superlayer != 2 ) {
    if ( superlayer == 3 ) layer += 4;
  }
  else {
    if ( station  == 4 ) return -4;
    layer += 8;
  }

  return seqLayerNum + layer;

}


int DTSequentialLayerNumber::max() {
  return 5 * layersPerWheel;
}


