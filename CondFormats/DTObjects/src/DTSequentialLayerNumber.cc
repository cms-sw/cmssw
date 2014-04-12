/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/04/30 16:20:08 $
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

namespace {
  constexpr int offsetChamber[] = { 0, 0, 12, 24, 36 };
  constexpr int layersPerSector = 44;
  constexpr int layersIn13Sectors = ( layersPerSector * 12 ) + 8;
  constexpr int layersPerWheel = layersIn13Sectors + 8;
}

DTSequentialLayerNumber DTSequentialLayer;


//----------------
// Constructors --
//----------------
DTSequentialLayerNumber::DTSequentialLayerNumber() {
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


