#ifndef DCCEEEVENTBLOCK_HH
#define DCCEEEVENTBLOCK_HH


/*
 *\ Class DCCEEventBlock
 *
 * Specialization of the DCCEventBlock class for the EE 
 * The class instantes the DCCSCBlock, DCCEETCCBlock and DCCEESRPBlock unpacking classes
 *
 * \file DCCEEEventBlock.h
 *
 * $Date: 2008/12/11 18:05:56 $
 * $Revision: 1.1 $
 *
 * \author N. Almeida
 *
*/

#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include "DCCRawDataDefinitions.h"
#include "DCCEventBlock.h"


class DCCEEEventBlock : public DCCEventBlock{
	
  public :

   DCCEEEventBlock( DCCDataUnpacker * u, EcalElectronicsMapper * m, bool hU, bool srpU, bool tccU, bool feU, bool memU, bool forceToKeepFRdata );
   
   void unpack( uint64_t * buffer, uint bufferSize, uint expFedId);
	
  protected :
  
   int unpackTCCBlocks();
   
   
};

#endif
