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
 * $Date: 2010/10/04 07:44:38 $
 * $Revision: 1.3 $
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
   
   void unpack(const uint64_t * buffer, size_t bufferSize, unsigned int expFedId);
	
  protected :
  
   int unpackTCCBlocks();
   
   
};

#endif
