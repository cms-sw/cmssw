#ifndef DCCEBEVENTBLOCK_HH
#define DCCEBEVENTBLOCK_HH


/*
 *\ Class DCCEBEventBlock
 *
 * Specialization of the DCCEventBlock for the EB 
 * The class instintes the DCCTowerBlock, DCCEBTCBlock and DCCEBSRPBlock unpacking classes
 *
 * \file DCCEBEventBlock.h
 *
 * $Date: 2008/12/11 18:05:56 $
 * $Revision: 1.1 $
 *
 * \author N. Almeida
 * 
 *
*/

#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include "DCCRawDataDefinitions.h"
#include "DCCEventBlock.h"


class DCCEBEventBlock : public DCCEventBlock{
	
  public :

   DCCEBEventBlock( DCCDataUnpacker * u, EcalElectronicsMapper *m ,  bool hU, bool srpU, bool tccU, bool feU, bool memU, bool forceToKeepFRdata);
   
   void unpack( uint64_t * buffer, uint bufferSize, uint expFedId);
   
  protected :
  
    int unpackTCCBlocks();
   
};

#endif
