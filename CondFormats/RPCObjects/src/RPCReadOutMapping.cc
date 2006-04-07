/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/03/16 17:00:20 $
 *  $Revision: 1.1 $
 *  \author Marcello Maggi INFN Bari
 *
 */
#include "CondFormats/RPCObjects/interface/RPCdeteIndex.h"
#include "CondFormats/RPCObjects/interface/RPCelecIndex.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"


RPCReadOutLink::RPCReadOutLink():
  dccId( 0 ),
  tbId( 0 ),
  lboxId( 0 ),
  mbId( 0 ),
  lboardId( 0 ),
  channelId( 0 ),

  regionId( 0 ),
  diskId( 0 ),
  stationId( 0 ),
  sectorId( 0 ),
  layerId( 0 ),
  subsectorId( 0 ),
  rollId( 0 ) ,
  stripId( 0 )
{}


RPCReadOutLink::~RPCReadOutLink() 
{
}



RPCReadOutMapping::RPCReadOutMapping()
{
}


RPCReadOutMapping::~RPCReadOutMapping() 
{
}


void RPCReadOutMapping::initSetup() 
{
  std::vector<RPCReadOutLink>::const_iterator iter =
    readOutRPCMap.begin();
  std::vector<RPCReadOutLink>::const_iterator iend =
    readOutRPCMap.end();

  int      dccId;
  int      tbId;
  int      lboxId;
  int      mbId;
  int      lboardId; 
  int      channelId;

  int      regionId;
  int      diskId;
  int      stationId;
  int      sectorId;
  int      layerId;
  int      subsectorId;
  int      rollId;
  int      stripId;

  while ( iter != iend ) {
    const RPCReadOutLink& link = *iter++;

    dccId       = link.dccId;
    tbId        = link.tbId;
    lboxId      = link.lboxId;
    mbId        = link.mbId;
    lboardId    = link.lboardId; 
    channelId   = link.channelId;

    RPCelecIndex eind(dccId,tbId,lboxId,mbId,lboardId,channelId);

    regionId    = link.regionId;
    diskId      = link.diskId;
    stationId   = link.stationId;
    sectorId    = link.sectorId;
    layerId     = link.layerId;
    subsectorId = link.subsectorId;
    rollId      = link.rollId;
    stripId     = link.stripId;

    RPCdeteIndex dind(regionId,diskId,stationId,sectorId,
		      layerId,subsectorId,rollId,stripId);

    dtoe[dind]=eind;
    etod[eind]=dind;

  }

}


void
RPCReadOutMapping::readOutToGeometry( int       dccId,
				      int        tbId,
				      int      lboxId,
				      int        mbId,
				      int    lboardId,
				      int   channelId,
				      int&   regionId,
				      int&     diskId,
				      int&  stationId,
				      int&   sectorId,
				      int&    layerId,
				      int&subsectorId,
				      int&     rollId,
				      int&    stripId) 
{
  RPCelecIndex eind(dccId,tbId,lboxId,mbId,lboardId,channelId);
  RPCdeteIndex dind=etod[eind];
  regionId    = dind.region();
  diskId      = dind.disk();
  stationId   = dind.station();
  sectorId    = dind.sector();
  layerId     = dind.layer();
  subsectorId = dind.subsector();
  rollId      = dind.roll();
  stripId     = dind.strip();
}



void 
RPCReadOutMapping::clear() 
{
  readOutRPCMap.clear();
  dtoe.clear();
  etod.clear();
}


void
RPCReadOutMapping::insertReadOutGeometryLink( int       dccId,
					      int        tbId,
					      int      lboxId,
					      int        mbId,
					      int    lboardId,
					      int   channelId,
					      int    regionId,
					      int      diskId,
					      int   stationId,
					      int    sectorId,
					      int     layerId,
					      int subsectorId,
					      int      rollId,
					      int     stripId) 
{
  RPCReadOutLink link;
  link.dccId     = dccId;
  link.tbId      = tbId;
  link.lboxId    = lboxId;
  link.mbId      = mbId;
  link.lboardId  = lboardId;
  link.channelId = channelId;

  link.regionId    = regionId;
  link.diskId      = diskId;
  link.stationId   = stationId;
  link.sectorId    = sectorId;
  link.layerId     = layerId;
  link.subsectorId = subsectorId;
  link.rollId      = rollId;
  link.stripId     = stripId;

  readOutRPCMap.push_back( link );

}


RPCReadOutMapping::const_iterator RPCReadOutMapping::begin() const {
  return readOutRPCMap.begin();
}


RPCReadOutMapping::const_iterator RPCReadOutMapping::end() const {
  return readOutRPCMap.end();
}

