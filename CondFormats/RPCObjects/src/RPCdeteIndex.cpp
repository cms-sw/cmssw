#include "CondFormats/RPCObjects/interface/RPCdeteIndex.h"

RPCdeteIndex::RPCdeteIndex() : reg(0), dis(0), sta(0), sec(0),
			       lay(0), sub(0), rol(0), str(0)
{
}


RPCdeteIndex::RPCdeteIndex(int regionId, int diskId, int stationId, 
			   int sectorId, int layerId, int subsectorId, 
			   int rollId, int stripId) : 
  reg(regionId), dis(diskId), sta(stationId), sec(sectorId),
  lay(layerId), sub(subsectorId), rol(rollId), str(stripId)
{
}


int 
RPCdeteIndex::region() const
{
  return reg;
}


int 
RPCdeteIndex::disk() const
{
  return dis;
}



int 
RPCdeteIndex::station() const
{
  return sta;
}


int 
RPCdeteIndex::sector()const
{
  return sec;
}


int 
RPCdeteIndex::layer() const
{
  return lay;
}

int 
RPCdeteIndex::subsector() const
{
  return sub;
}


int 
RPCdeteIndex::roll() const
{
  return rol;
}


int 
RPCdeteIndex::strip() const
{
  return str;
}



bool 
RPCdeteIndex::operator <(const RPCdeteIndex& dinx) const
{
  if      (this->region()    != dinx.region())
    return this->region()    <  dinx.region();
  else if (this->disk()      != dinx.disk())
    return this->disk()      <  dinx.disk();
  else if (this->station()   != dinx.station())
    return this->station()   <  dinx.station();
  else if (this->sector()    != dinx.sector())
    return this->sector()    <  dinx.sector();
  else if (this->layer()     != dinx.layer())
    return this->layer()     <  dinx.layer();
  else if (this->subsector() != dinx.subsector())
    return this->subsector() <  dinx.subsector();
  else if (this->roll()      != dinx.roll())
    return this->roll()      <  dinx.roll();
  else if (this->strip()     != dinx.strip())
    return this->strip()     <  dinx.strip();
  else
    return false;
 }

