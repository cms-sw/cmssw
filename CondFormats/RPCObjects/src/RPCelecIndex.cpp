#include "CondFormats/RPCObjects/interface/RPCelecIndex.h"

RPCelecIndex::RPCelecIndex() : dat(0), trb(0), lbb(0), mbo(0),
			       lbo(0), cha(0)
{
}


RPCelecIndex::RPCelecIndex(int dccId, int tbId, int lboxId, 
			   int mbId, int lboardId, int channelId) :
  dat(dccId), trb(tbId), lbb(lboxId), mbo(mbId),
  lbo(lboardId), cha(channelId)
{
}


int 
RPCelecIndex::dcc() const
{
  return dat;
}


int 
RPCelecIndex::tb() const
{
  return trb;
}



int 
RPCelecIndex::lbox() const
{
  return lbb;
}


int 
RPCelecIndex::mb() const
{
  return mbo;
}


int 
RPCelecIndex::lboard() const
{
  return lbo;
}

int 
RPCelecIndex::channel() const
{
  return cha;
}


bool 
RPCelecIndex::operator <(const RPCelecIndex& einx) const
{
  if      (this->dcc()     != einx.dcc())
    return this->dcc()     <  einx.dcc();
  else if (this->tb()      != einx.tb())
    return this->tb()      <  einx.tb();
  else if (this->lbox()    != einx.lbox())
    return this->lbox()    <  einx.lbox();
  else if (this->mb()      != einx.mb())
    return this->mb()      <  einx.mb();
  else if (this->lboard()  != einx.lboard())
    return this->lboard()  <  einx.lboard();
  else if (this->channel() != einx.channel())
    return this->channel() <  einx.channel();
  else
    return false;
 
}
