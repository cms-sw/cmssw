#include "L1Trigger/RPCTechnicalTrigger/src/RBCLinkSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCId.h"

RBCLinkSignal::RBCLinkSignal(const RBCId& id, int layer, int bx) :
  rid(id),l(layer),x(bx)
{
}

RBCLinkSignal::~RBCLinkSignal()
{
}


const RBCId& 
RBCLinkSignal::rbcid() const
{
  return rid;
}

int 
RBCLinkSignal::triggerLayer() const
{
  return l;
}

int
RBCLinkSignal::bx() const
{
  return x;
}


bool
RBCLinkSignal::operator < (const RBCLinkSignal& link) const
{
  if ( this->rbcid().wheel() == link.rbcid().wheel() ){
    if ( this->rbcid().sector() == link.rbcid().sector() ) {
      if ( this->triggerLayer() == link.triggerLayer() ) {
	return ( this->bx() < link.bx() );
      } else {
	return ( this->triggerLayer() < link.triggerLayer() );
      }
    } else {
      return ( this->rbcid().sector() < link.rbcid().sector() );
    }
  } else {
    return ( this->rbcid().wheel() < link.rbcid().wheel() );
  }
}
