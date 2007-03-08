#include "L1Trigger/RPCTechnicalTrigger/interface/RBCOutputSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCId.h"


RBCOutputSignal::RBCOutputSignal(const RBCId& rid, int obx) : id(rid), x(obx)
{}


RBCOutputSignal::~RBCOutputSignal(){}


const RBCId&
RBCOutputSignal::rbcid() const
{
  return id;
}


int
RBCOutputSignal::bx() const
{
  return x;
}

bool
RBCOutputSignal::operator < (const RBCOutputSignal& o) const
{
  if ( this->rbcid().wheel() ==  o.rbcid().wheel() ){
    if ( this->rbcid().sector() ==  o.rbcid().sector() )
      return (  this-> bx() < o.bx());
    else
      return (this->rbcid().sector() < o.rbcid().sector());

  } else 
    return (this->rbcid().wheel() < o.rbcid().wheel());
}
