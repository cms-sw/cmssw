#include "L1Trigger/RBCEmulator/interface/RBCId.h"

RBCId::RBCId() : w(-9),s(0)
{}

RBCId::RBCId(int wheel, int sector) : w(wheel), s(sector)
{}

RBCId::~RBCId(){}


int
RBCId::wheel() const
{
  return w;
}



int
RBCId::sector() const
{
  return s;
}