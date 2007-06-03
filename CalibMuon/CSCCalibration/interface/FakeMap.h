#ifndef _FAKE_MAP_H
#define _FAKE_MAP_H

#include <iostream>
#include "DataFormats/DetId/interface/DetId.h"
#include <map>

class FakeMap{

 public:
  FakeMap(){}
  virtual ~FakeMap(){}

 public:
  virtual void csc(const DetId &cell, float scaling_factor)=0;
  
};


#endif
