#ifndef _CALO_MISCALIB_MAP_H
#define _CALO_MISCALIB_MAP_H

#include <iostream>
#include "DataFormats/DetId/interface/DetId.h"
#include <map>

class CaloMiscalibMap{

public:
CaloMiscalibMap(){}
virtual ~CaloMiscalibMap(){}

public:


virtual void addCell(const DetId &cell, float scaling_factor)=0;

};


#endif
