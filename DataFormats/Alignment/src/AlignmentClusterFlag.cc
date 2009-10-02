#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h"


AlignmentClusterFlag::AlignmentClusterFlag(){
  detId_=0;
  hitflag_=0;
}

AlignmentClusterFlag::AlignmentClusterFlag(const DetId & id){
  detId_ = id;
  hitflag_ = 0;
}

AlignmentClusterFlag::AlignmentClusterFlag(const AlignmentClusterFlag & acf){detId_ = acf.detId_;hitflag_ = acf.hitflag_;}

AlignmentClusterFlag::~AlignmentClusterFlag(){
  //
}

bool AlignmentClusterFlag::isTaken() const{
  return ( (hitflag_ & (1<<0)) != 0);
}


bool AlignmentClusterFlag::isOverlap() const{
  return ( (hitflag_ & (1<<1)) != 0);
}

void AlignmentClusterFlag::SetTakenFlag(){
  hitflag_ |= (1<<0);
}

void AlignmentClusterFlag::SetOverlapFlag(){
  hitflag_ |= (1<<1);
}

void AlignmentClusterFlag::SetDetId(const DetId & newdetid){
  detId_=newdetid;
}
