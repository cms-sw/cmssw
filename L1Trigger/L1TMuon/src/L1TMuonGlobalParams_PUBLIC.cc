#include "L1Trigger/L1TMuon/interface/L1TMuonGlobalParams_PUBLIC.h"

const L1TMuonGlobalParams_PUBLIC & cast_to_L1TMuonGlobalParams_PUBLIC(const L1TMuonGlobalParams & x){
  assert(sizeof(L1TMuonGlobalParams_PUBLIC) == sizeof(L1TMuonGlobalParams));
  const void * px = &x;
  const L1TMuonGlobalParams_PUBLIC * py = static_cast<const L1TMuonGlobalParams_PUBLIC *>(px);
  return *py;
}

const L1TMuonGlobalParams & cast_to_L1TMuonGlobalParams(const L1TMuonGlobalParams_PUBLIC & x){
  assert(sizeof(L1TMuonGlobalParams_PUBLIC) == sizeof(L1TMuonGlobalParams));
  const void * px = &x;
  const L1TMuonGlobalParams * py = static_cast<const L1TMuonGlobalParams *>(px);
  return *py;
}


