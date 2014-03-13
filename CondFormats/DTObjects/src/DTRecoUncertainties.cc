
/*
 *  See header file for a description of this class.
 *
 *  \author G. Cerminara - CERN
 */

#include "CondFormats/DTObjects/interface/DTRecoUncertainties.h"
#include "DataFormats/MuonDetId/src/DTWireId.cc"
#include <iostream>

using std::map;
using std::vector;
using std::cout;
using std::endl;


DTRecoUncertainties::DTRecoUncertainties(){}

DTRecoUncertainties::~DTRecoUncertainties(){}


float DTRecoUncertainties::get(const DTWireId& wireid, unsigned int index) const {
  // FIXME: what to do in case the superlayerId is not found in the map?
  // FIXME: any check on the type?
  map<uint32_t, vector<float> >::const_iterator slIt = payload.find(wireid.superlayerId().rawId());
  if(slIt == payload.end()) {
    cout << "[DTRecoUncertainties]***Error: the SLId: " << wireid.superlayerId() << " is not in the paylaod map!" << endl;
    // FIXME: what to do here???
    return -1.;
  } else if(vector<float>::size_type(index) >= (*slIt).second.size()) {
    cout << "[DTRecoUncertainties]***Error: requesting parameter index: " << index << " for vector of size " << (*slIt).second.size() << endl;
    // FIXME: what to do here???
    return -1.;
  }
  

  return (*slIt).second[index];
}


void DTRecoUncertainties::set(const DTWireId& wireid, const std::vector<float>& values) {
  payload[wireid.superlayerId()] = values;
}


DTRecoUncertainties::const_iterator DTRecoUncertainties::begin() const {
  return payload.begin();
}

DTRecoUncertainties::const_iterator DTRecoUncertainties::end() const {
  return payload.end();
}
