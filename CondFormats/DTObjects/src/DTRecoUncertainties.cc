
/*
 *  See header file for a description of this class.
 *
 *  $Date: $
 *  $Revision: $
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


float DTRecoUncertainties::get(const DTWireId& wireid, DTRecoUncertainties::Type type) const {
  // FIXME: what to do in case the superlayerId is not found in the map?
  // FIXME: any check on the type?
  map<uint32_t, vector<float> >::const_iterator slIt = payload.find(wireid.superlayerId().rawId());
  if(slIt == payload.end()) {
    cout << "[DTRecoUncertainties]***Error: the SLId: " << wireid.superlayerId() << " is not in the paylaod map!" << endl;
    // FIXME: what to do here???
    return -1.;
  } else if(vector<float>::size_type(type) >= (*slIt).second.size()) {
    cout << "[DTRecoUncertainties]***Error: requesting parameter index: " << type << " for vector of size " << (*slIt).second.size() << endl;
    // FIXME: what to do here???
    return -1.;
  }
  

  return (*slIt).second[type];
}
  
void DTRecoUncertainties::set(const DTWireId& wireid, DTRecoUncertainties::Type type, float value) {
  map<uint32_t, vector<float> >::iterator slIt = payload.find(wireid.superlayerId().rawId());
  if(slIt == payload.end()) {
    // in this case the vector of values needs to be initialized
    // FIXME: the max numbr of parameters should be coded in the algorithm somehow and not harcoded here!
    vector<float> slPayload(4, 0.);
    slPayload[type] = value;
    payload[wireid.superlayerId().rawId()] = slPayload;
    
  } else {
    (*slIt).second[type] = value;
  }
}


