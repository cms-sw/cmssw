#ifndef CALIBTRACKER_SISTRIPCONNECTIVITY_MODULECOMPOSITEHEADER_H
#define CALIBTRACKER_SISTRIPCONNECTIVITY_MODULECOMPOSITEHEADER_H

#include "CalibTracker/SiStripConnectivity/interface/CompositeHeader.h"
#include <vector>
using namespace std;

/**
 * completely empty, used to add header infos
 */

class ModuleCompositeHeader : public CompositeHeader {
 public:
  ModuleCompositeHeader(){}
  ~ModuleCompositeHeader(){}
  
  void setPosition(float x, float y, float z) {x_position = x; y_position = y; z_position = z;}
  vector<float> getPosition() {vector<float> temp; temp.push_back(x_position); temp.push_back(y_position); temp.push_back(z_position); return temp;}
  
 private:


  float    x_position;
  float    y_position;
  float    z_position;

};

#endif
