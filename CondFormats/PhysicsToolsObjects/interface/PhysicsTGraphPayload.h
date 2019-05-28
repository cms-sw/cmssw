#ifndef CondFormats_PhysicsToolsObjects_PhysicsTGraphPayload_h
#define CondFormats_PhysicsToolsObjects_PhysicsTGraphPayload_h

/*
 * PhysicsTGraphPayload
 *
 * Class to persist TGraph objects in Conditions database
 * (The TGraphs are used to evaluate Pt dependent cuts on the output of tau ID MVAs)
 *
 * Author: Christian Veelken, LLR
 *
 */

#include "CondFormats/Serialization/interface/Serializable.h"

#include "TGraph.h"

#include <vector>
#include <string>
#include <iostream>

class PhysicsTGraphPayload {
public:
  /// default constructor
  PhysicsTGraphPayload();

  /// constructor from TGraph object
  PhysicsTGraphPayload(const TGraph& graph);

  /// conversion to TGraph
  operator TGraph() const;

  /// print points of TGraph object
  void print(std::ostream& stream) const;

protected:
  std::string name_;
  int numPoints_;
  std::vector<float> x_;
  std::vector<float> y_;

  COND_SERIALIZABLE;
};

#endif
