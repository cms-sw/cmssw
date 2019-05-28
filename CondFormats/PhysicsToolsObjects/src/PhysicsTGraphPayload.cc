#include "CondFormats/PhysicsToolsObjects/interface/PhysicsTGraphPayload.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <iomanip>

PhysicsTGraphPayload::PhysicsTGraphPayload() : numPoints_(0) {}

PhysicsTGraphPayload::PhysicsTGraphPayload(const TGraph& graph) : numPoints_(0) {
  if (graph.GetN() >= 1) {
    name_ = graph.GetName();
    numPoints_ = graph.GetN();
    x_.resize(numPoints_);
    y_.resize(numPoints_);
    for (int iPoint = 0; iPoint < numPoints_; ++iPoint) {
      Double_t xPoint, yPoint;
      graph.GetPoint(iPoint, xPoint, yPoint);
      x_[iPoint] = xPoint;
      y_[iPoint] = yPoint;
    }
  }
}

PhysicsTGraphPayload::operator TGraph() const {
  if (numPoints_ >= 1) {
    TGraph graph(numPoints_);
    graph.SetName(name_.data());
    for (int iPoint = 0; iPoint < numPoints_; ++iPoint) {
      graph.SetPoint(iPoint, x_[iPoint], y_[iPoint]);
    }
    return graph;
  } else {
    throw cms::Exception("PhysicsTGraphPayload") << "Invalid TGraph object !!\n";
  }
}

void PhysicsTGraphPayload::print(std::ostream& stream) const {
  stream << "<PhysicsTGraphPayload::print (name = " << name_ << ")>:" << std::endl;
  for (int iPoint = 0; iPoint < numPoints_; ++iPoint) {
    stream << "point #" << iPoint << ": x = " << x_[iPoint] << ", y = " << y_[iPoint] << std::endl;
  }
}
