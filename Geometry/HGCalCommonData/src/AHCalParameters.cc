#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/AHCalParameters.h"

AHCalParameters::AHCalParameters(edm::ParameterSet const& iC)
    : maxDepth_(iC.getUntrackedParameter<int>("maxDepth", 12)),
      deltaX_(iC.getUntrackedParameter<double>("deltaX", 30.0)),
      deltaY_(iC.getUntrackedParameter<double>("deltaY", 30.0)),
      deltaZ_(iC.getUntrackedParameter<double>("deltaZ", 81.0)),
      zFirst_(iC.getUntrackedParameter<double>("zFirst", 17.6)) {
  edm::LogVerbatim("HGCalGeom") << "AHCalParameters: maxDepth = " << maxDepth_ << " deltaX = " << deltaX_
                                << " deltaY = " << deltaY_ << " deltaZ = " << deltaZ_ << " zFirst = " << zFirst_;
}
