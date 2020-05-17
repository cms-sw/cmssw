#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <map>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Entry.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerESSourceBase.h"

namespace PhysicsTools {

  MVAComputerESSourceBase::MVAComputerESSourceBase(const edm::ParameterSet &params) {
    std::vector<std::string> names = params.getParameterNames();
    for (const auto &name : names) {
      if (name.c_str()[0] == '@')
        continue;

      const edm::Entry &entry = params.retrieve(name);

      std::string path;
      if (entry.typeCode() == 'F')
        path = entry.getFileInPath().fullPath();
      else
        path = entry.getString();

      mvaCalibrations[name] = path;
    }
  }

  MVAComputerESSourceBase::~MVAComputerESSourceBase() {}

  MVAComputerESSourceBase::ReturnType MVAComputerESSourceBase::produce() const {
    auto container = std::make_unique<Calibration::MVAComputerContainer>();

    for (const auto &mvaCalibration : mvaCalibrations) {
      std::unique_ptr<Calibration::MVAComputer> calibration(
          MVAComputer::readCalibration(mvaCalibration.second.c_str()));

      container->add(mvaCalibration.first) = *calibration;
    }

    return container;
  }

}  // namespace PhysicsTools
