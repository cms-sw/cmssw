#ifndef PhysicsTools_MVATrainer_MVATrainerSaveImpl_h
#define PhysicsTools_MVATrainer_MVATrainerSaveImpl_h

#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

#include "PhysicsTools/MVATrainer/interface/MVATrainerSave.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerContainerSave.h"

namespace PhysicsTools {

  template <typename Record_t>
  class MVATrainerSaveImpl : public MVATrainerSave {
  public:
    explicit MVATrainerSaveImpl(const edm::ParameterSet& params) : MVATrainerSave(params) {}

  protected:
    const Calibration::MVAComputer* getToPut(const edm::EventSetup& es) const override {
      edm::ESHandle<Calibration::MVAComputer> handle;
      es.get<Record_t>().get("trained", handle);
      return handle.product();
    }

    std::string getRecordName() const override { return Record_t::keyForClass().type().name(); }
  };

  template <typename Record_t>
  class MVATrainerContainerSaveImpl : public MVATrainerContainerSave {
  public:
    explicit MVATrainerContainerSaveImpl(const edm::ParameterSet& params) : MVATrainerContainerSave(params) {}

  protected:
    const Calibration::MVAComputerContainer* getToPut(const edm::EventSetup& es) const override {
      edm::ESHandle<Calibration::MVAComputerContainer> handle;
      es.get<Record_t>().get("trained", handle);
      return handle.product();
    }

    const Calibration::MVAComputerContainer* getToCopy(const edm::EventSetup& es) const override {
      edm::ESHandle<Calibration::MVAComputerContainer> handle;
      es.get<Record_t>().get(handle);
      return handle.product();
    }

    std::string getRecordName() const override { return Record_t::keyForClass().type().name(); }
  };

}  // namespace PhysicsTools

#endif  // PhysicsTools_MVATrainer_MVATrainerSaveImpl_h
