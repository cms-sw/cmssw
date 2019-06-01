#include "EventFilter/Utilities/interface/MicroStateServiceClassic.h"

namespace evf {

  MicroStateServiceClassic::MicroStateServiceClassic(const edm::ParameterSet& iPS, edm::ActivityRegistry& reg)
      : MicroStateService(iPS, reg),
        microstate2_(&(reservedMicroStateNames[MicroStateService::mInvalid].moduleLabel())) {
    reg.watchPostBeginJob(this, &MicroStateServiceClassic::postBeginJob);
    reg.watchPostEndJob(this, &MicroStateServiceClassic::postEndJob);

    reg.watchPreProcessEvent(this, &MicroStateServiceClassic::preEventProcessing);
    reg.watchPostProcessEvent(this, &MicroStateServiceClassic::postEventProcessing);
    reg.watchPreSourceEvent(this, &MicroStateServiceClassic::preSourceEvent);
    reg.watchPostSourceEvent(this, &MicroStateServiceClassic::postSourceEvent);

    reg.watchPreModule(this, &MicroStateServiceClassic::preModule);
    reg.watchPostModule(this, &MicroStateServiceClassic::postModule);
    microstate1_ = "BJ";
  }

  MicroStateServiceClassic::~MicroStateServiceClassic() {}

  void MicroStateServiceClassic::postBeginJob() {
    boost::mutex::scoped_lock sl(lock_);
    microstate1_ = "BJD";
  }

  void MicroStateServiceClassic::postEndJob() {
    boost::mutex::scoped_lock sl(lock_);
    microstate1_ = "EJ";
    microstate2_ = &done;
  }

  void MicroStateServiceClassic::preEventProcessing(const edm::EventID& iID, const edm::Timestamp& iTime) {
    boost::mutex::scoped_lock sl(lock_);
    microstate1_ = "PRO";
  }

  void MicroStateServiceClassic::postEventProcessing(const edm::Event& e, const edm::EventSetup&) {
    boost::mutex::scoped_lock sl(lock_);
    microstate2_ = &input;
  }

  void MicroStateServiceClassic::preSourceEvent(edm::StreamID) {
    boost::mutex::scoped_lock sl(lock_);
    microstate2_ = &input;
  }

  void MicroStateServiceClassic::postSourceEvent(edm::StreamID) {
    boost::mutex::scoped_lock sl(lock_);
    microstate2_ = &fwkovh;
  }

  void MicroStateServiceClassic::preModule(const edm::ModuleDescription& desc) {
    boost::mutex::scoped_lock sl(lock_);
    microstate2_ = &(desc.moduleLabel());
  }

  void MicroStateServiceClassic::postModule(const edm::ModuleDescription& desc) {
    boost::mutex::scoped_lock sl(lock_);
    microstate2_ = &fwkovh;
  }

  std::string MicroStateServiceClassic::getMicroState1() {
    boost::mutex::scoped_lock sl(lock_);
    return microstate1_;
  }

  std::string const& MicroStateServiceClassic::getMicroState2() {
    boost::mutex::scoped_lock sl(lock_);
    return *microstate2_;
  }

  void MicroStateServiceClassic::setMicroState(MicroStateService::Microstate m) {
    boost::mutex::scoped_lock sl(lock_);
    microstate2_ = &(reservedMicroStateNames[m].moduleLabel());
  }

}  //end namespace evf
