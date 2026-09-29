#ifndef PhysicsTools_MVAComputer_MVAComputerESSourceImpl_h
#define PhysicsTools_MVAComputer_MVAComputerESSourceImpl_h

#include "FWCore/Framework/interface/EventSetupRecordInfiniteIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerESSourceBase.h"

namespace PhysicsTools {

  template <class RecordType>
  class MVAComputerESSourceImpl : public MVAComputerESSourceBase, public edm::EventSetupRecordInfiniteIntervalFinder {
  public:
    MVAComputerESSourceImpl(const edm::ParameterSet &params) : MVAComputerESSourceBase(params) {
      setWhatProduced(this);
      findingRecord<RecordType>();
    }

    ~MVAComputerESSourceImpl() override {}

    ReturnType produce(const RecordType &record) { return this->produce(); }

  protected:
    using MVAComputerESSourceBase::produce;
  };

}  // namespace PhysicsTools

#endif  // PhysicsTools_MVAComputer_MVAComputerESSourceImpl_h
