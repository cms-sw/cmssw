#ifndef PhysicsTools_MVAComputer_MVAComputerESSourceImpl_h
#define PhysicsTools_MVAComputer_MVAComputerESSourceImpl_h

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerESSourceBase.h"

namespace PhysicsTools {

template<class RecordType>
class MVAComputerESSourceImpl : public MVAComputerESSourceBase,
                                public edm::EventSetupRecordIntervalFinder {
    public:
	MVAComputerESSourceImpl(const edm::ParameterSet &params) :
		MVAComputerESSourceBase(params)
	{
		setWhatProduced(this);
		findingRecord<RecordType>();
	}

	virtual ~MVAComputerESSourceImpl() {}

	ReturnType produce(const RecordType &record)
	{ return this->produce(); }

    protected:
	using MVAComputerESSourceBase::produce;

    private:
	void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
	                    const edm::IOVSyncValue &syncValue,
	                    edm::ValidityInterval &oValidity)
	{
		oValidity = edm::ValidityInterval(
					edm::IOVSyncValue::beginOfTime(),
					edm::IOVSyncValue::endOfTime());
	}
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVAComputer_MVAComputerESSourceImpl_h
