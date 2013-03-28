#ifndef GeneratorInterface_LHEInterface_LHESource_h
#define GeneratorInterface_LHEInterface_LHESource_h

#include <memory>

#include <boost/shared_ptr.hpp>
#include <boost/ptr_container/ptr_deque.hpp>

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "GeneratorInterface/LHEInterface/plugins/LHEProvenanceHelper.h"
#include "FWCore/Sources/interface/ProducerSourceFromFiles.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"

namespace lhef {
	class LHERunInfo;
	class LHEEvent;
	class LHEReader;
}

namespace edm {
	class EventPrincipal;
	class LuminosityBlockPrincipal;
	class ParameterSet;
	class Run;
	class RunPrincipal;
}

class LHERunInfoProduct;

 class LHESource : public edm::ProducerSourceFromFiles {
    public:
	explicit LHESource(const edm::ParameterSet &params,
	                   const edm::InputSourceDescription &desc);
	virtual ~LHESource();

    private:
	virtual void endJob() override;
	virtual void beginRun(edm::Run &run) override;
	virtual void endRun(edm::Run &run) override;
 	virtual bool setRunAndEventInfo(edm::EventID&, edm::TimeValue_t&) override;
	virtual boost::shared_ptr<edm::RunPrincipal> readRun_(boost::shared_ptr<edm::RunPrincipal> runPrincipal) override;
	virtual boost::shared_ptr<edm::LuminosityBlockPrincipal> readLuminosityBlock_(boost::shared_ptr<edm::LuminosityBlockPrincipal> lumiPrincipal) override;
	virtual edm::EventPrincipal* readEvent_(edm::EventPrincipal& eventPrincipal) override;
        virtual void produce(edm::Event&) {}

	void nextEvent();

	std::auto_ptr<lhef::LHEReader>		reader;

	boost::shared_ptr<lhef::LHERunInfo>	runInfoLast;
	boost::shared_ptr<lhef::LHERunInfo>	runInfo;
	boost::shared_ptr<lhef::LHEEvent>	partonLevel;

	boost::ptr_deque<LHERunInfoProduct>	runInfoProducts;
	bool					wasMerged;
	edm::LHEProvenanceHelper		lheProvenanceHelper_;
	edm::ProcessHistoryID			phid_;
	boost::shared_ptr<edm::RunPrincipal>	runPrincipal_;
};

#endif // GeneratorInterface_LHEInterface_LHESource_h

