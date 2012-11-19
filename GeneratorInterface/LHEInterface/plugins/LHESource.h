#ifndef GeneratorInterface_LHEInterface_LHESource_h
#define GeneratorInterface_LHEInterface_LHESource_h

#include <memory>

#include <boost/shared_ptr.hpp>
#include <boost/ptr_container/ptr_deque.hpp>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Sources/interface/ProducerSourceFromFiles.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"

namespace lhef {
	class LHERunInfo;
	class LHEEvent;
	class LHEReader;
}

class LHERunInfoProduct;

class LHESource : public edm::ProducerSourceFromFiles {
    public:
	explicit LHESource(const edm::ParameterSet &params,
	                   const edm::InputSourceDescription &desc);
	virtual ~LHESource();

    private:
	virtual void endJob();
	virtual void beginRun(edm::Run &run);
	virtual void endRun(edm::Run &run);
	virtual bool setRunAndEventInfo(edm::EventID&, edm::TimeValue_t&);
	virtual void produce(edm::Event &event);

	virtual void nextEvent();

	std::auto_ptr<lhef::LHEReader>		reader;

	boost::shared_ptr<lhef::LHERunInfo>	runInfoLast;
	boost::shared_ptr<lhef::LHERunInfo>	runInfo;
	boost::shared_ptr<lhef::LHEEvent>	partonLevel;

	boost::ptr_deque<LHERunInfoProduct>	runInfoProducts;
	bool					wasMerged;
};

#endif // GeneratorInterface_LHEInterface_LHESource_h
