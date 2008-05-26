#ifndef GeneratorInterface_LHEInterface_LHESource_h
#define GeneratorInterface_LHEInterface_LHESource_h

#include <memory>

#include <boost/shared_ptr.hpp>

#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"

namespace lhef {
	class LHERunInfo;
	class LHEEvent;
	class LHEReader;
}

class LHESource : public edm::GeneratedInputSource {
    public:
	explicit LHESource(const edm::ParameterSet &params,
	                   const edm::InputSourceDescription &desc);
	virtual ~LHESource();

    protected:
	LHESource(const edm::ParameterSet &params,
	          const edm::InputSourceDescription &desc,
	          lhef::LHEReader *reader);

	virtual void endJob();
	virtual void beginRun(edm::Run &run);
	virtual bool produce(edm::Event &event);

	virtual void nextEvent();

	std::auto_ptr<lhef::LHEReader>		reader;

	boost::shared_ptr<lhef::LHERunInfo>	runInfo;
	boost::shared_ptr<lhef::LHEEvent>	partonLevel;

	unsigned int				skipEvents;
};

#endif // GeneratorInterface_LHEInterface_LHESource_h
