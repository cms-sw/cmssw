#ifndef GeneratorInterface_LHEInterface_LHESource_h
#define GeneratorInterface_LHEInterface_LHESource_h

#include <memory>

#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/LHEInterface/interface/LesHouches.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHEReader.h"

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

	std::auto_ptr<lhef::LHEReader>	reader;
	std::auto_ptr<lhef::HEPRUP>	heprup;
	std::auto_ptr<lhef::HEPEUP>	hepeup;

	unsigned int			skipEvents;
};

#endif // GeneratorInterface_LHEInterface_LHESource_h
