#ifndef GeneratorInterface_LHEInterface_LHESource_h
#define GeneratorInterface_LHEInterface_LHESource_h

#include <memory>

#include <boost/shared_ptr.hpp>

#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/LHEInterface/interface/LHEReader.h"
#include "GeneratorInterface/LHEInterface/interface/LHECommon.h"
#include "GeneratorInterface/LHEInterface/interface/Hadronisation.h"
#include "GeneratorInterface/LHEInterface/interface/JetMatching.h"

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
	virtual void endRun(edm::Run &run);
	virtual bool produce(edm::Event &event);

	std::auto_ptr<lhef::LHEReader>		reader;
	unsigned int				skipEvents;
	unsigned int				eventsToPrint;
	std::auto_ptr<lhef::Hadronisation>	hadronisation;
	std::auto_ptr<lhef::JetMatching>	jetMatching;

	boost::shared_ptr<lhef::LHECommon>	common;

	double					extCrossSect;
	double					extFilterEff;

    private:
	void init(const edm::ParameterSet &params);
};

#endif // GeneratorInterface_LHEInterface_LHESource_h
