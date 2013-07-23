#ifndef GeneratorInterface_ThePEGInterface_ThePEGInterface_h
#define GeneratorInterface_ThePEGInterface_ThePEGInterface_h

/** \class ThePEGInterface
 *  $Id: ThePEGInterface.h,v 1.11 2009/11/04 18:28:40 stober Exp $
 *  
 *  Oliver Oberst <oberst@ekp.uni-karlsruhe.de>
 *  Fred-Markus Stober <stober@ekp.uni-karlsruhe.de>
 */

#include <memory>
#include <string>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>
#include <HepMC/PdfInfo.h>
#include <HepMC/IO_BaseClass.h>

#include <ThePEG/Repository/EventGenerator.h>
#include <ThePEG/EventRecord/Event.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/ThePEGInterface/interface/RandomEngineGlue.h"

class ThePEGInterface {
    public:
	ThePEGInterface(const edm::ParameterSet &params);
	virtual ~ThePEGInterface();

    protected:
	void initRepository(const edm::ParameterSet &params) const;
	void initGenerator();
	void flushRandomNumberGenerator();

	static std::auto_ptr<HepMC::GenEvent>
				convert(const ThePEG::EventPtr &event);
	static void clearAuxiliary(HepMC::GenEvent *hepmc,
	                           HepMC::PdfInfo *pdf);
	static void fillAuxiliary(HepMC::GenEvent *hepmc,
	                          HepMC::PdfInfo *pdf,
	                          const ThePEG::EventPtr &event);

	static double pthat(const ThePEG::EventPtr &event);

	std::string dataFile(const std::string &fileName) const;
	std::string dataFile(const edm::ParameterSet &pset,
	                     const std::string &paramName) const;

	ThePEG::EGPtr				eg_;
	std::auto_ptr<HepMC::IO_BaseClass>	iobc_;

    private:
	boost::shared_ptr<ThePEG::RandomEngineGlue::Proxy>
						randomEngineGlueProxy_;

	const std::string			dataLocation_;
	const std::string			generator_;
	const std::string			run_;
	const std::string			dumpConfig_;
	const unsigned int			skipEvents_;
};

#endif // GeneratorInterface_ThePEGInterface_ThePEGInterface_h
