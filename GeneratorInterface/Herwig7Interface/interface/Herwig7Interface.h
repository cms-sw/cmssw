/**
Marco A. Harrendorf
**/


#ifndef GeneratorInterface_Herwig7Interface_Herwig7Interface_h
#define GeneratorInterface_Herwig7Interface_Herwig7Interface_h



#include <memory>
#include <string>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>
#include <HepMC/PdfInfo.h>
#include <HepMC/IO_BaseClass.h>

#include <ThePEG/Repository/EventGenerator.h>
#include <ThePEG/EventRecord/Event.h>
#include <ThePEG/Vectors/HepMCTraits.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/Herwig7Interface/interface/RandomEngineGlue.h"
#include "GeneratorInterface/Herwig7Interface/interface/HerwigUIProvider.h"

namespace ThePEG {

  template<> struct HepMCTraits<HepMC::GenEvent> :
                public HepMCTraitsBase<
    HepMC::GenEvent, HepMC::GenParticle,
    HepMC::GenVertex, HepMC::Polarization,
    HepMC::PdfInfo> {};

}

namespace CLHEP {
  class HepRandomEngine;
}

class Herwig7Interface {
    public:
	Herwig7Interface(const edm::ParameterSet &params);
	~Herwig7Interface();

        void setPEGRandomEngine(CLHEP::HepRandomEngine*);

	ThePEG::EGPtr				eg_;



    protected:
	void initRepository(const edm::ParameterSet &params);
	bool initGenerator();
	void flushRandomNumberGenerator();

	static std::unique_ptr<HepMC::GenEvent>
				convert(const ThePEG::EventPtr &event);

	static double pthat(const ThePEG::EventPtr &event);

	

	std::unique_ptr<HepMC::IO_BaseClass>	iobc_;

	// HerwigUi contains settings piped to Herwig7
	Herwig::HerwigUIProvider* HwUI_;

	/**
        * Function calls Herwig event generator via API
	*
	* According to the run mode different steps of event generation are done
	**/
	void callHerwigGenerator();


	// The Inputfile ist created according to the parameter set
	void createInputFile(const edm::ParameterSet &params);



    private:
	boost::shared_ptr<ThePEG::RandomEngineGlue::Proxy>
						randomEngineGlueProxy_;

	const std::string			dataLocation_;
	const std::string			generator_;
	const std::string			run_;
	// File name containing Herwig input config 
	std::string				dumpConfig_;
	const unsigned int			skipEvents_;
};







#endif // GeneratorInterface_Herwig7Interface_Herwig7Interface_h
