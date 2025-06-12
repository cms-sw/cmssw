/**
Marco A. Harrendorf
**/

#ifndef GeneratorInterface_Herwig7Interface_Herwig7HepMC3Interface_h
#define GeneratorInterface_Herwig7Interface_Herwig7HepMC3Interface_h

#include <memory>
#include <string>

#include <HepMC/GenEvent.h>
#include <HepMC/PdfInfo.h>
#include <HepMC/IO_BaseClass.h>

#include <HepMC3/GenEvent.h>
#include <HepMC3/GenParticle.h>
#include <HepMC3/GenPdfInfo.h>

#include <ThePEG/Repository/EventGenerator.h>
#include <ThePEG/EventRecord/Event.h>
#include <GeneratorInterface/Herwig7Interface/interface/HepMC3Helper.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/Herwig7Interface/interface/RandomEngineGlue.h"
#include "GeneratorInterface/Herwig7Interface/interface/HerwigUIProvider.h"

namespace CLHEP {
  class HepRandomEngine;
}

class Herwig7HepMC3Interface {
public:
  Herwig7HepMC3Interface(const edm::ParameterSet &params);
  ~Herwig7HepMC3Interface() noexcept;

  void setPEGRandomEngine(CLHEP::HepRandomEngine *);

  ThePEG::EGPtr eg_;

protected:
  void initRepository(const edm::ParameterSet &params);
  bool initGenerator();
  void flushRandomNumberGenerator();

  static std::unique_ptr<HepMC3::GenEvent> convert(const ThePEG::EventPtr &event);

  static double pthat(const ThePEG::EventPtr &event);

  //std::unique_ptr<HepMC::IO_BaseClass> iobc_;

  // HerwigUi contains settings piped to Herwig7
  std::shared_ptr<Herwig::HerwigUIProvider> HwUI_;

  /**
        * Function calls Herwig event generator via API
	*
	* According to the run mode different steps of event generation are done
	**/
  void callHerwigGenerator();

  // The Inputfile ist created according to the parameter set
  void createInputFile(const edm::ParameterSet &params);

private:
  std::shared_ptr<ThePEG::RandomEngineGlue::Proxy> randomEngineGlueProxy_;

  const std::string dataLocation_;
  const std::string generator_;
  const std::string run_;
  // File name containing Herwig input config
  std::string dumpConfig_;
  const unsigned int skipEvents_;
  CLHEP::HepRandomEngine *randomEngine;
};

#endif  // GeneratorInterface_Herwig7Interface_Herwig7HepMC3Interface_h
