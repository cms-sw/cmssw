#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>

#include <boost/bind.hpp>

#include "FWCore/Sources/interface/ExternalInputSource.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

#include "GeneratorInterface/Herwig6Interface/interface/Herwig6Instance.h"

#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"

#include "MCatNLOSource.h"


// MC@NLO v3.4 LHE file rountines (mcatnlo_hwlhin.f)
extern "C" {
  void mcatnloupinit_(int*, const char*, int);
  void mcatnloupevnt_(int*, int*, int*);
}


using namespace lhef;

MCatNLOSource::MCatNLOSource(const edm::ParameterSet &params,
                     const edm::InputSourceDescription &desc) :
	ExternalInputSource(params, desc, false),
        gen::Herwig6Instance(0),
	skipEvents(params.getUntrackedParameter<unsigned int>("skipEvents", 0)),
        nEvents(0),
	processCode(params.getParameter<int>("processCode"))
{
        std::vector<std::string> allFileNames = fileNames();

        // Only one filename
	if (allFileNames.size() != 1)
	  throw cms::Exception("Generator|MCatNLOInterface")
	    << "MCatNLOSource needs exactly one file specified. " <<std::endl;

	fileName = allFileNames[0];

	// Strip the "file:" prefix 
	if (fileName.find("file:") != 0)
	  throw cms::Exception("Generator|MCatNLOInterface") << "MCatNLOSource only supports the file: scheme. "<<std::endl;
	fileName.erase(0, 5);

	// open input file
	reader.reset(new std::ifstream(fileName.c_str()));

	produces<LHEEventProduct>();
	produces<LHERunInfoProduct, edm::InRun>();
}

MCatNLOSource::~MCatNLOSource()
{
}

void MCatNLOSource::endJob()
{
  reader.reset();
}

void MCatNLOSource::nextEvent()
{
  return;
}

template<typename T>
static std::string makeConfigLine(const char *var, T value)
{
  std::ostringstream ss;
  ss << var << " = " << value << "\n";
  return ss.str();
}

template<typename T>
static std::string makeConfigLine(const char *var, unsigned int index, T value)
{
  std::ostringstream ss;
  ss << var << "(" << index << ") = " << value << "\n";
  return ss.str();
}

void MCatNLOSource::beginRun(edm::Run &run)
{
  InstanceWrapper wrapper(this);

  // call UPINIT privided by MC@NLO (v3.4)
  mcatnloupinit_(&processCode, fileName.c_str(), fileName.length());

  // fill HEPRUP common block and store in edm::Run
  lhef::HEPRUP heprup;
  lhef::CommonBlocks::readHEPRUP(&heprup);

  // make sure we write a valid LHE header, Herwig6Hadronizer
  // will interpret it correctly and set up LHAPDF
  heprup.PDFGUP.first = 0;
  heprup.PDFGUP.second = 0;

  std::auto_ptr<LHERunInfoProduct> runInfo(new LHERunInfoProduct(heprup));

  LHERunInfoProduct::Header hw6header("herwig6header");
  hw6header.addLine("\n# Herwig6 parameters\n");
  hw6header.addLine(makeConfigLine("IPROC", processCode));
  // add lines for parameter that have been touched by UPINIT
  if(mcpars_.emmins) 
    hw6header.addLine(makeConfigLine("EMMIN", mcpars_.emmin));
  if(mcpars_.emmaxs) 
    hw6header.addLine(makeConfigLine("EMMAX", mcpars_.emmax));
  if(mcpars_.gammaxs) 
    hw6header.addLine(makeConfigLine("GAMMAX",mcpars_.gammax));
  for(unsigned int i=0; i<1000; ++i) {
    if(mcpars_.rmasss[i])
      hw6header.addLine(makeConfigLine("RMASS",i+1,mcpars_.rmass[i]));
  }
  
  runInfo->addHeader(hw6header);

  run.put(runInfo);

  return;
}

bool MCatNLOSource::produce(edm::Event &event)
{
  InstanceWrapper wrapper(this);

  int lastEventDone=0;
  int ihpro=0;
  // skip events if asked to...
  while(skipEvents--) {
    mcatnloupevnt_(&processCode,&lastEventDone,&ihpro);
    if(lastEventDone) return false;
  }

  // call UPINIT privided by MC@NLO (v3.4)
  mcatnloupevnt_(&processCode,&lastEventDone,&ihpro);

  if(lastEventDone) return false;

  // fill HEPRUP common block and store in edm::Run
  lhef::HEPEUP hepeup;
  lhef::CommonBlocks::readHEPEUP(&hepeup);
  std::auto_ptr<LHEEventProduct> lhEvent(new LHEEventProduct(hepeup));
  lhEvent->addComment(makeConfigLine("#IHPRO", ihpro));
  event.put(lhEvent);

  return true;
}

bool MCatNLOSource::hwwarn(const std::string &fn, int code)
{
  // dummy ignoring useless HERWIG warnings
  return true;
}

DEFINE_ANOTHER_FWK_INPUT_SOURCE(MCatNLOSource);
