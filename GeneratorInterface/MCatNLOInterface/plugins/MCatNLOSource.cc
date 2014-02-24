#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>

#include <boost/bind.hpp>

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
	ProducerSourceFromFiles(params, desc, false),
        gen::Herwig6Instance(0),
	skipEvents(params.getUntrackedParameter<unsigned int>("skipEvents", 0)),
        nEvents(0),
        ihpro(0),
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
  hw6header.addLine("\n");
  hw6header.addLine("# Herwig6 parameters\n");
  hw6header.addLine(makeConfigLine("IPROC", processCode));
  // add lines for parameter that have been touched by UPINIT
  if(mcpars_.emmins) 
    hw6header.addLine(makeConfigLine("EMMIN", mcpars_.emmin));
  if(mcpars_.emmaxs) 
    hw6header.addLine(makeConfigLine("EMMAX", mcpars_.emmax));
  if(mcpars_.gammaxs) 
    hw6header.addLine(makeConfigLine("GAMMAX",mcpars_.gammax));
  if(mcpars_.gamzs) 
    hw6header.addLine(makeConfigLine("GAMZ",mcpars_.gamz));
  if(mcpars_.gamws) 
    hw6header.addLine(makeConfigLine("GAMW",mcpars_.gamw));
  for(unsigned int i=0; i<1000; ++i) {
    if(mcpars_.rmasss[i])
      hw6header.addLine(makeConfigLine("RMASS",i+1,mcpars_.rmass[i]));
  }

  // other needed MC@NLO defaults (from mcatnlo_hwdriver.f v3.4)
  hw6header.addLine(makeConfigLine("SOFTME", false));
  hw6header.addLine(makeConfigLine("NOWGT", false));
  hw6header.addLine(makeConfigLine("NEGWTS", true));
  if(abs(processCode)==1705 || abs(processCode)==11705)
    hw6header.addLine(makeConfigLine("PSPLT",2,0.5));
  double wgtmax_=1.000001;
  hw6header.addLine(makeConfigLine("WGTMAX", wgtmax_));
  hw6header.addLine(makeConfigLine("AVABW", wgtmax_));
  hw6header.addLine(makeConfigLine("RLTIM",6, 1.0E-23));
  hw6header.addLine(makeConfigLine("RLTIM",12, 1.0E-23));
  

  runInfo->addHeader(hw6header);

  run.put(runInfo);

  return;
}

bool MCatNLOSource::setRunAndEventInfo(edm::EventID&, edm::TimeValue_t&)
{
  InstanceWrapper wrapper(this);

  int lastEventDone=0;
  ihpro=0;
  // skip events if asked to...

  while(skipEvents>0) {
    skipEvents--;
    mcatnloupevnt_(&processCode,&lastEventDone,&ihpro);
    if(lastEventDone) return false;
  }

  // call UPINIT privided by MC@NLO (v3.4)
  mcatnloupevnt_(&processCode,&lastEventDone,&ihpro);

  if(lastEventDone) return false;
  return true;
}

void MCatNLOSource::produce(edm::Event &event)
{
  InstanceWrapper wrapper(this);

  // fill HEPRUP common block and store in edm::Run
  lhef::HEPRUP heprup;
  lhef::HEPEUP hepeup;
  lhef::CommonBlocks::readHEPRUP(&heprup);
  lhef::CommonBlocks::readHEPEUP(&hepeup);
  hepeup.IDPRUP = heprup.LPRUP[0];
  std::auto_ptr<LHEEventProduct> lhEvent(new LHEEventProduct(hepeup,hepeup.XWGTUP));
  lhEvent->addComment(makeConfigLine("#IHPRO", ihpro));
  event.put(lhEvent);
}

bool MCatNLOSource::hwwarn(const std::string &fn, int code)
{
  // dummy ignoring useless HERWIG warnings
  return true;
}

DEFINE_FWK_INPUT_SOURCE(MCatNLOSource);
