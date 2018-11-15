///////////////////////////////////////////////////////////////////////
///
/// Module to read trigger bit mappings (AlCaRecoTriggerBits) from
/// DB and put out as text. 
/// Several output formats can be configured via parameter 'outputType':
/// - simple text ('text'),
/// - text in format of a Twiki table for cut and paste ('twiki')
/// - or a python snippet to be inserted into configuration of the
///   AlCaRecoTriggerBitsRcdWrite ('python').
///
/// Where to put the text is decided by parameter 'rawFileName':
/// - if empty, use MessageLogger
/// - otherwise open file <rawFileName>.<suffix> where the suffix
///   is chosen to match 'outputType'.
///
///////////////////////////////////////////////////////////////////////

#include <string>
#include <map>
//#include <vector>
#include <sstream>
#include <fstream>

// Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

// What I want to read:
#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"
#include "CondFormats/DataRecord/interface/AlCaRecoTriggerBitsRcd.h"


class  AlCaRecoTriggerBitsRcdRead : public edm::EDAnalyzer {
public:
  explicit  AlCaRecoTriggerBitsRcdRead(const edm::ParameterSet &cfg);
  ~AlCaRecoTriggerBitsRcdRead() override {}
  
  void analyze(const edm::Event &evt, const edm::EventSetup &evtSetup) override {}
  void beginRun(const edm::Run &run, const edm::EventSetup &evtSetup) override;
  void endJob() override;

  
private:
  // types
  enum OutputType {kText, kTwiki, kPython}; //kHtml};

  // methods
  OutputType stringToEnum(const std::string &outputType) const;
  void printMap(edm::RunNumber_t firstRun, edm::RunNumber_t lastRun,
		const AlCaRecoTriggerBits &triggerMap) const;

  // members
  const OutputType outputType_;
  edm::ESWatcher<AlCaRecoTriggerBitsRcd> watcher_;
  edm::RunNumber_t firstRun_; 
  edm::RunNumber_t lastRun_;
  AlCaRecoTriggerBits lastTriggerBits_;
  std::unique_ptr<std::ofstream> output_;
};

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

AlCaRecoTriggerBitsRcdRead::AlCaRecoTriggerBitsRcdRead(const edm::ParameterSet& cfg)
  : outputType_(this->stringToEnum(cfg.getUntrackedParameter<std::string>("outputType"))),
    firstRun_(0), lastRun_(0)
{
  //   edm::LogInfo("") << "@SUB=AlCaRecoTriggerBitsRcdRead" 
  // 		   << cfg.getParameter<std::string>("@module_label");

  std::string fileName(cfg.getUntrackedParameter<std::string>("rawFileName"));
  switch (outputType_) { // now append suffix
  case kText:   fileName += ".txt";   break;
  case kPython: fileName += ".py";    break;
  case kTwiki:  fileName += ".twiki"; break;
  }
  if (!fileName.empty()) {
    output_.reset(new std::ofstream(fileName.c_str()));
    if (!output_->good()) {
      edm::LogError("IOproblem") << "Could not open output file " << fileName << ".";
      output_.reset();
    }
  }

}

///////////////////////////////////////////////////////////////////////
AlCaRecoTriggerBitsRcdRead::OutputType
AlCaRecoTriggerBitsRcdRead::stringToEnum(const std::string &outputTypeStr) const
{
  if (outputTypeStr == "text") return kText;
  if (outputTypeStr == "twiki") return kTwiki;
  if (outputTypeStr == "python") return kPython;
  // if (outputTypeStr == "html") return kHtml;

  throw cms::Exception("BadConfig") << "AlCaRecoTriggerBitsRcdRead: "
 				    << "outputType '" << outputTypeStr << "' not known,"
 				    << " use 'text', 'twiki' or 'python'\n";

  return kTwiki; // never reached, to please compiler
}

///////////////////////////////////////////////////////////////////////
void AlCaRecoTriggerBitsRcdRead::beginRun(const edm::Run& run, const edm::EventSetup& iSetup)
{
  if (watcher_.check(iSetup)) { // new IOV for this run
    // Print last IOV - if there has already been one:
    if (lastRun_ != 0) this->printMap(firstRun_, lastRun_, lastTriggerBits_);
  
    // Get AlCaRecoTriggerBits from EventSetup:
    edm::ESHandle<AlCaRecoTriggerBits> triggerBits;
    iSetup.get<AlCaRecoTriggerBitsRcd>().get(triggerBits);
    lastTriggerBits_ = *triggerBits; // copy for later use
    firstRun_ = run.run();           // keep track where it started
  }

  lastRun_ = run.run(); // keep track of last visited run
}

///////////////////////////////////////////////////////////////////////
void AlCaRecoTriggerBitsRcdRead::endJob()
{
  // Print for very last IOV, not treated yet in beginRun(..):
  this->printMap(firstRun_, lastRun_, lastTriggerBits_);
}

///////////////////////////////////////////////////////////////////////
void AlCaRecoTriggerBitsRcdRead::printMap(edm::RunNumber_t firstRun,
					  edm::RunNumber_t lastRun, 
					  const AlCaRecoTriggerBits &triggerBits) const
{
  // Get map of strings to concatenated list of names of HLT paths:
  typedef std::map<std::string, std::string> TriggerMap;
  const TriggerMap &triggerMap = triggerBits.m_alcarecoToTrig;

  // Collect output for given run numbers via ostringstream.
  // Format depends on outputType_ configuration.
  std::ostringstream output;
  switch (outputType_) {
  case kPython:
    output << "  triggerLists = cms.VPSet(\n";
    // no 'break;'!
  case kText:
    output << "#\n# AlCaRecoTriggerBits settings for IOV "
	   << firstRun << "-" << lastRun << ":\n#\n";
    break;
  case kTwiki:
    output << "---+++++ *IOV*: " << firstRun << "-" << lastRun << "\n"
	   << "| *TriggerBits list key* | *HLT paths* |\n";
    break;
  }

  //  if (outputType_ == kPython) output << "  triggerLists = cms.VPSet(\n";

  // loop over entries in map
  for (TriggerMap::const_iterator i = triggerMap.begin(); i != triggerMap.end(); ++i) {

    if (outputType_ == kPython && i != triggerMap.begin()) output << ",\n";

    switch (outputType_) {
    case kPython:
      output << "      cms.PSet(listName = cms.string('" << i->first << "'),\n"
	     << "               hltPaths = cms.vstring(";
      break;
    case kText:
      output << "trigger list key: '" << i->first << "'\npaths:\n";
      break;
    case kTwiki:
      output << "| '" << i->first << "' | ";
    }
    // We must avoid a map<string,vector<string> > in DB for performance reason,
    // so the paths are mapped into one string separated by ';':
    const std::vector<std::string> paths = triggerBits.decompose(i->second);
    for (unsigned int iPath = 0; iPath < paths.size(); ++iPath) {
      if (iPath != 0) {
	output << ", "; // next path
	switch (outputType_) {
	case kPython: // only 2 per line
	case kText:   // only 4 per line
	  if (0 == (iPath % (outputType_ == kPython ? 2 : 4))) {
	    output << "\n";
	    if (outputType_ == kPython) output << "                                      ";
	  }
	  break;
	case kTwiki: // Twiki will handle that
	  break;
	}
      }
      output << "'" << paths[iPath] << "'";
    }
    switch (outputType_) {
    case kPython:
      output << ")\n              )";
      break;
    case kText:
      output << "\n#\n";
      break;
    case kTwiki:
      output << " |\n";
    }
  }
  if (outputType_ == kPython) output << "\n      ) # closing of VPSet triggerLists\n"; 
  
  // Final output - either message logger or output file:
  if (output_.get()) *output_ << output.str();
  else edm::LogInfo("") << output.str();
}



//define this as a plug-in
DEFINE_FWK_MODULE(AlCaRecoTriggerBitsRcdRead);
