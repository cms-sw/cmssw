///////////////////////////////////////////////////////////////////////
///
/// Module to read trigger bit mappings (AlCaRecoTriggerBits) from
/// DB and put as text. If configured, a python snippet to be inserted
/// into configuration of the AlCaRecoTriggerBitsRcdWrite is put out.
///
///////////////////////////////////////////////////////////////////////

#include <string>
#include <map>
//#include <vector>
#include <sstream>

// Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

// What I want to read:
#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"
#include "CondFormats/DataRecord/interface/AlCaRecoTriggerBitsRcd.h"


class  AlCaRecoTriggerBitsRcdRead : public edm::EDAnalyzer {
public:
  explicit  AlCaRecoTriggerBitsRcdRead(const edm::ParameterSet& cfg);
  ~AlCaRecoTriggerBitsRcdRead() {}
  
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {}
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& evtSetup);

  
private:
  enum OutputType {kText, kTwiki, kPython}; //kHtml
  OutputType stringToEnum(const std::string &outputType) const;

  const OutputType outputType_;
};

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

AlCaRecoTriggerBitsRcdRead::AlCaRecoTriggerBitsRcdRead(const edm::ParameterSet& cfg)
  : outputType_(this->stringToEnum(cfg.getUntrackedParameter<std::string>("outputType")))
{
  //   edm::LogInfo("") << "@SUB=AlCaRecoTriggerBitsRcdRead" 
  // 		   << cfg.getParameter<std::string>("@module_label");
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
  // Get map of strings to concatenated list of names of HLT paths from EventSetup:
  edm::ESHandle<AlCaRecoTriggerBits> triggerBits;
  iSetup.get<AlCaRecoTriggerBitsRcd>().get(triggerBits);
  typedef std::map<std::string, std::string> TriggerMap;
  const TriggerMap &triggerMap = triggerBits->m_alcarecoToTrig;

  // Collect output in an ostringstream, starting with run number.
  // Format depends on outputType_ configuration:
  std::ostringstream output;
  output << "#\n# Run number " << run.run() << " has these AlCaRecoTriggerBits settings:\n#\n";
  if (outputType_ == kPython) output << "  triggerLists = cms.VPSet(\n";

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
      //'HLT_Ele15_SW_L1R', 'HLT_DoubleEle10_SW_L1R', 'HLT_Ele10_LW_L1R',
      //'HLT_DoubleEle5_SW_L1R' |
    }
    // We must avoid a map<string,vector<string> > in DB for performance reason,
    // so the paths are mapped into one string separated by ';':
    const std::vector<std::string> paths = triggerBits->decompose(i->second);
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
  if (outputType_ == kPython) output << "\n      ) # closing of VPSet triggerLists"; 
  
  // Final output:
  edm::LogInfo("") << "@SUB=beginRun" << output.str();
}

//define this as a plug-in
DEFINE_FWK_MODULE(AlCaRecoTriggerBitsRcdRead);
