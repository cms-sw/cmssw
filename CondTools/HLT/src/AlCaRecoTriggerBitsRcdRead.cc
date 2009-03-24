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

// What I want to read:
#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"
#include "CondFormats/DataRecord/interface/AlCaRecoTriggerBitsRcd.h"


class  AlCaRecoTriggerBitsRcdRead : public edm::EDAnalyzer {
public:
  explicit  AlCaRecoTriggerBitsRcdRead(const edm::ParameterSet& cfg);
  ~AlCaRecoTriggerBitsRcdRead() {}
  
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);

  
private:
  const bool pythonOutput_;
};

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

AlCaRecoTriggerBitsRcdRead::AlCaRecoTriggerBitsRcdRead(const edm::ParameterSet& cfg)
  : pythonOutput_(cfg.getUntrackedParameter<bool>("pythonOutput"))
{
  //   edm::LogInfo("") << "@SUB=AlCaRecoTriggerBitsRcdRead" 
  // 		   << cfg.getParameter<std::string>("@module_label");
}
  
///////////////////////////////////////////////////////////////////////
void AlCaRecoTriggerBitsRcdRead::analyze(const edm::Event& evt, const edm::EventSetup& iSetup)
{

  // Get map of strings to concatenated list of names of HLT paths from EventSetup:
  edm::ESHandle<AlCaRecoTriggerBits> triggerBits;
  iSetup.get<AlCaRecoTriggerBitsRcd>().get(triggerBits);
  typedef std::map<std::string, std::string> TriggerMap;
  const TriggerMap &triggerMap = triggerBits->m_alcarecoToTrig;

  // Collect output in an ostringstream, starting with run number.
  // Format depends on pythonOutput configuration:
  std::ostringstream output;
  output << "#\n# Run number " << evt.run() << " has these AlCaRecoTriggerBits settings:\n#\n";
  if (pythonOutput_) output << "  triggerLists = cms.VPSet(\n";

  // loop over entries in map
  for (TriggerMap::const_iterator i = triggerMap.begin(); i != triggerMap.end(); ++i) {

    if (pythonOutput_ && i != triggerMap.begin()) output << ",\n";

    if (pythonOutput_) {
      output << "      cms.PSet(listName = cms.string('" << i->first << "'),\n"
	     << "               hltPaths = cms.vstring(";
    } else {
      output << "trigger list key: '" << i->first << "'\npaths:\n";
    }
    // We must avoid a map<string,vector<string> > in DB for performance reason,
    // so the paths are mapped into one string separated by ';':
    const std::vector<std::string> paths = triggerBits->decompose(i->second);
    for (unsigned int iPath = 0; iPath < paths.size(); ++iPath) {
      if (iPath != 0) {
	output << ", "; // next path, only few per line:
	if (0 == (iPath % (pythonOutput_ ? 2 : 4))) {
	  output << "\n";
	  if (pythonOutput_) output << "                                      ";
	}
      }
      output << "'" << paths[iPath] << "'";
    }
    if (pythonOutput_) output << ")\n              )";
    else output << "\n#\n";
  }
  if (pythonOutput_) output << "\n      ) # closing of VPSet triggerLists"; 
  
  // Final output:
  edm::LogInfo("") << "@SUB=analyze" << output.str();
}

//define this as a plug-in
DEFINE_FWK_MODULE(AlCaRecoTriggerBitsRcdRead);
