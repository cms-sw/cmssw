#include "DQM/RPCMonitorClient/interface/RPCMonitorLinkSynchroMerger.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>
#include <fstream>

using namespace edm;
using namespace std;



RPCMonitorLinkSynchroMerger::RPCMonitorLinkSynchroMerger(const edm::ParameterSet& cfg)
  : RPCMonitorLinkSynchro(cfg), isInitialised(false)
{}

void RPCMonitorLinkSynchroMerger::beginRun(const edm::Run& ev, const edm::EventSetup& es)
{
  RPCMonitorLinkSynchro::beginRun(ev,es);
  if(!isInitialised) {
    isInitialised = true;
    std::vector<std::string> fileNames = theConfig.getUntrackedParameter<std::vector<std::string> >("preFillLinkSynchroFileNames");
    if (!fileNames.empty())for (std::vector<std::string>::const_iterator it=fileNames.begin(); it != fileNames.end(); ++it) preFillFromFile(*it);
  }
}

void RPCMonitorLinkSynchroMerger::preFillFromFile(const std::string & fileName)
{
  std::ifstream file( fileName.c_str() );
  if ( !file ) {
    edm::LogError(" ** RPCMonitorLinkSynchroMerger ** ") << " cant open data file: " << fileName;
    return;
  } else {
    edm::LogInfo("RPCMonitorLinkSynchroMerger, read data from: ") <<fileName;
  }
  string line, lbName, tmp;
  unsigned int hits[8];
  while (getline(file,line) ) {
    stringstream str(line);
    str >> lbName
        >>tmp>>tmp>>tmp>>tmp>>tmp
        >>hits[0]>>hits[1]>>hits[2]>>hits[3]>>hits[4]>>hits[5]>>hits[6]>>hits[7];
    if (str.good()) theSynchroStat.add(lbName,hits); 
  }
  file.close();
}
