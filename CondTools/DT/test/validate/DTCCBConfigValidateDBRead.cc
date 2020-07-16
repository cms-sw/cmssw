
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondTools/DT/test/validate/DTCCBConfigValidateDBRead.h"
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"
#include "CondFormats/DataRecord/interface/DTCCBConfigRcd.h"

DTCCBConfigValidateDBRead::DTCCBConfigValidateDBRead(edm::ParameterSet const& p)
    : dataFileName(p.getParameter<std::string>("chkFile")), elogFileName(p.getParameter<std::string>("logFile")) {}

DTCCBConfigValidateDBRead::DTCCBConfigValidateDBRead(int i) {}

DTCCBConfigValidateDBRead::~DTCCBConfigValidateDBRead() {}

void DTCCBConfigValidateDBRead::analyze(const edm::Event& e, const edm::EventSetup& context) {
  using namespace edm::eventsetup;
  // Context is not used.
  std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
  std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
  std::stringstream run_fn;
  run_fn << "run" << e.id().run() << dataFileName;
  std::ifstream chkFile(run_fn.str().c_str());
  std::ofstream logFile(elogFileName.c_str(), std::ios_base::app);
  edm::ESHandle<DTCCBConfig> conf;
  context.get<DTCCBConfigRcd>().get(conf);
  std::cout << conf->version() << std::endl;
  std::cout << std::distance(conf->begin(), conf->end()) << " data in the container" << std::endl;
  int whe;
  int sta;
  int sec;
  std::vector<DTConfigKey> fullConf;
  std::vector<DTConfigKey> fchkConf;
  std::vector<int> ccbConf;
  std::vector<int> chkConf;
  int nbtype;
  int ibtype;
  int nbrick;
  int fkey;
  int bkey;

  int status;
  chkFile >> nbtype;
  fullConf.reserve(nbtype);
  while (nbtype--) {
    chkFile >> ibtype >> fkey;
    DTConfigKey confKey;
    confKey.confType = ibtype;
    confKey.confKey = fkey;
    fchkConf.push_back(confKey);
  }
  fullConf = conf->fullKey();
  if (cfrDiff(fchkConf, fullConf))
    logFile << "MISMATCH IN WRITING AND READING full configuration" << std::endl;

  while (chkFile >> whe >> sta >> sec >> nbrick) {
    chkConf.clear();
    while (nbrick--) {
      chkFile >> bkey;
      chkConf.push_back(bkey);
    }
    status = conf->configKey(whe, sta, sec, ccbConf);
    if (status)
      logFile << "ERROR while reading CCB configuration" << whe << " " << sta << " " << sec << " , status = " << status
              << std::endl;
    if (cfrDiff(chkConf, ccbConf))
      logFile << "MISMATCH WHEN READING CCB configuration " << whe << " " << sta << " " << sec << std::endl;
  }
}

void DTCCBConfigValidateDBRead::endJob() {
  std::ifstream logFile(elogFileName.c_str());
  char* line = new char[1000];
  int errors = 0;
  std::cout << "CCBConfig validation result:" << std::endl;
  while (logFile.getline(line, 1000)) {
    std::cout << line << std::endl;
    errors++;
  }
  if (!errors) {
    std::cout << " ********************************* " << std::endl;
    std::cout << " ***                           *** " << std::endl;
    std::cout << " ***      NO ERRORS FOUND      *** " << std::endl;
    std::cout << " ***                           *** " << std::endl;
    std::cout << " ********************************* " << std::endl;
  }
  return;
}

bool DTCCBConfigValidateDBRead::cfrDiff(const std::vector<int>& l_conf, const std::vector<int>& r_conf) {
  if (l_conf.size() != r_conf.size())
    return true;
  std::vector<int>::const_iterator l_iter = l_conf.begin();
  std::vector<int>::const_iterator l_iend = l_conf.end();
  std::vector<int>::const_iterator r_iter = r_conf.begin();
  std::vector<int>::const_iterator r_iend = r_conf.end();
  while ((l_iter != l_iend) && (r_iter != r_iend)) {
    if (*l_iter++ != *r_iter++)
      return true;
  }
  return false;
}

bool DTCCBConfigValidateDBRead::cfrDiff(const std::vector<DTConfigKey>& l_conf,
                                        const std::vector<DTConfigKey>& r_conf) {
  if (l_conf.size() != r_conf.size())
    return true;
  std::vector<DTConfigKey>::const_iterator l_iter = l_conf.begin();
  std::vector<DTConfigKey>::const_iterator l_iend = l_conf.end();
  std::vector<DTConfigKey>::const_iterator r_iter = r_conf.begin();
  std::vector<DTConfigKey>::const_iterator r_iend = r_conf.end();
  while ((l_iter != l_iend) && (r_iter != r_iend)) {
    const DTConfigKey& l_key = *l_iter++;
    const DTConfigKey& r_key = *r_iter++;
    if (l_key.confType != r_key.confType)
      return true;
    if (l_key.confKey != r_key.confKey)
      return true;
  }
  return false;
}

DEFINE_FWK_MODULE(DTCCBConfigValidateDBRead);
