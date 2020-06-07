#include <iostream>
#include <fstream>
#include <stdexcept>

#include "tmEventSetup/tmEventSetup.hh"

#include "tmEventSetup/esTriggerMenu.hh"
#include "tmEventSetup/esAlgorithm.hh"
#include "tmEventSetup/esCondition.hh"
#include "tmEventSetup/esObject.hh"
#include "tmEventSetup/esCut.hh"
#include "tmEventSetup/esScale.hh"
#include "tmGrammar/Algorithm.hh"

#include "CondTools/L1TriggerExt/interface/L1ConfigOnlineProdBaseExt.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuO2ORcd.h"

class L1TUtmTriggerMenuOnlineProd : public L1ConfigOnlineProdBaseExt<L1TUtmTriggerMenuO2ORcd, L1TUtmTriggerMenu> {
private:
public:
  std::unique_ptr<const L1TUtmTriggerMenu> newObject(const std::string& objectKey,
                                                     const L1TUtmTriggerMenuO2ORcd& record) override;

  L1TUtmTriggerMenuOnlineProd(const edm::ParameterSet&);
  ~L1TUtmTriggerMenuOnlineProd(void) override {}
};

L1TUtmTriggerMenuOnlineProd::L1TUtmTriggerMenuOnlineProd(const edm::ParameterSet& iConfig)
    : L1ConfigOnlineProdBaseExt<L1TUtmTriggerMenuO2ORcd, L1TUtmTriggerMenu>(iConfig) {
  m_setWhatProduced(iConfig);
    }

std::unique_ptr<const L1TUtmTriggerMenu> L1TUtmTriggerMenuOnlineProd::newObject(const std::string& objectKey,
                                                                                const L1TUtmTriggerMenuO2ORcd& record) {
  std::string stage2Schema = "CMS_TRG_L1_CONF";
  edm::LogInfo("L1-O2O: L1TUtmTriggerMenuOnlineProd") << "Producing L1TUtmTriggerMenu with key =" << objectKey;

  if (objectKey.empty()) {
    edm::LogError("L1-O2O: L1TUtmTriggerMenuOnlineProd") << "Key is empty, returning empty L1TUtmTriggerMenu object";
    throw std::runtime_error("Empty objectKey");
  }

  std::vector<std::string> queryColumns;
  queryColumns.push_back("CONF");

  l1t::OMDSReader::QueryResults queryResult = m_omdsReader.basicQuery(
      queryColumns, stage2Schema, "UGT_L1_MENU", "UGT_L1_MENU.ID", m_omdsReader.singleAttribute(objectKey));

  if (queryResult.queryFailed() || queryResult.numberRows() != 1) {
    edm::LogError("L1-O2O: L1TUtmTriggerMenuOnlineProd") << "Cannot get UGT_L1_MENU.CONF for ID = " << objectKey;
    throw std::runtime_error("Broken key");
  }

  std::string l1Menu;
  queryResult.fillVariable("CONF", l1Menu);

  std::istringstream iss(l1Menu);

  const L1TUtmTriggerMenu* cmenu = reinterpret_cast<const L1TUtmTriggerMenu*>(tmeventsetup::getTriggerMenu(iss));
  return std::unique_ptr<const L1TUtmTriggerMenu>(cmenu);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TUtmTriggerMenuOnlineProd);
