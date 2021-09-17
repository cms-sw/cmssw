#include <strings.h>  // strcasecmp
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "tmEventSetup/tmEventSetup.hh"
#include "tmEventSetup/esTriggerMenu.hh"
#include "tmEventSetup/esAlgorithm.hh"
#include "tmEventSetup/esCondition.hh"
#include "tmEventSetup/esObject.hh"
#include "tmEventSetup/esCut.hh"
#include "tmEventSetup/esScale.hh"
#include "tmGrammar/Algorithm.hh"

#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"

#include "CondTools/L1TriggerExt/interface/L1ConfigOnlineProdBaseExt.h"
#include "CondFormats/L1TObjects/interface/L1TGlobalPrescalesVetosFract.h"
#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosFractRcd.h"
#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosFractO2ORcd.h"
#include "L1Trigger/L1TGlobal/interface/PrescalesVetosFractHelper.h"
#include "L1Trigger/L1TCommon/interface/TriggerSystem.h"
#include "L1Trigger/L1TCommon/interface/XmlConfigParser.h"
#include "OnlineDBqueryHelper.h"

class L1TGlobalPrescalesVetosOnlineProd
    : public L1ConfigOnlineProdBaseExt<L1TGlobalPrescalesVetosFractO2ORcd, L1TGlobalPrescalesVetosFract> {
private:
  bool transactionSafe;

public:
  std::unique_ptr<const L1TGlobalPrescalesVetosFract> newObject(
      const std::string &objectKey, const L1TGlobalPrescalesVetosFractO2ORcd &record) override;

  L1TGlobalPrescalesVetosOnlineProd(const edm::ParameterSet &);
  ~L1TGlobalPrescalesVetosOnlineProd(void) override {}
};

L1TGlobalPrescalesVetosOnlineProd::L1TGlobalPrescalesVetosOnlineProd(const edm::ParameterSet &iConfig)
    : L1ConfigOnlineProdBaseExt<L1TGlobalPrescalesVetosFractO2ORcd, L1TGlobalPrescalesVetosFract>(iConfig) {
  wrappedSetWhatProduced(iConfig);
  transactionSafe = iConfig.getParameter<bool>("transactionSafe");
}

std::unique_ptr<const L1TGlobalPrescalesVetosFract> L1TGlobalPrescalesVetosOnlineProd::newObject(
    const std::string &objectKey, const L1TGlobalPrescalesVetosFractO2ORcd &record) {
  edm::LogInfo("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
      << "Producing L1TGlobalPrescalesVetos with TSC:RS key = " << objectKey;

  if (objectKey.empty()) {
    if (transactionSafe)
      throw std::runtime_error("SummaryForFunctionManager: uGTrs | Faulty  | Empty objectKey");
    else {
      edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
          << "returning empty L1TGlobalPrescalesVetosFract object";
      return std::make_unique<const L1TGlobalPrescalesVetosFract>();
    }
  }

  unsigned int m_numberPhysTriggers = 512;

  // dictionary that maps algorithm name to it's index
  std::unordered_map<std::string, int, std::hash<std::string>> algoName2bit;

  std::string uGTtscKey = objectKey.substr(0, objectKey.find(':'));
  std::string uGTrsKey = objectKey.substr(objectKey.find(':') + 1, std::string::npos);

  std::string stage2Schema = "CMS_TRG_L1_CONF";

  std::string l1_menu_key;
  std::vector<std::string> queryStrings;
  queryStrings.push_back("L1_MENU");

  // select L1_MENU from CMS_TRG_L1_CONF.UGT_KEYS where ID = objectKey ;
  l1t::OMDSReader::QueryResults queryResult = m_omdsReader.basicQuery(
      queryStrings, stage2Schema, "UGT_KEYS", "UGT_KEYS.ID", m_omdsReader.singleAttribute(uGTtscKey));

  if (queryResult.queryFailed() || queryResult.numberRows() != 1) {
    edm::LogError("L1-O2O") << "Cannot get UGT_KEYS.L1_MENU for ID = " << uGTtscKey << " ";
    if (transactionSafe)
      throw std::runtime_error("SummaryForFunctionManager: uGTrs | Faulty  | Broken key");
    else {
      edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
          << "returning empty L1TGlobalPrescalesVetosFract object";
      return std::make_unique<const L1TGlobalPrescalesVetosFract>();
    }
  }

  if (!queryResult.fillVariable("L1_MENU", l1_menu_key))
    l1_menu_key = "";

  edm::LogInfo("L1-O2O: L1TGlobalPrescalesVetosOnlineProd") << "Producing L1TUtmTriggerMenu with key =" << l1_menu_key;

  if (uGTtscKey.empty()) {
    edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd") << "TSC key is empty, returning";
    if (transactionSafe)
      throw std::runtime_error("SummaryForFunctionManager: uGTrs | Faulty  | Empty objectKey");
    else {
      edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
          << "returning empty L1TGlobalPrescalesVetosFract object";
      return std::make_unique<const L1TGlobalPrescalesVetosFract>();
    }
  }

  std::vector<std::string> queryColumns;
  queryColumns.push_back("CONF");

  queryResult = m_omdsReader.basicQuery(
      queryColumns, stage2Schema, "UGT_L1_MENU", "UGT_L1_MENU.ID", m_omdsReader.singleAttribute(l1_menu_key));

  if (queryResult.queryFailed() || queryResult.numberRows() != 1) {
    edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
        << "Cannot get UGT_L1_MENU.CONF for ID = " << l1_menu_key;
    if (transactionSafe)
      throw std::runtime_error("SummaryForFunctionManager: uGTrs | Faulty  | Broken key");
    else {
      edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
          << "returning empty L1TGlobalPrescalesVetosFract object";
      return std::make_unique<const L1TGlobalPrescalesVetosFract>();
    }
  }

  std::string l1Menu;
  queryResult.fillVariable("CONF", l1Menu);
  ///
  std::istringstream iss(l1Menu);

  std::shared_ptr<L1TUtmTriggerMenu> pMenu(
      const_cast<L1TUtmTriggerMenu *>(reinterpret_cast<const L1TUtmTriggerMenu *>(tmeventsetup::getTriggerMenu(iss))));

  for (const auto &algo : pMenu->getAlgorithmMap())
    algoName2bit[algo.first] = algo.second.getIndex();

  ///std::vector< std::string > queryColumns;
  queryColumns.clear();
  queryColumns.push_back("ALGOBX_MASK");
  queryColumns.push_back("ALGO_FINOR_MASK");
  queryColumns.push_back("ALGO_FINOR_VETO");
  queryColumns.push_back("ALGO_PRESCALE");

  std::string prescale_key, bxmask_key, mask_key, vetomask_key;
  std::string xmlPayload_prescale, xmlPayload_mask_algobx, xmlPayload_mask_finor, xmlPayload_mask_veto;
  try {
    std::map<std::string, std::string> subKeys =
        l1t::OnlineDBqueryHelper::fetch(queryColumns, "UGT_RS_KEYS", uGTrsKey, m_omdsReader);
    prescale_key = subKeys["ALGO_PRESCALE"];
    bxmask_key = subKeys["ALGOBX_MASK"];
    mask_key = subKeys["ALGO_FINOR_MASK"];
    vetomask_key = subKeys["ALGO_FINOR_VETO"];
    xmlPayload_prescale = l1t::OnlineDBqueryHelper::fetch({"CONF"}, "UGT_RS_CLOBS", prescale_key, m_omdsReader)["CONF"];
    xmlPayload_mask_algobx =
        l1t::OnlineDBqueryHelper::fetch({"CONF"}, "UGT_RS_CLOBS", bxmask_key, m_omdsReader)["CONF"];
    xmlPayload_mask_finor = l1t::OnlineDBqueryHelper::fetch({"CONF"}, "UGT_RS_CLOBS", mask_key, m_omdsReader)["CONF"];
    xmlPayload_mask_veto =
        l1t::OnlineDBqueryHelper::fetch({"CONF"}, "UGT_RS_CLOBS", vetomask_key, m_omdsReader)["CONF"];
  } catch (std::runtime_error &e) {
    edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd") << e.what();
    if (transactionSafe)
      throw std::runtime_error(std::string("SummaryForFunctionManager: uGTrs | Faulty  | ") + e.what());
    else {
      edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
          << "returning empty L1TGlobalPrescalesVetosFract object";
      return std::make_unique<const L1TGlobalPrescalesVetosFract>();
    }
  }

  // for debugging purposes dump the payloads to /tmp
  std::ofstream output1(std::string("/tmp/").append(prescale_key.substr(0, prescale_key.find('/'))).append(".xml"));
  output1 << xmlPayload_prescale;
  output1.close();
  std::ofstream output2(std::string("/tmp/").append(mask_key.substr(0, mask_key.find('/'))).append(".xml"));
  output2 << xmlPayload_mask_finor;
  output2.close();
  std::ofstream output3(std::string("/tmp/").append(bxmask_key.substr(0, bxmask_key.find('/'))).append(".xml"));
  output3 << xmlPayload_mask_algobx;
  output3.close();
  std::ofstream output4(std::string("/tmp/").append(vetomask_key.substr(0, vetomask_key.find('/'))).append(".xml"));
  output4 << xmlPayload_mask_veto;
  output4.close();

  //////////////////

  std::vector<std::vector<double>> prescales;
  std::vector<unsigned int> triggerMasks;
  std::vector<int> triggerVetoMasks;
  std::map<int, std::vector<int>> triggerAlgoBxMaskAlgoTrig;

  // Prescales
  try {
    l1t::XmlConfigParser xmlReader_prescale;
    l1t::TriggerSystem ts_prescale;
    ts_prescale.addProcessor("uGtProcessor", "uGtProcessor", "-1", "-1");

    // run the parser
    xmlReader_prescale.readDOMFromString(xmlPayload_prescale);  // initialize it
    xmlReader_prescale.readRootElement(ts_prescale, "uGT");     // extract all of the relevant context
    ts_prescale.setConfigured();

    const std::map<std::string, l1t::Parameter> &settings_prescale = ts_prescale.getParameters("uGtProcessor");
    std::map<std::string, unsigned int> prescaleColumns = settings_prescale.at("prescales").getColumnIndices();

    unsigned int numColumns_prescale = prescaleColumns.size();
    int nPrescaleSets = numColumns_prescale - 1;
    std::vector<std::string> algoNames =
        settings_prescale.at("prescales").getTableColumn<std::string>("algo/prescale-index");

    if (nPrescaleSets > 0) {
      // Fill default prescale set
      for (int iSet = 0; iSet < nPrescaleSets; iSet++) {
        prescales.push_back(std::vector<double>());
        for (unsigned int iBit = 0; iBit < m_numberPhysTriggers; ++iBit) {
          double inputDefaultPrescale = 0;  // only prescales that are set in the block below are used
          prescales[iSet].push_back(inputDefaultPrescale);
        }
      }

      for (auto &col : prescaleColumns) {
        if (col.second < 1)
          continue;  // we don't care for the algorithms' indicies in 0th column
        int iSet = col.second - 1;
        std::vector<double> prescalesForSet =
            settings_prescale.at("prescales").getTableColumn<double>(col.first.c_str());
        for (unsigned int row = 0; row < prescalesForSet.size(); row++) {
          double prescale = prescalesForSet[row];
          std::string algoName = algoNames[row];
          unsigned int algoBit = algoName2bit[algoName];
          prescales[iSet][algoBit] = prescale;
        }
      }
    }
  } catch (std::runtime_error &e) {
    if (transactionSafe)
      throw std::runtime_error(std::string("SummaryForFunctionManager: uGTrs | Faulty  | ") + e.what());
    else {
      edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
          << "returning empty L1TGlobalPrescalesVetosFract object";
      return std::make_unique<const L1TGlobalPrescalesVetosFract>();
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // finor mask
  try {
    l1t::XmlConfigParser xmlReader_mask_finor;
    l1t::TriggerSystem ts_mask_finor;
    ts_mask_finor.addProcessor("uGtProcessor", "uGtProcessor", "-1", "-1");

    // run the parser
    xmlReader_mask_finor.readDOMFromString(xmlPayload_mask_finor);  // initialize it
    xmlReader_mask_finor.readRootElement(ts_mask_finor, "uGT");     // extract all of the relevant context
    ts_mask_finor.setConfigured();

    const std::map<std::string, l1t::Parameter> &settings_mask_finor = ts_mask_finor.getParameters("uGtProcessor");

    std::vector<std::string> algo_mask_finor = settings_mask_finor.at("finorMask").getTableColumn<std::string>("algo");
    std::vector<unsigned int> mask_mask_finor =
        settings_mask_finor.at("finorMask").getTableColumn<unsigned int>("mask");

    // mask (default=1 - unmask)
    unsigned int default_finor_mask = 1;
    auto default_finor_row = std::find_if(algo_mask_finor.cbegin(), algo_mask_finor.cend(), [](const std::string &s) {
      // simpler than overweight std::tolower(s[], std::locale()) POSIX solution (thx to BA):
      return strcasecmp("all", s.c_str()) == 0;
    });
    if (default_finor_row == algo_mask_finor.cend()) {
      edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
          << "\nWarning: No default found in FinOR mask xml, use 1 (unmasked) as default" << std::endl;
    } else {
      default_finor_mask = mask_mask_finor[std::distance(algo_mask_finor.cbegin(), default_finor_row)];
    }

    for (unsigned int iAlg = 0; iAlg < m_numberPhysTriggers; iAlg++)
      triggerMasks.push_back(default_finor_mask);

    for (unsigned int row = 0; row < algo_mask_finor.size(); row++) {
      std::string algoName = algo_mask_finor[row];
      if (strcasecmp("all", algoName.c_str()) == 0)
        continue;
      unsigned int algoBit = algoName2bit[algoName];
      unsigned int mask = mask_mask_finor[row];
      if (algoBit < m_numberPhysTriggers)
        triggerMasks[algoBit] = mask;
    }
  } catch (std::runtime_error &e) {
    if (transactionSafe)
      throw std::runtime_error(std::string("SummaryForFunctionManager: uGTrs | Faulty  | ") + e.what());
    else {
      edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
          << "returning empty L1TGlobalPrescalesVetosFract object";
      return std::make_unique<const L1TGlobalPrescalesVetosFract>();
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // veto mask
  try {
    l1t::XmlConfigParser xmlReader_mask_veto;
    l1t::TriggerSystem ts_mask_veto;
    ts_mask_veto.addProcessor("uGtProcessor", "uGtProcessor", "-1", "-1");

    // run the parser
    xmlReader_mask_veto.readDOMFromString(xmlPayload_mask_veto);  // initialize it
    xmlReader_mask_veto.readRootElement(ts_mask_veto, "uGT");     // extract all of the relevant context
    ts_mask_veto.setConfigured();

    const std::map<std::string, l1t::Parameter> &settings_mask_veto = ts_mask_veto.getParameters("uGtProcessor");
    std::vector<std::string> algo_mask_veto = settings_mask_veto.at("vetoMask").getTableColumn<std::string>("algo");
    std::vector<unsigned int> veto_mask_veto = settings_mask_veto.at("vetoMask").getTableColumn<unsigned int>("veto");

    // veto mask (default=0 - no veto)
    unsigned int default_veto_mask = 1;
    auto default_veto_row = std::find_if(algo_mask_veto.cbegin(), algo_mask_veto.cend(), [](const std::string &s) {
      // simpler than overweight std::tolower(s[], std::locale()) POSIX solution (thx to BA):
      return strcasecmp("all", s.c_str()) == 0;
    });
    if (default_veto_row == algo_mask_veto.cend()) {
      edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
          << "\nWarning: No default found in Veto mask xml, use 0 (unvetoed) as default" << std::endl;
    } else {
      default_veto_mask = veto_mask_veto[std::distance(algo_mask_veto.cbegin(), default_veto_row)];
    }

    for (unsigned int iAlg = 0; iAlg < m_numberPhysTriggers; iAlg++)
      triggerVetoMasks.push_back(default_veto_mask);

    for (unsigned int row = 0; row < algo_mask_veto.size(); row++) {
      std::string algoName = algo_mask_veto[row];
      if (strcasecmp("all", algoName.c_str()) == 0)
        continue;
      unsigned int algoBit = algoName2bit[algoName];
      unsigned int veto = veto_mask_veto[row];
      if (algoBit < m_numberPhysTriggers)
        triggerVetoMasks[algoBit] = int(veto);
    }
  } catch (std::runtime_error &e) {
    if (transactionSafe)
      throw std::runtime_error(std::string("SummaryForFunctionManager: uGTrs | Faulty  | ") + e.what());
    else {
      edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
          << "returning empty L1TGlobalPrescalesVetosFract object";
      return std::make_unique<const L1TGlobalPrescalesVetosFract>();
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Algo bx mask
  unsigned int m_bx_mask_default = 1;

  std::vector<std::string> bx_algo_name;
  std::vector<std::string> bx_range;
  std::vector<unsigned int> bx_mask;

  try {
    l1t::XmlConfigParser xmlReader_mask_algobx;
    l1t::TriggerSystem ts_mask_algobx;
    ts_mask_algobx.addProcessor("uGtProcessor", "uGtProcessor", "-1", "-1");

    // run the parser
    xmlReader_mask_algobx.readDOMFromString(xmlPayload_mask_algobx);  // initialize it
    xmlReader_mask_algobx.readRootElement(ts_mask_algobx, "uGT");     // extract all of the relevant context
    ts_mask_algobx.setConfigured();

    const std::map<std::string, l1t::Parameter> &settings_mask_algobx = ts_mask_algobx.getParameters("uGtProcessor");
    bx_algo_name = settings_mask_algobx.at("algorithmBxMask").getTableColumn<std::string>("algo");
    bx_range = settings_mask_algobx.at("algorithmBxMask").getTableColumn<std::string>("range");
    bx_mask = settings_mask_algobx.at("algorithmBxMask").getTableColumn<unsigned int>("mask");
  } catch (std::runtime_error &e) {
    if (transactionSafe)
      throw std::runtime_error(std::string("SummaryForFunctionManager: uGTrs | Faulty  | ") + e.what());
    else {
      edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
          << "returning empty L1TGlobalPrescalesVetosFract object";
      return std::make_unique<const L1TGlobalPrescalesVetosFract>();
    }
  }

  int default_bxmask_row = -1;
  typedef std::pair<unsigned long, unsigned long> Range_t;
  // auto comp = [] (Range_t a, Range_t b){ return a.first < b.first; };
  struct RangeComp_t {
    bool operator()(const Range_t &a, const Range_t &b) const { return a.first < b.first; }
  };
  std::map<std::string, std::set<Range_t, RangeComp_t>> non_default_bx_ranges;

  for (unsigned int row = 0; row < bx_algo_name.size(); row++) {
    const std::string &s1 = bx_algo_name[row];
    const std::string &s2 = bx_range[row];
    // find "all" broadcast keywords
    bool broadcastAlgo = false;
    bool broadcastRange = false;
    if (strcasecmp("all", s1.c_str()) == 0)
      broadcastAlgo = true;
    if (strcasecmp("all", s2.c_str()) == 0)
      broadcastRange = true;
    // ALL-ALL-default:
    if (broadcastAlgo && broadcastRange) {
      if (row != 0) {
        edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
            << "\nWarning: ALL-ALL row is not the first one, ignore it assuming 1 (unmasked) as the default"
            << std::endl;
        continue;
      }
      if (default_bxmask_row >= 0) {
        edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
            << "\nWarning: multiple ALL-ALL rows found, using the first" << std::endl;
        continue;
      }
      default_bxmask_row = row;
      m_bx_mask_default = bx_mask[row];
      continue;
    }
    // interpret the range
    unsigned long first = 0, last = 0;
    if (broadcastRange) {
      first = 0;
      last = 3563;
    } else {
      char *dash = nullptr;
      first = strtoul(s2.data(), &dash, 0);
      while (*dash != '\0' && *dash != '-')
        ++dash;
      last = (*dash != '\0' ? strtoul(++dash, &dash, 0) : first);
      if (first == 3564)
        first = 0;
      if (last == 3564)
        last = 0;
      // what could possibly go wrong?
      if (*dash != '\0') {
        edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
            << "\nWarning: parsing " << s2 << " as [" << first << "," << last << "] range" << std::endl;
      }
      if (first > 3563) {
        edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
            << "\nWarning: start of interval is out of range: " << s2 << ", skipping the row" << std::endl;
        continue;
      }
      if (last > 3563) {
        last = 3563;
        edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
            << "\nWarning: end of interval is out of range: " << s2 << ", force [" << first << "," << last << "] range"
            << std::endl;
      }
      if (first > last) {
        edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
            << "\nWarning: inverse/spillover range " << s2 << ", accounting for circular orbit as [0," << last
            << "] & [" << first << ",3563]" << std::endl;
      }
    }
    // {algo,ALL}-{range,ALL}-{0,1}:
    std::vector<std::string> algos;
    std::vector<std::pair<unsigned long, unsigned long>> orderedRanges;
    if (first <= last) {
      if (!broadcastAlgo) {
        algos.push_back(bx_algo_name[row]);
        orderedRanges.push_back(std::make_pair(first, last));
      } else {
        for (const auto &i : non_default_bx_ranges) {
          algos.push_back(i.first);
          orderedRanges.push_back(std::make_pair(first, last));
        }
      }
    } else {
      if (!broadcastAlgo) {
        algos.push_back(bx_algo_name[row]);
        algos.push_back(bx_algo_name[row]);
        orderedRanges.push_back(std::make_pair(0, last));
        orderedRanges.push_back(std::make_pair(first, 3563));
      } else {
        for (const auto &i : non_default_bx_ranges) {
          algos.push_back(i.first);
          algos.push_back(i.first);
          orderedRanges.push_back(std::make_pair(0, last));
          orderedRanges.push_back(std::make_pair(first, 3563));
        }
      }
    }

    for (unsigned int item = 0; item < algos.size(); item++) {
      const std::string &algoName = algos[item];
      unsigned int first = orderedRanges[item].first;
      unsigned int last = orderedRanges[item].second;

      std::set<Range_t, RangeComp_t> &ranges = non_default_bx_ranges[algoName];
      //           .insert
      //           (
      //               std::pair< std::string, std::set<Range_t,RangeComp_t> >
      //               (
      //                   algoName,  std::set<Range_t,RangeComp_t>()
      //               )
      //           ).first->second; // I don't care if insert was successfull or if I've got a hold on existing range

      // current range may or may not overlap with the already present ranges
      // if end of the predecessor starts before begin of the current range and begin
      //  of the successor starts after end of the current range there is no overlap
      //  and I save this range only if it has mask different from the default
      //  otherwise modify predecessor/successor ranges accordingly
      std::set<Range_t>::iterator curr = ranges.end();  // inserted range
      std::set<Range_t>::iterator succ =
          ranges.lower_bound(std::make_pair(first, last));  // successor starts at current or later
      std::set<Range_t>::iterator pred = succ;
      if (pred != ranges.begin())
        pred--;
      else
        pred = ranges.end();

      if ((pred == ranges.end() || pred->second < first) && (succ == ranges.end() || succ->first > last)) {
        // no overlap
        if (m_bx_mask_default != bx_mask[row])
          curr = ranges.insert(std::make_pair(first, last)).first;
        // do nothing if this is a default-mask interval
      } else {
        // pred/succ iterators are read-only, create intermediate adjusted copies
        Range_t newPred, newSucc;
        bool modifiedPred = false, gapInPred = false, modifiedSucc = false, dropSucc = false;
        // overlap found with predecessor range
        if (pred != ranges.end() && pred->second >= first && pred->second <= last) {
          if (m_bx_mask_default != bx_mask[row]) {
            if (last == pred->second) {
              // both ranges end in the same place - nothing to do
              modifiedPred = false;
            } else {
              // extend predecessor range
              newPred.first = pred->first;
              newPred.second = last;
              modifiedPred = true;
            }
          } else {
            // shrink predecessor range
            newPred.first = pred->first;
            newPred.second = first - 1;  // non-negative for the predecessor by design
            // by design pred->first < first, so the interval above is always valid
            modifiedPred = true;
          }
        }
        // current range is fully contained in predecessor
        if (pred != ranges.end() && pred->second > first && pred->second > last) {
          if (m_bx_mask_default != bx_mask[row]) {
            // no change to the predecessor range
            modifiedPred = false;
          } else {
            // make a "gap" in predecessor range
            newPred.first = first;
            newPred.second = last;
            gapInPred = true;
            modifiedPred = true;
          }
        }
        // overlap found with successor range
        if (succ != ranges.end() && succ->first <= last) {
          if (m_bx_mask_default != bx_mask[row]) {
            // extend successor range
            newSucc.first = first;
            newSucc.second = succ->second;
          } else {
            // shrink successor range
            newSucc.first = last + 1;
            newSucc.second = succ->second;
            if (newSucc.first > 3563 || newSucc.first > newSucc.second)
              dropSucc = true;
          }
          modifiedSucc = true;
        }
        // overlap found with both, predecessor and successor, such that I need to merge them
        if (modifiedPred && modifiedSucc && newPred.second >= newSucc.first) {
          // make newPred and newSucc identical just in case
          newPred.second = newSucc.second;
          newSucc.first = newPred.first;
          ranges.erase(pred, ++succ);
          curr = ranges.insert(newPred).first;
        } else {
          // merging is not the case, but I still need to propagate the new ranges back to the source
          if (modifiedPred) {
            if (!gapInPred) {
              ranges.erase(pred);
              curr = ranges.insert(newPred).first;
            } else {
              // make a gap by splitting predecessor into two ranges
              Range_t r1(pred->first, newPred.first - 1);  // non-negative for the predecessor by design
              Range_t r2(newPred.second + 1, pred->second);
              ranges.erase(pred);
              ranges.insert(r1);
              ranges.insert(r2);
              curr = ranges.end();  // gap cannot cover any additional ranges
            }
          }
          if (modifiedSucc) {
            ranges.erase(succ);
            if (!dropSucc)
              curr = ranges.insert(newSucc).first;
          }
        }
      }
      // if current range spans over few more ranges after the successor
      //  remove those from the consideration up until the last covered range
      //  that may or may not extend beyond the current range end
      if (curr != ranges.end()) {  // insertion took place
        std::set<Range_t, RangeComp_t>::iterator last_covered = ranges.upper_bound(std::make_pair(curr->second, 0));
        if (last_covered != ranges.begin())
          last_covered--;
        else
          last_covered = ranges.end();

        if (last_covered != ranges.end() && last_covered->first != curr->first) {
          // ranges is not empty and last_covered is not current itself (i.e. it is different)
          if (curr->second < last_covered->second) {
            // the range needs to be extended
            Range_t newRange(curr->first, last_covered->second);
            ranges.erase(curr);
            curr = ranges.insert(newRange).first;
          }
          ranges.erase(++curr, last_covered);
        }
      }
    }
  }

  if (default_bxmask_row < 0) {
    edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
        << "\nWarning: No default found in BX mask xml, used 1 (unmasked) as default" << std::endl;
  }

  for (auto &algo : non_default_bx_ranges) {
    const std::string &algoName = algo.first;
    unsigned int algoBit = algoName2bit[algoName];
    for (auto range : algo.second)
      for (unsigned int bx = range.first; bx <= range.second; bx++) {
        triggerAlgoBxMaskAlgoTrig[bx].push_back(algoBit);
      }
  }

  // Set prescales to zero if masked
  for (unsigned int iSet = 0; iSet < prescales.size(); iSet++) {
    for (unsigned int iBit = 0; iBit < prescales[iSet].size(); iBit++) {
      // Add protection in case prescale table larger than trigger mask size
      if (iBit >= triggerMasks.size()) {
        edm::LogError("L1-O2O: L1TGlobalPrescalesVetosOnlineProd")
            << "\nWarning: algoBit in prescale table >= triggerMasks.size() "
            << "\nWarning: no information on masking bit or not, setting as unmasked " << std::endl;
      } else {
        prescales[iSet][iBit] *= triggerMasks[iBit];
      }
    }
  }

  /////////////

  l1t::PrescalesVetosFractHelper data_(new L1TGlobalPrescalesVetosFract());

  data_.setBxMaskDefault(m_bx_mask_default);
  data_.setPrescaleFactorTable(prescales);
  data_.setTriggerMaskVeto(triggerVetoMasks);
  data_.setTriggerAlgoBxMask(triggerAlgoBxMaskAlgoTrig);

  auto payload = std::make_unique<const L1TGlobalPrescalesVetosFract>(*data_.getWriteInstance());

  edm::LogInfo("L1-O2O: L1TCaloParamsOnlineProd") << "SummaryForFunctionManager: uGTrs | OK      | All looks good";

  return payload;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TGlobalPrescalesVetosOnlineProd);
