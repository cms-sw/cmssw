#include "DataFormats/Common/interface/Handle.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

#include "HtrXmlPatternTool.h"
#include "HtrXmlPatternToolParameters.h"

// system include files
#include <memory>

// default include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

class HtrXmlPattern : public edm::one::EDAnalyzer<> {
public:
  explicit HtrXmlPattern(const edm::ParameterSet&);
  ~HtrXmlPattern() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  virtual void do_hand_fill(const HcalElectronicsMap*);
  HtrXmlPatternTool* m_tool;
  HtrXmlPatternToolParameters* m_toolparameters;
  int m_sets_to_show;
  int m_hand_pattern_number;
  bool m_fill_by_hand;
  bool m_filled;
  bool m_write_root_file;

  const edm::ESGetToken<HcalDbService, HcalDbRecord> m_hcalElectronicsMapToken;
};

HtrXmlPattern::HtrXmlPattern(const edm::ParameterSet& iConfig) {
  m_filled = false;
  m_fill_by_hand = iConfig.getUntrackedParameter<bool>("fill_by_hand");
  m_hand_pattern_number = iConfig.getUntrackedParameter<int>("hand_pattern_number");
  m_sets_to_show = iConfig.getUntrackedParameter<int>("sets_to_show");
  m_write_root_file = iConfig.getUntrackedParameter<bool>("write_root_file");

  m_toolparameters = new HtrXmlPatternToolParameters;
  m_toolparameters->m_show_errors = iConfig.getUntrackedParameter<bool>("show_errors");
  m_toolparameters->m_presamples_per_event = iConfig.getUntrackedParameter<int>("presamples_per_event");
  m_toolparameters->m_samples_per_event = iConfig.getUntrackedParameter<int>("samples_per_event");

  m_toolparameters->m_XML_file_mode = iConfig.getUntrackedParameter<int>("XML_file_mode");
  m_toolparameters->m_file_tag = iConfig.getUntrackedParameter<std::string>("file_tag");
  m_toolparameters->m_user_output_directory = iConfig.getUntrackedParameter<std::string>("user_output_directory");

  std::string out_dir = m_toolparameters->m_user_output_directory;
  while (out_dir.find_last_of('/') == out_dir.length() - 1)
    out_dir.erase(out_dir.find_last_of('/'));
  m_toolparameters->m_output_directory = out_dir + "/" + (m_toolparameters->m_file_tag) + "/";

  m_tool = new HtrXmlPatternTool(m_toolparameters);
}

HtrXmlPattern::~HtrXmlPattern() {
  delete m_tool;
  delete m_toolparameters;
}

// ------------ method called to for each event  ------------
void HtrXmlPattern::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace std;

  if (m_filled)
    return;

  // Get the HCAL Electronics Map from the Event setup
  const auto pSetup = iSetup.getHandle(m_hcalElectronicsMapToken);
  const HcalElectronicsMap* readoutMap = pSetup->getHcalMapping();

  if (m_fill_by_hand) {
    do_hand_fill(readoutMap);
    m_filled = true;
    return;
  }

  std::vector<edm::Handle<HBHEDigiCollection> > hbhe;
  std::vector<edm::Handle<HODigiCollection> > ho;
  std::vector<edm::Handle<HFDigiCollection> > hf;
  std::vector<edm::Handle<ZDCDigiCollection> > zdc;
  std::vector<edm::Handle<HcalCalibDigiCollection> > hc;
  std::vector<edm::Handle<HcalTrigPrimDigiCollection> > htp;
  std::vector<edm::Handle<HcalHistogramDigiCollection> > hh;

  //iEvent.getManyByType(hbhe);
  throw cms::Exception("UnsupportedFunction") << "HtrXmlPattern::analyze: "
                                              << "getManyByType has not been supported by the Framework since 2015. "
                                              << "This module has been broken since then. Maybe it should be deleted. "
                                              << "Another possibility is to upgrade to use GetterOfProducts instead.";

  if (hbhe.empty()) {
    edm::LogPrint("HtrXmlPattern") << "No HB/HE Digis.";
  } else {
    std::vector<edm::Handle<HBHEDigiCollection> >::iterator i;
    for (i = hbhe.begin(); i != hbhe.end(); i++) {
      const HBHEDigiCollection& c = *(*i);

      int count = 0;
      for (HBHEDigiCollection::const_iterator j = c.begin(); j != c.end(); j++) {
        const HcalElectronicsId HEID = readoutMap->lookup(j->id());
        m_tool->Fill(HEID, j);

        if (count++ < m_sets_to_show || m_sets_to_show < 0) {
          edm::LogPrint("HtrXmlPattern") << *j;
          edm::LogPrint("HtrXmlPattern") << HEID;
          edm::LogPrint("HtrXmlPattern") << "count: " << count;
        }
      }
      if (m_sets_to_show != 0)
        edm::LogPrint("HtrXmlPattern") << "HB/HE count: " << count;
    }
  }

  //iEvent.getManyByType(hf);
  throw cms::Exception("UnsupportedFunction") << "HtrXmlPattern::analyze: "
                                              << "getManyByType has not been supported by the Framework since 2015. "
                                              << "This module has been broken since then. Maybe it should be deleted. "
                                              << "Another possibility is to upgrade to use GetterOfProducts instead.";

  if (hf.empty()) {
    edm::LogPrint("HtrXmlPattern") << "No HF Digis.";
  } else {
    std::vector<edm::Handle<HFDigiCollection> >::iterator i;
    for (i = hf.begin(); i != hf.end(); i++) {
      const HFDigiCollection& c = *(*i);

      int count = 0;
      for (HFDigiCollection::const_iterator j = c.begin(); j != c.end(); j++) {
        const HcalElectronicsId HEID = readoutMap->lookup(j->id());
        m_tool->Fill(HEID, j);

        if (count++ < m_sets_to_show || m_sets_to_show < 0) {
          edm::LogPrint("HtrXmlPattern") << *j;
          edm::LogPrint("HtrXmlPattern") << HEID;
          edm::LogPrint("HtrXmlPattern") << "count: " << count;
        }
      }
      if (m_sets_to_show != 0)
        edm::LogPrint("HtrXmlPattern") << "HF    count: " << count;
    }
  }

  //iEvent.getManyByType(ho);
  throw cms::Exception("UnsupportedFunction") << "HtrXmlPattern::analyze: "
                                              << "getManyByType has not been supported by the Framework since 2015. "
                                              << "This module has been broken since then. Maybe it should be deleted. "
                                              << "Another possibility is to upgrade to use GetterOfProducts instead.";

  if (ho.empty()) {
    edm::LogPrint("HtrXmlPattern") << "No HO Digis.";
  } else {
    std::vector<edm::Handle<HODigiCollection> >::iterator i;
    for (i = ho.begin(); i != ho.end(); i++) {
      const HODigiCollection& c = *(*i);

      int count = 0;
      for (HODigiCollection::const_iterator j = c.begin(); j != c.end(); j++) {
        const HcalElectronicsId HEID = readoutMap->lookup(j->id());
        m_tool->Fill(HEID, j);

        if (count++ < m_sets_to_show || m_sets_to_show < 0) {
          edm::LogPrint("HtrXmlPattern") << *j;
          edm::LogPrint("HtrXmlPattern") << HEID;
          edm::LogPrint("HtrXmlPattern") << "count: " << count;
        }
      }
      if (m_sets_to_show != 0)
        edm::LogPrint("HtrXmlPattern") << "HO    count: " << count;
    }
  }

  edm::LogPrint("HtrXmlPattern");
}

void HtrXmlPattern::do_hand_fill(const HcalElectronicsMap* emap) {
  HtrXmlPatternSet* hxps = m_tool->GetPatternSet();

  for (int iCrate = 0; iCrate < ChannelPattern::NUM_CRATES; iCrate++) {
    CrateData* cd = hxps->getCrate(iCrate);
    if (!cd)
      continue;
    for (int iSlot = 0; iSlot < ChannelPattern::NUM_SLOTS; iSlot++) {
      for (int iTb = 0; iTb < 2; iTb++) {
        HalfHtrData* hhd = cd->getHalfHtrData(iSlot, iTb);
        if (!hhd)
          continue;
        for (int iChannel = 1; iChannel < 25; iChannel++) {
          ChannelPattern* cp = hhd->getPattern(iChannel);
          if (!cp)
            continue;
          cp->Fill_by_hand(emap, m_hand_pattern_number);
        }
      }
    }
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void HtrXmlPattern::endJob() {
  bool modeg0 = m_toolparameters->m_XML_file_mode > 0;
  if (modeg0 || m_write_root_file)
    m_tool->prepareDirs();
  if (modeg0)
    m_tool->writeXML();
  if (m_write_root_file)
    m_tool->createHists();
}

DEFINE_FWK_MODULE(HtrXmlPattern);
