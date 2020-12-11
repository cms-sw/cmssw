// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      CommonModeAnalyzer
//
/**\class CommonModeAnalyzer CommonModeAnalyzer.cc DPGAnalysis/SiStripTools/plugins/CommonModeAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Tue Jul 19 11:56:00 CEST 2009
//
//

// system include files
#include <memory>

// user include files
#include "TH1D.h"
#include "TProfile.h"
#include <vector>
#include <algorithm>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"
#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"

#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"
#include "CommonTools/UtilAlgos/interface/DetIdSelector.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
//
// class decleration
//

class CommonModeAnalyzer : public edm::EDAnalyzer {
public:
  explicit CommonModeAnalyzer(const edm::ParameterSet&);
  ~CommonModeAnalyzer() override;

private:
  void beginJob() override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void updateDetCabling(const SiStripDetCablingRcd& iRcd);

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi> > m_digicollectionToken;
  edm::EDGetTokenT<EventWithHistory> m_historyProductToken;
  edm::EDGetTokenT<APVCyclePhaseCollection> m_apvphasecollToken;
  edm::EDGetTokenT<DetIdCollection> m_digibadmodulecollectionToken;
  const std::string m_phasepart;
  const bool m_ignorebadfedmod;
  const bool m_ignorenotconnected;
  int m_nevents;

  std::vector<DetIdSelector> m_selections;
  std::vector<std::string> m_labels;
  std::vector<TH1D*> m_cmdist;
  std::vector<TH1D*> m_nmodules;
  std::vector<TH1D*> m_napvs;
  std::vector<TProfile*> m_cmvsdbxincycle;
  std::vector<TProfile**> m_cmvsbxrun;
  std::vector<TProfile**> m_cmvsorbitrun;

  RunHistogramManager m_rhm;

  edm::ESWatcher<SiStripDetCablingRcd> m_detCablingWatcher;
  edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> m_detCablingToken;
  const SiStripDetCabling* m_detCabling = nullptr;  //!< The cabling object.
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CommonModeAnalyzer::CommonModeAnalyzer(const edm::ParameterSet& iConfig)
    : m_digicollectionToken(
          consumes<edm::DetSetVector<SiStripRawDigi> >(iConfig.getParameter<edm::InputTag>("digiCollection"))),
      m_historyProductToken(consumes<EventWithHistory>(iConfig.getParameter<edm::InputTag>("historyProduct"))),
      m_apvphasecollToken(consumes<APVCyclePhaseCollection>(iConfig.getParameter<edm::InputTag>("apvPhaseCollection"))),
      m_digibadmodulecollectionToken(
          consumes<DetIdCollection>(iConfig.getParameter<edm::InputTag>("badModuleDigiCollection"))),
      m_phasepart(iConfig.getUntrackedParameter<std::string>("phasePartition", "None")),
      m_ignorebadfedmod(iConfig.getParameter<bool>("ignoreBadFEDMod")),
      m_ignorenotconnected(iConfig.getParameter<bool>("ignoreNotConnected")),
      m_nevents(0),
      m_selections(),
      m_labels(),
      m_cmdist(),
      m_nmodules(),
      m_napvs(),
      m_cmvsdbxincycle(),
      m_cmvsbxrun(),
      m_cmvsorbitrun(),
      m_rhm(consumesCollector()),
      m_detCablingWatcher(this, &CommonModeAnalyzer::updateDetCabling),
      m_detCablingToken(esConsumes()) {
  //now do what ever initialization is needed

  edm::Service<TFileService> tfserv;

  std::vector<edm::ParameterSet> selconfigs = iConfig.getParameter<std::vector<edm::ParameterSet> >("selections");

  for (std::vector<edm::ParameterSet>::const_iterator selconfig = selconfigs.begin(); selconfig != selconfigs.end();
       ++selconfig) {
    std::string label = selconfig->getParameter<std::string>("label");
    DetIdSelector selection(*selconfig);
    m_selections.push_back(selection);

    {
      std::string hname = label + "_CommonMode";
      std::string htitle = label + " Common Mode";
      m_cmdist.push_back(tfserv->make<TH1D>(hname.c_str(), htitle.c_str(), 1024, -0.5, 1024 - 0.5));
      m_cmdist.back()->GetXaxis()->SetTitle("ADC");
    }
    {
      std::string hname = label + "_nmodules";
      std::string htitle = label + " Number of Modules with CM value";
      m_nmodules.push_back(tfserv->make<TH1D>(hname.c_str(), htitle.c_str(), 20000, -0.5, 20000 - 0.5));
      m_nmodules.back()->GetXaxis()->SetTitle("#modules");
    }
    {
      std::string hname = label + "_napvs";
      std::string htitle = label + " Number of APVs with CM value";
      m_napvs.push_back(tfserv->make<TH1D>(hname.c_str(), htitle.c_str(), 2000, -0.5, 80000 - 0.5));
      m_napvs.back()->GetXaxis()->SetTitle("#apvs");
    }
    {
      std::string hname = label + "_CMvsDBXinCycle";
      std::string htitle = label + " Common Mode vs DBX in Cycle";
      m_cmvsdbxincycle.push_back(tfserv->make<TProfile>(hname.c_str(), htitle.c_str(), 1000, -0.5, 1000 - 0.5));
      m_cmvsdbxincycle.back()->GetXaxis()->SetTitle("DBX in cycle");
      m_cmvsdbxincycle.back()->GetYaxis()->SetTitle("CM (ADC counts)");
    }
    {
      std::string hname = label + "_CMvsBX";
      std::string htitle = label + " Common Mode vs BX";
      m_cmvsbxrun.push_back(m_rhm.makeTProfile(hname.c_str(), htitle.c_str(), 3565, -0.5, 3565 - 0.5));
    }
    {
      std::string hname = label + "_CMvsOrbit";
      std::string htitle = label + " Common Mode vs Orbit";
      m_cmvsorbitrun.push_back(m_rhm.makeTProfile(hname.c_str(), htitle.c_str(), 4 * 500, 0, 500 * 262144));
    }
  }
}

CommonModeAnalyzer::~CommonModeAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void CommonModeAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  m_detCablingWatcher.check(iSetup);

  m_nevents++;

  edm::Handle<EventWithHistory> he;
  iEvent.getByToken(m_historyProductToken, he);

  edm::Handle<APVCyclePhaseCollection> apvphase;
  iEvent.getByToken(m_apvphasecollToken, apvphase);

  Handle<DetIdCollection> badmodules;
  iEvent.getByToken(m_digibadmodulecollectionToken, badmodules);

  int thephase = APVCyclePhaseCollection::invalid;
  if (apvphase.isValid() && !apvphase.failedToGet()) {
    thephase = apvphase->getPhase(m_phasepart);
  }
  bool isphaseok = (thephase != APVCyclePhaseCollection::invalid && thephase != APVCyclePhaseCollection::multiphase &&
                    thephase != APVCyclePhaseCollection::nopartition);

  Handle<edm::DetSetVector<SiStripRawDigi> > digis;
  iEvent.getByToken(m_digicollectionToken, digis);

  // loop on detector with digis

  std::vector<int> nmodules(m_selections.size(), 0);
  std::vector<int> napvs(m_selections.size(), 0);

  for (edm::DetSetVector<SiStripRawDigi>::const_iterator mod = digis->begin(); mod != digis->end(); mod++) {
    std::vector<const FedChannelConnection*> conns = m_detCabling->getConnections(mod->detId());

    if (!m_ignorebadfedmod || std::find(badmodules->begin(), badmodules->end(), mod->detId()) == badmodules->end()) {
      for (unsigned int isel = 0; isel < m_selections.size(); ++isel) {
        if (m_selections[isel].isSelected(mod->detId())) {
          unsigned int strip = 0;
          ++nmodules[isel];
          for (edm::DetSet<SiStripRawDigi>::const_iterator digi = mod->begin(); digi != mod->end(); digi++, strip++) {
            LogDebug("StripNumber") << "Strip number " << strip;
            if (!m_ignorenotconnected ||
                ((conns.size() > strip / 2) && conns[strip / 2] && conns[strip / 2]->isConnected())) {
              ++napvs[isel];
              m_cmdist[isel]->Fill(digi->adc());
              if (isphaseok)
                m_cmvsdbxincycle[isel]->Fill(he->deltaBXinCycle(thephase), digi->adc());
              if (m_cmvsbxrun[isel] && *(m_cmvsbxrun[isel]))
                (*(m_cmvsbxrun[isel]))->Fill(iEvent.bunchCrossing(), digi->adc());
              if (m_cmvsorbitrun[isel] && *(m_cmvsorbitrun[isel]))
                (*(m_cmvsorbitrun[isel]))->Fill(iEvent.orbitNumber(), digi->adc());
            } else if (digi->adc() > 0) {
              edm::LogWarning("NonZeroCMWhenDisconnected")
                  << " Non zero CM in " << mod->detId() << " APV " << strip << " with " << conns.size()
                  << " connections and connection pointer" << conns[strip / 2];
            }
          }
        }
      }
    }
  }
  for (unsigned int isel = 0; isel < m_selections.size(); ++isel) {
    m_nmodules[isel]->Fill(nmodules[isel]);
    m_napvs[isel]->Fill(napvs[isel]);
  }
}

void CommonModeAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup&) {
  m_rhm.beginRun(iRun);

  for (std::vector<TProfile**>::const_iterator cmvsbx = m_cmvsbxrun.begin(); cmvsbx != m_cmvsbxrun.end(); ++cmvsbx) {
    if (*cmvsbx && *(*cmvsbx)) {
      (*(*cmvsbx))->GetXaxis()->SetTitle("BX");
      (*(*cmvsbx))->GetYaxis()->SetTitle("CM (ADC counts)");
    }
  }
  for (std::vector<TProfile**>::const_iterator cmvsorbit = m_cmvsorbitrun.begin(); cmvsorbit != m_cmvsorbitrun.end();
       ++cmvsorbit) {
    if (*cmvsorbit && *(*cmvsorbit)) {
      (*(*cmvsorbit))->GetXaxis()->SetTitle("orbit");
      (*(*cmvsorbit))->GetYaxis()->SetTitle("CM (ADC counts)");
      (*(*cmvsorbit))->SetCanExtend(TH1::kXaxis);
    }
  }
}

void CommonModeAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup&) {}

// ------------ method called once each job just before starting event loop  ------------
void CommonModeAnalyzer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void CommonModeAnalyzer::endJob() { edm::LogInfo("EndOfJob") << m_nevents << " analyzed events"; }

void CommonModeAnalyzer::updateDetCabling(const SiStripDetCablingRcd& iRcd) {
  m_detCabling = &iRcd.get(m_detCablingToken);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CommonModeAnalyzer);
