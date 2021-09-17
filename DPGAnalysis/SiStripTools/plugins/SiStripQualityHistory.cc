// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      SiStripQualityHistory
//
/**\class SiStripQualityHistory SiStripQualityHistory.cc DPGAnalysis/SiStripTools/plugins/SiStripQualityHistory.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Tue Sep 18 17:52:00 CEST 2009
//
//

// system include files
#include <memory>

// user include files

#include <vector>
#include <map>

//#include "TGraph.h"
#include "TH1F.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"
//
// class decleration
//

class SiStripQualityHistory : public edm::EDAnalyzer {
public:
  explicit SiStripQualityHistory(const edm::ParameterSet&);
  ~SiStripQualityHistory() override;

  enum { Module, Fiber, APV, Strip };

private:
  void beginJob() override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------

  RunHistogramManager m_rhm;
  const std::vector<edm::ParameterSet> _monitoredssq;
  std::vector<edm::ESGetToken<SiStripQuality, SiStripQualityRcd>> _ssqTokens;
  const unsigned int _mode;
  const bool m_run;
  const unsigned int m_maxLS;
  const unsigned int m_LSfrac;
  //  std::map<std::string,TGraph*> _history;
  std::map<std::string, TH1F*> _history;
  std::map<std::string, TProfile**> m_badmodrun;
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
SiStripQualityHistory::SiStripQualityHistory(const edm::ParameterSet& iConfig)
    : m_rhm(consumesCollector()),
      _monitoredssq(iConfig.getParameter<std::vector<edm::ParameterSet>>("monitoredSiStripQuality")),
      _mode(iConfig.getUntrackedParameter<unsigned int>("granularityMode", Module)),
      m_run(iConfig.getParameter<bool>("runProcess")),
      m_maxLS(iConfig.getUntrackedParameter<unsigned int>("maxLSBeforeRebin", 100)),
      m_LSfrac(iConfig.getUntrackedParameter<unsigned int>("startingLSFraction", 4)),
      _history(),
      m_badmodrun() {
  //now do what ever initialization is needed

  edm::Service<TFileService> tfserv;

  for (const auto& ps : _monitoredssq) {
    _ssqTokens.emplace_back(
        esConsumes<edm::Transition::BeginRun>(edm::ESInputTag{"", ps.getParameter<std::string>("ssqLabel")}));
    std::string name = ps.getParameter<std::string>("name");
    //    _history[name] = tfserv->make<TGraph>();
    //    _history[name]->SetName(name.c_str());     _history[name]->SetTitle(name.c_str());

    if (m_run)
      _history[name] = tfserv->make<TH1F>(name.c_str(), name.c_str(), 10, 0, 10);

    char hrunname[400];
    sprintf(hrunname, "badmodrun_%s", name.c_str());
    char hruntitle[400];
    sprintf(hruntitle, "Number of bad modules %s", name.c_str());
    m_badmodrun[name] = m_rhm.makeTProfile(hrunname, hruntitle, m_LSfrac * m_maxLS, 0, m_maxLS * 262144);
  }
}

SiStripQualityHistory::~SiStripQualityHistory() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void SiStripQualityHistory::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //  edm::LogInfo("EventProcessing") << "event being processed";

  for (std::size_t iMon = 0; iMon != _monitoredssq.size(); ++iMon) {
    std::string name = _monitoredssq[iMon].getParameter<std::string>("name");
    const auto& ssq = iSetup.getData(_ssqTokens[iMon]);

    std::vector<SiStripQuality::BadComponent> bads = ssq.getBadComponentList();

    LogDebug("BadComponents") << bads.size() << " bad components found";

    int nbad = 0;

    if (_mode == Module || _mode == Fiber || _mode == APV) {
      for (const auto& bc : bads) {
        if (_mode == Module) {
          if (bc.BadModule)
            ++nbad;
        } else if (_mode == Fiber) {
          for (int fiber = 1; fiber < 5; fiber *= 2) {
            if ((bc.BadFibers & fiber) > 0)
              ++nbad;
          }
        } else if (_mode == APV) {
          for (int apv = 1; apv < 33; apv *= 2) {
            if ((bc.BadApvs & apv) > 0)
              ++nbad;
          }
        }
      }
    } else if (_mode == Strip) {
      SiStripBadStrip::ContainerIterator dbegin = ssq.getDataVectorBegin();
      SiStripBadStrip::ContainerIterator dend = ssq.getDataVectorEnd();
      for (SiStripBadStrip::ContainerIterator data = dbegin; data < dend; ++data) {
        nbad += ssq.decode(*data).range;
      }
    }

    if (m_badmodrun.find(name) != m_badmodrun.end() && m_badmodrun[name] && *m_badmodrun[name]) {
      (*m_badmodrun[name])->Fill(iEvent.orbitNumber(), nbad);
    }
  }
}

void SiStripQualityHistory::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  m_rhm.beginRun(iRun);

  // loop on all the SiStripQuality objects to be monitored
  for (std::size_t iMon = 0; iMon != _monitoredssq.size(); ++iMon) {
    std::string name = _monitoredssq[iMon].getParameter<std::string>("name");

    if (m_badmodrun.find(name) != m_badmodrun.end()) {
      if (m_badmodrun[name] && *m_badmodrun[name]) {
        (*m_badmodrun[name])->SetCanExtend(TH1::kXaxis);
        (*m_badmodrun[name])->GetXaxis()->SetTitle("time [Orb#]");
        (*m_badmodrun[name])->GetYaxis()->SetTitle("bad components");
      }
    }

    if (m_run) {
      const auto& ssq = iSetup.getData(_ssqTokens[iMon]);

      std::vector<SiStripQuality::BadComponent> bads = ssq.getBadComponentList();

      LogDebug("BadComponents") << bads.size() << " bad components found";

      int nbad = 0;

      if (_mode == Module || _mode == Fiber || _mode == APV) {
        for (const auto& bc : bads) {
          if (_mode == Module) {
            if (bc.BadModule)
              ++nbad;
          } else if (_mode == Fiber) {
            for (int fiber = 1; fiber < 5; fiber *= 2) {
              if ((bc.BadFibers & fiber) > 0)
                ++nbad;
            }
          } else if (_mode == APV) {
            for (int apv = 1; apv < 33; apv *= 2) {
              if ((bc.BadApvs & apv) > 0)
                ++nbad;
            }
          }
        }
      } else if (_mode == Strip) {
        SiStripBadStrip::ContainerIterator dbegin = ssq.getDataVectorBegin();
        SiStripBadStrip::ContainerIterator dend = ssq.getDataVectorEnd();
        for (SiStripBadStrip::ContainerIterator data = dbegin; data < dend; ++data) {
          nbad += ssq.decode(*data).range;
        }
      }

      //    _history[name]->SetPoint(_history[name]->GetN(),iRun.run(),nbad);
      char runname[100];
      sprintf(runname, "%d", iRun.run());
      LogDebug("AnalyzedRun") << name << " " << runname << " " << nbad;
      _history[name]->Fill(runname, nbad);
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void SiStripQualityHistory::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void SiStripQualityHistory::endJob() {
  /*
  for(std::vector<edm::ParameterSet>::const_iterator ps=_monitoredssq.begin();ps!=_monitoredssq.end();++ps) {

    std::string name = ps->getParameter<std::string>("name");
    _history[name]->Write();

  }
  */
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripQualityHistory);
