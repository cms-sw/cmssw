/* \class HTXSRivetProducer
 *
 * \author David Sperka, University of Florida
 *
 * $Id: HTXSRivetProducer.cc,v 1.1 2016/09/27 13:07:29 dsperka Exp $
 *
 */

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"

#include "Rivet/AnalysisHandler.hh"
#include "GeneratorInterface/RivetInterface/src/HiggsTemplateCrossSections.cc"
#include "SimDataFormats/HTXS/interface/HiggsTemplateCrossSections.h"

#include <memory>

#include <vector>
#include <cstdio>
#include <cstring>

using namespace Rivet;
using namespace edm;
using namespace std;

class HTXSRivetProducer : public edm::one::EDProducer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HTXSRivetProducer(const edm::ParameterSet& cfg)
      : _hepmcCollection(consumes<HepMCProduct>(cfg.getParameter<edm::InputTag>("HepMCCollection"))),
        _lheRunInfo(consumes<LHERunInfoProduct, edm::InRun>(cfg.getParameter<edm::InputTag>("LHERunInfo"))) {
    usesResource("Rivet");
    _prodMode = cfg.getParameter<string>("ProductionMode");
    m_HiggsProdMode = HTXS::UNKNOWN;
    _HTXS = nullptr;
    _analysisHandler = nullptr;
    produces<HTXS::HiggsClassification>("HiggsClassification").setBranchAlias("HiggsClassification");
  }

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  void beginRun(edm::Run const& iRun, edm::EventSetup const& es) override;
  void endRun(edm::Run const& iRun, edm::EventSetup const& es) override;

  edm::EDGetTokenT<edm::HepMCProduct> _hepmcCollection;
  edm::EDGetTokenT<LHERunInfoProduct> _lheRunInfo;

  std::unique_ptr<Rivet::AnalysisHandler> _analysisHandler;
  Rivet::HiggsTemplateCrossSections* _HTXS;

  std::string _prodMode;
  HTXS::HiggsProdMode m_HiggsProdMode;

  HTXS::HiggsClassification cat_;
};

void HTXSRivetProducer::produce(edm::Event& iEvent, const edm::EventSetup&) {
  //get the hepmc product from the event
  edm::Handle<HepMCProduct> evt;

  bool product_exists = iEvent.getByToken(_hepmcCollection, evt);
  if (product_exists) {
    // get HepMC GenEvent
    const HepMC::GenEvent* myGenEvent = evt->GetEvent();

    if (_prodMode == "AUTO") {
      // for these prod modes, don't change what is set in BeginRun
      if (m_HiggsProdMode != HTXS::GGF && m_HiggsProdMode != HTXS::VBF && m_HiggsProdMode != HTXS::GG2ZH) {
        unsigned nWs = 0;
        unsigned nZs = 0;
        unsigned nTs = 0;
        unsigned nBs = 0;
        unsigned nHs = 0;

        HepMC::GenVertex* HSvtx = myGenEvent->signal_process_vertex();

        if (HSvtx) {
          for (auto ptcl : HepMCUtils::particles(HSvtx, HepMC::children)) {
            if (std::abs(ptcl->pdg_id()) == 24)
              ++nWs;
            if (ptcl->pdg_id() == 23)
              ++nZs;
            if (abs(ptcl->pdg_id()) == 6)
              ++nTs;
            if (abs(ptcl->pdg_id()) == 5)
              ++nBs;
            if (ptcl->pdg_id() == 25)
              ++nHs;
          }
        }

        if (nZs == 1 && nHs == 1 && (nWs + nTs) == 0) {
          m_HiggsProdMode = HTXS::QQ2ZH;
        } else if (nWs == 1 && nHs == 1 && (nZs + nTs) == 0) {
          m_HiggsProdMode = HTXS::WH;
        } else if (nTs == 2 && nHs == 1 && nZs == 0) {
          m_HiggsProdMode = HTXS::TTH;
        } else if (nTs == 1 && nHs == 1 && nZs == 0) {
          m_HiggsProdMode = HTXS::TH;
        } else if (nBs == 2 && nHs == 1 && nZs == 0) {
          m_HiggsProdMode = HTXS::BBH;
        }
      }
    }

    if (!_HTXS || !_HTXS->hasProjection("FS")) {
      _analysisHandler = std::make_unique<Rivet::AnalysisHandler>();
      _HTXS = new Rivet::HiggsTemplateCrossSections();
      _analysisHandler->addAnalysis(_HTXS);

      // set the production mode if not done already
      if (_prodMode == "GGF")
        m_HiggsProdMode = HTXS::GGF;
      else if (_prodMode == "VBF")
        m_HiggsProdMode = HTXS::VBF;
      else if (_prodMode == "WH")
        m_HiggsProdMode = HTXS::WH;
      else if (_prodMode == "ZH")
        m_HiggsProdMode = HTXS::QQ2ZH;
      else if (_prodMode == "QQ2ZH")
        m_HiggsProdMode = HTXS::QQ2ZH;
      else if (_prodMode == "GG2ZH")
        m_HiggsProdMode = HTXS::GG2ZH;
      else if (_prodMode == "TTH")
        m_HiggsProdMode = HTXS::TTH;
      else if (_prodMode == "BBH")
        m_HiggsProdMode = HTXS::BBH;
      else if (_prodMode == "TH")
        m_HiggsProdMode = HTXS::TH;
      else if (_prodMode == "AUTO") {
        edm::LogInfo("HTXSRivetProducer")
            << "Using AUTO for HiggsProdMode, found it to be: " << m_HiggsProdMode << "\n";
        edm::LogInfo("HTXSRivetProducer")
            << "(UNKNOWN=0, GGF=1, VBF=2, WH=3, QQ2ZH=4, GG2ZH=5, TTH=6, BBH=7, TH=8)" << endl;
      } else {
        throw cms::Exception("HTXSRivetProducer")
            << "ProductionMode must be one of: GGF,VBF,WH,ZH,QQ2ZH,GG2ZH,TTH,BBH,TH,AUTO ";
      }
      _HTXS->setHiggsProdMode(m_HiggsProdMode);

      // at this point the production mode must be known
      if (m_HiggsProdMode == HTXS::UNKNOWN) {
        edm::LogInfo("HTXSRivetProducer") << "HTXSRivetProducer WARNING: HiggsProduction mode is UNKNOWN" << endl;
      }

      // initialize rivet analysis
      _analysisHandler->init(*myGenEvent);
    }

    // classify the event
    Rivet::HiggsClassification rivet_cat = _HTXS->classifyEvent(*myGenEvent, m_HiggsProdMode);
    cat_ = HTXS::Rivet2Root(rivet_cat);

    unique_ptr<HTXS::HiggsClassification> cat(new HTXS::HiggsClassification(cat_));

    iEvent.put(std::move(cat), "HiggsClassification");
  }
}

void HTXSRivetProducer::endRun(edm::Run const& iRun, edm::EventSetup const& es) {
  if (_HTXS)
    _HTXS->printClassificationSummary();
}

void HTXSRivetProducer::beginRun(edm::Run const& iRun, edm::EventSetup const& es) {
  if (_prodMode == "AUTO") {
    edm::Handle<LHERunInfoProduct> run;
    bool product_exists = iRun.getByLabel(edm::InputTag("externalLHEProducer"), run);
    if (product_exists) {
      typedef std::vector<LHERunInfoProduct::Header>::const_iterator headers_const_iterator;
      LHERunInfoProduct myLHERunInfoProduct = *(run.product());
      for (headers_const_iterator iter = myLHERunInfoProduct.headers_begin(); iter != myLHERunInfoProduct.headers_end();
           iter++) {
        std::vector<std::string> lines = iter->lines();
        for (unsigned int iLine = 0; iLine < lines.size(); iLine++) {
          std::string line = lines.at(iLine);
          // POWHEG
          if (line.find("gg_H_quark-mass-effects") != std::string::npos) {
            edm::LogInfo("HTXSRivetProducer") << iLine << " " << line << std::endl;
            m_HiggsProdMode = HTXS::GGF;
            break;
          }
          if (line.find("Process: HJ") != std::string::npos) {
            edm::LogInfo("HTXSRivetProducer") << iLine << " " << line << std::endl;
            m_HiggsProdMode = HTXS::GGF;
            break;
          }
          if (line.find("Process: HJJ") != std::string::npos) {
            edm::LogInfo("HTXSRivetProducer") << iLine << " " << line << std::endl;
            m_HiggsProdMode = HTXS::GGF;
            break;
          }
          if (line.find("VBF_H") != std::string::npos) {
            edm::LogInfo("HTXSRivetProducer") << iLine << " " << line << std::endl;
            m_HiggsProdMode = HTXS::VBF;
            break;
          }
          if (line.find("HZJ") != std::string::npos) {
            edm::LogInfo("HTXSRivetProducer") << iLine << " " << line << std::endl;
            m_HiggsProdMode = HTXS::QQ2ZH;
            break;
          }
          if (line.find("ggHZ") != std::string::npos) {
            edm::LogInfo("HTXSRivetProducer") << iLine << " " << line << std::endl;
            m_HiggsProdMode = HTXS::GG2ZH;
            break;
          }
          // MC@NLO
          if (line.find("ggh012j") != std::string::npos) {
            edm::LogInfo("HTXSRivetProducer") << iLine << " " << line << std::endl;
            m_HiggsProdMode = HTXS::GGF;
            break;
          }
          if (line.find("vbfh") != std::string::npos) {
            edm::LogInfo("HTXSRivetProducer") << iLine << " " << line << std::endl;
            m_HiggsProdMode = HTXS::VBF;
            break;
          }
          if (line.find("zh012j") != std::string::npos) {
            edm::LogInfo("HTXSRivetProducer") << iLine << " " << line << std::endl;
            m_HiggsProdMode = HTXS::QQ2ZH;
            break;
          }
          if (line.find("ggzh01j") != std::string::npos) {
            edm::LogInfo("HTXSRivetProducer") << iLine << " " << line << std::endl;
            m_HiggsProdMode = HTXS::GG2ZH;
            break;
          }
        }

        if (m_HiggsProdMode != HTXS::UNKNOWN)
          break;
      }
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HTXSRivetProducer);
