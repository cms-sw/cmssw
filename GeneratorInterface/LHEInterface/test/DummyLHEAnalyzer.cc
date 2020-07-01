#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <iomanip>
#include <iostream>
using namespace std;
using namespace edm;
using namespace lhef;

class DummyLHEAnalyzer : public EDAnalyzer {
private:
  bool dumpEvent_;
  bool dumpHeader_;

public:
  explicit DummyLHEAnalyzer(const ParameterSet& cfg)
      : dumpEvent_(cfg.getUntrackedParameter<bool>("dumpEvent", true)),
        dumpHeader_(cfg.getUntrackedParameter<bool>("dumpHeader", false)),
        tokenLHERunInfo_(consumes<LHERunInfoProduct, edm::InRun>(
            cfg.getUntrackedParameter<edm::InputTag>("moduleLabel", std::string("source")))),
        tokenLHEEvent_(
            consumes<LHEEventProduct>(cfg.getUntrackedParameter<edm::InputTag>("moduleLabel", std::string("source")))) {
  }

private:
  void analyze(const Event& iEvent, const EventSetup& iSetup) override {
    edm::Handle<LHEEventProduct> evt;
    iEvent.getByToken(tokenLHEEvent_, evt);

    const lhef::HEPEUP hepeup_ = evt->hepeup();

    const int nup_ = hepeup_.NUP;
    const std::vector<int> idup_ = hepeup_.IDUP;
    const std::vector<lhef::HEPEUP::FiveVector> pup_ = hepeup_.PUP;

    if (!dumpEvent_) {
      return;
    }
    std::cout << "Number of particles = " << nup_ << std::endl;

    if (evt->pdf() != nullptr) {
      std::cout << "PDF scale = " << std::setw(14) << std::fixed << evt->pdf()->scalePDF << std::endl;
      std::cout << "PDF 1 : id = " << std::setw(14) << std::fixed << evt->pdf()->id.first << " x = " << std::setw(14)
                << std::fixed << evt->pdf()->x.first << " xPDF = " << std::setw(14) << std::fixed
                << evt->pdf()->xPDF.first << std::endl;
      std::cout << "PDF 2 : id = " << std::setw(14) << std::fixed << evt->pdf()->id.second << " x = " << std::setw(14)
                << std::fixed << evt->pdf()->x.second << " xPDF = " << std::setw(14) << std::fixed
                << evt->pdf()->xPDF.second << std::endl;
    }

    for (unsigned int icount = 0; icount < (unsigned int)nup_; icount++) {
      std::cout << "# " << std::setw(14) << std::fixed << icount << std::setw(14) << std::fixed << idup_[icount]
                << std::setw(14) << std::fixed << (pup_[icount])[0] << std::setw(14) << std::fixed << (pup_[icount])[1]
                << std::setw(14) << std::fixed << (pup_[icount])[2] << std::setw(14) << std::fixed << (pup_[icount])[3]
                << std::setw(14) << std::fixed << (pup_[icount])[4] << std::endl;
    }
    if (!evt->weights().empty()) {
      std::cout << "weights:" << std::endl;
      for (size_t iwgt = 0; iwgt < evt->weights().size(); ++iwgt) {
        const LHEEventProduct::WGT& wgt = evt->weights().at(iwgt);
        std::cout << "\t" << wgt.id << ' ' << std::scientific << wgt.wgt << std::endl;
      }
    }
  }

  void endRun(edm::Run const& iRun, edm::EventSetup const& es) override {
    Handle<LHERunInfoProduct> run;
    //iRun.getByLabel( src_, run );
    iRun.getByToken(tokenLHERunInfo_, run);

    const lhef::HEPRUP thisHeprup_ = run->heprup();

    std::cout << "HEPRUP \n" << std::endl;
    std::cout << "IDBMUP " << std::setw(14) << std::fixed << thisHeprup_.IDBMUP.first << std::setw(14) << std::fixed
              << thisHeprup_.IDBMUP.second << std::endl;
    std::cout << "EBMUP  " << std::setw(14) << std::fixed << thisHeprup_.EBMUP.first << std::setw(14) << std::fixed
              << thisHeprup_.EBMUP.second << std::endl;
    std::cout << "PDFGUP " << std::setw(14) << std::fixed << thisHeprup_.PDFGUP.first << std::setw(14) << std::fixed
              << thisHeprup_.PDFGUP.second << std::endl;
    std::cout << "PDFSUP " << std::setw(14) << std::fixed << thisHeprup_.PDFSUP.first << std::setw(14) << std::fixed
              << thisHeprup_.PDFSUP.second << std::endl;
    std::cout << "IDWTUP " << std::setw(14) << std::fixed << thisHeprup_.IDWTUP << std::endl;
    std::cout << "NPRUP  " << std::setw(14) << std::fixed << thisHeprup_.NPRUP << std::endl;
    std::cout << "        XSECUP " << std::setw(14) << std::fixed << "        XERRUP " << std::setw(14) << std::fixed
              << "        XMAXUP " << std::setw(14) << std::fixed << "        LPRUP  " << std::setw(14) << std::fixed
              << std::endl;
    for (unsigned int iSize = 0; iSize < thisHeprup_.XSECUP.size(); iSize++) {
      std::cout << std::setw(14) << std::fixed << thisHeprup_.XSECUP[iSize] << std::setw(14) << std::fixed
                << thisHeprup_.XERRUP[iSize] << std::setw(14) << std::fixed << thisHeprup_.XMAXUP[iSize]
                << std::setw(14) << std::fixed << thisHeprup_.LPRUP[iSize] << std::endl;
    }
    std::cout << " " << std::endl;

    if (dumpHeader_) {
      std::cout << " HEADER " << std::endl;
      for (auto it = run->headers_begin(); it != run->headers_end(); ++it) {
        std::cout << "tag: '" << it->tag() << "'" << std::endl;
        for (auto const& l : it->lines()) {
          std::cout << "   " << l << std::endl;
        }
      }
    }
  }

  edm::EDGetTokenT<LHERunInfoProduct> tokenLHERunInfo_;
  edm::EDGetTokenT<LHEEventProduct> tokenLHEEvent_;
};

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(DummyLHEAnalyzer);
