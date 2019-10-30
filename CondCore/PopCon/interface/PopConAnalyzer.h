#ifndef POPCON_ANALYZER_H
#define POPCON_ANALYZER_H

//
// Original Author:  Marcin BOGUSZ
//         Created:  Tue Jul  3 10:48:22 CEST 2007

#include "CondCore/PopCon/interface/PopCon.h"
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace popcon {
  template <typename S>
  class PopConAnalyzer : public edm::EDAnalyzer {
  public:
    typedef S SourceHandler;

    PopConAnalyzer(const edm::ParameterSet& pset)
        : m_populator(pset), m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

    ~PopConAnalyzer() override {}

  protected:
    SourceHandler& source() { return m_source; }

  private:
    void beginJob() override {}
    void endJob() override { write(); }

    void analyze(const edm::Event&, const edm::EventSetup&) override {}

    void write() { m_populator.write(m_source); }

  private:
    PopCon m_populator;
    SourceHandler m_source;
  };

}  // namespace popcon
#endif
