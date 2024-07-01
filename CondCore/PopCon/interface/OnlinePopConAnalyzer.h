#ifndef CONDCORE_POPCON_ONLINEPOPCONANALYZER_H
#define CONDCORE_POPCON_ONLINEPOPCONANALYZER_H

//
// Authors:
//  - Francesco Brivio (Milano-Bicocca)
//  - Jan Chyczynski   (AGH University of Krakow)
//

#include "CondCore/PopCon/interface/OnlinePopCon.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

namespace popcon {
  template <typename S>
  class OnlinePopConAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    typedef S SourceHandler;

    OnlinePopConAnalyzer(const edm::ParameterSet& pset)
        : m_populator(pset), m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

    ~OnlinePopConAnalyzer() override {}

  protected:
    SourceHandler& source() { return m_source; }

  private:
    void beginJob() override {}
    void endJob() override { write(); }

    void analyze(const edm::Event&, const edm::EventSetup&) override {}

    void write() { m_populator.write(m_source); }

  private:
    OnlinePopCon m_populator;
    SourceHandler m_source;
  };

}  // namespace popcon
#endif  // CONDCORE_POPCON_ONLINEPOPCONANALYZER_H
