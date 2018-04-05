#ifndef DQMOffline_CalibTracker_SiStripPopConDQMEDHarvester_H
#define DQMOffline_CalibTracker_SiStripPopConDQMEDHarvester_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "CondCore/PopCon/interface/PopCon.h"

// copied from popCon::PopConAnalyzer
// modified to pass an edm::EventSetup reference at begin run
// and inherit from DQMEDHarvester
// Compared to popCon::SourceHandler, the concrete types should additionally implement
// the `void initES(const edm::EventSetup&)` and
// `void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&)` methods
template<class SourceHandler>
class SiStripPopConDQMEDHarvester : public DQMEDHarvester
{
public:
  SiStripPopConDQMEDHarvester(const edm::ParameterSet& pset) :
    m_populator(pset),
    m_source(pset.getParameter<edm::ParameterSet>("Source")) {}

  ~SiStripPopConDQMEDHarvester() override {}

private:
  void beginRun(const edm::Run&, const edm::EventSetup& setup) override
  {
    m_source.initES(setup);
  }

  void dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) override {
    m_source.dqmEndJob(booker, getter);
    m_populator.write(m_source);
  }

private:
  popcon::PopCon m_populator;
  SourceHandler m_source;
};

#endif // DQMOffline_CalibTracker_SiStripPopConDQMEDHarvester_H
