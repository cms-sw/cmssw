#ifndef BeamConditionsMonitor_H
#define BeamConditionsMonitor_H

/** \class BeamConditionsMonitor
 * *
 *  \author  Geng-yuan Jeng/UC Riverside
 *           Francisco Yumiceva/FNAL
 *   
 */
// C++
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/ConcurrentMonitorElement.h"

//
// class declaration
//
namespace beamcond {
  struct RunCache {
    // MonitorElements
    ConcurrentMonitorElement h_x0_lumi;
    ConcurrentMonitorElement h_y0_lumi;
  };
};

class BeamConditionsMonitor : public DQMGlobalEDAnalyzer<beamcond::RunCache, edm::LuminosityBlockCache<void>> {
 public:
  BeamConditionsMonitor( const edm::ParameterSet& );
  ~BeamConditionsMonitor() override = default;

 protected:
   
  // Book Histograms
  void bookHistograms(DQMStore::ConcurrentBooker& i, const edm::Run& r, const edm::EventSetup& c, beamcond::RunCache& ) const override;
  
  // Fake Analyze
  void dqmAnalyze(const edm::Event& e, const edm::EventSetup& c, beamcond::RunCache const& ) const override;
  
  // DQM Client Diagnostic (come from edm::LuminosityBlockCache use)
  std::shared_ptr<void> globalBeginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                                                   const edm::EventSetup& c) const override;
  void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override;

 private:
  
  std::string monitorName_;
  const edm::InputTag bsSrc_; // beam spot
  
};

#endif
