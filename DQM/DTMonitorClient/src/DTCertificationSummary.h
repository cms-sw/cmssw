#ifndef DTMonitorClient_DTCertificationSummary_H
#define DTMonitorClient_DTCertificationSummary_H

/** \class DTCertificationSummary
 *  No description available.
 *
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <DQMServices/Core/interface/DQMEDHarvester.h>

#include <map>

class DQMStore;
class MonitorElement;

class DTCertificationSummary : public  DQMEDHarvester {
public:
  /// Constructor
  DTCertificationSummary(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTCertificationSummary();

  // Operations

protected:
  
private:

    void beginRun(const edm::Run& run, const  edm::EventSetup& setup);

    /// DQM Client Diagnostic in online mode
    void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&);

    void endRun(const edm::Run& run, const edm::EventSetup& setup);

    /// DQM Client Diagnostic in offline mode
    void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &);
    
  MonitorElement*  totalCertFraction;
  MonitorElement*  certMap;
  std::map<int, MonitorElement*> certFractions;

};


#endif

