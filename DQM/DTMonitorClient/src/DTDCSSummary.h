#ifndef DTMonitorClient_DTDCSSummary_H
#define DTMonitorClient_DTDCSSummary_H

/** \class DTDCSSummary
 *  No description available.
 *
 *  \author G. Cerminara - INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah ncpp-um-my
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <DQMServices/Core/interface/DQMEDHarvester.h>

#include <map>

class DQMStore;
class MonitorElement;

class DTDCSSummary : public DQMEDHarvester {
public:
  /// Constructor
  DTDCSSummary(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTDCSSummary();

  // Operations

  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, 
                               edm::LuminosityBlock const &, edm::EventSetup const &);

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;


protected:
  
private:

  MonitorElement*  totalDCSFraction;
  std::map<int, MonitorElement*> dcsFractions;

  bool bookingdone;

};


#endif

