#ifndef DTMonitorClient_DTDAQInfo_H
#define DTMonitorClient_DTDAQInfo_H

/** \class DTDAQInfo
 *  No description available.
 *
 *  \author G. Cerminara - INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah ncpp-um-my
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <DQMServices/Core/interface/DQMEDHarvester.h>

#include <map>

class DQMStore;
class MonitorElement;
class DTReadOutMapping;

class DTDAQInfo : public DQMEDHarvester {
public:
  /// Constructor
  DTDAQInfo(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTDAQInfo();

  // Operations

protected:
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, 
                                                      edm::EventSetup const &);
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &);

private:

  bool bookingdone;
  
  MonitorElement*  totalDAQFraction;
  MonitorElement*  daqMap;
  std::map<int, MonitorElement*> daqFractions;
  edm::ESHandle<DTReadOutMapping> mapping;

};


#endif

