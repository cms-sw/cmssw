#ifndef EcalEndcapMonitorDbModule_H
#define EcalEndcapMonitorDbModule_H

/*
 * \file EcalEndcapMonitorDbModule.h
 *
 * $Date: 2008/04/08 15:06:26 $
 * $Revision: 1.7 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>

class DQMStore;

class MonitorElementsDb;
class coral::ISessionProxy;

class EcalEndcapMonitorDbModule: public edm::EDAnalyzer{

 public:

  /// Constructor
  EcalEndcapMonitorDbModule( const edm::ParameterSet& ps );

  /// Destructor
  virtual ~EcalEndcapMonitorDbModule();

 protected:

  /// Analyze
  void analyze( const edm::Event& e, const edm::EventSetup& c );

  // BeginJob
  void beginJob( const edm::EventSetup& c );

  // EndJob
  void endJob( void );

 private:
  
  int icycle_;
 
  DQMStore* dqmStore_;

  std::string prefixME_;

  std::string htmlDir_;

  std::string xmlFile_;

  MonitorElementsDb* ME_Db_;

  unsigned int sleepTime_;

  coral::ISessionProxy* session_;

};

#endif
