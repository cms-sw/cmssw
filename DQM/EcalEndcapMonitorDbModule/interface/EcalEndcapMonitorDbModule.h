#ifndef EcalEndcapMonitorDbModule_H
#define EcalEndcapMonitorDbModule_H

/*
 * \file EcalEndcapMonitorDbModule.h
 *
 * $Date: 2010/08/08 09:00:46 $
 * $Revision: 1.12 $
 * \author G. Della Ricca
 *
*/

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class DQMStore;

class MonitorElementsDb;

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
  void beginJob( void );

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
