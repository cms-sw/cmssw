
#ifndef MonitorElementsDb_H
#define MonitorElementsDb_H

/*!
  \file MonitorElementsDb.h
  \brief Generate a Monitor Element from DB data
  \author B. Gobbo 
*/


#include <string>
#include <vector>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MonitorXMLParser.h"

#include "RelationalAccess/ISessionProxy.h"

class MonitorElement;
class DQMStore;

class MonitorElementsDb{

  friend class EcalBarrelMonitorDbModule;

 public:

  /// Constructors
  MonitorElementsDb(const edm::ParameterSet& ps, std::string& xmlFile );

  /// Destructor
  virtual ~MonitorElementsDb();

 protected:

  /// Analyze
  void analyze( const edm::Event& e, const edm::EventSetup& c, coral::ISessionProxy* s );

  // BeginJob
  void beginJob( void );

  // EndJob
  void endJob( void ); 
 
  // HtmlOutput
  void htmlOutput( std::string& htmlDir );

  void startSession( void );

 private:

  std::string                   prefixME_;
  std::string                   xmlFile_;
  MonitorXMLParser*             parser_;
  std::vector<DB_ME>            MEinfo_;
  std::vector<MonitorElement*>  MEs_;
  int                           ievt_;
  DQMStore*        dqmStore_;
  
};

#endif  // MonitorElementsDb_H
