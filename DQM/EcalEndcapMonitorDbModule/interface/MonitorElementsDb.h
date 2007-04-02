// $Id: MonitorElementsDb.h,v 1.1 2006/06/28 10:46:17 benigno Exp $

#ifndef MonitorElementsDb_H
#define MonitorElementsDb_H

/*!
  \file MonitorElementsDb.h
  \brief Generate a Monitor Element from DB data
  \author B. Gobbo 
  \version $Revision: 1.1 $
  \date $Date: 2006/06/28 10:46:17 $
*/


#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <DQM/EcalEndcapMonitorDbModule/interface/MonitorXMLParser.h>

#include "RelationalAccess/ISessionProxy.h"

//class MonitorElementsDb: public edm::EDAnalyzer{
class MonitorElementsDb{

  friend class EcalEndcapMonitorDbModule;

 public:

  /// Constructors
  MonitorElementsDb(const edm::ParameterSet& ps, std::string xmlFile );

  /// Destructor
  virtual ~MonitorElementsDb();

 protected:

  /// Analyze
  void analyze( const edm::Event& e, const edm::EventSetup& c,   coral::ISessionProxy* s );

  // BeginJob
  void beginJob(const edm::EventSetup& c);

  // EndJob
  void endJob( void ); 
 
  // HtmlOutput
  void htmlOutput( std::string htmlDir );

  void startSession( void );

 private:

  std::string                   xmlFile_;
  MonitorXMLParser*             parser_;
  std::vector<DB_ME>            MEinfo_;
  std::vector<MonitorElement*>  MEs_;
  int                           ievt_;
  DaqMonitorBEInterface*        dbe_;
  
};

#endif  // MonitorElementsDb_H
