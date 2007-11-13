// $Id: MonitorElementsDb.h,v 1.2 2007/10/24 06:22:03 dellaric Exp $

#ifndef MonitorElementsDb_H
#define MonitorElementsDb_H

/*!
  \file MonitorElementsDb.h
  \brief Generate a Monitor Element from DB data
  \author B. Gobbo 
  \version $Revision: 1.2 $
  \date $Date: 2007/10/24 06:22:03 $
*/


#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <DQM/EcalBarrelMonitorDbModule/interface/MonitorXMLParser.h>

#include "RelationalAccess/ISessionProxy.h"

class MonitorElement;
class DaqMonitorBEInterface;

class MonitorElementsDb{

  friend class EcalBarrelMonitorDbModule;
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
