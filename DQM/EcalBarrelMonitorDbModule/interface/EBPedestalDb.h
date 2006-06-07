#ifndef EBPedestalDb_H
#define EBPedestalDb_H

/*
 * \file EBPedestalDb.h
 *
 * $Date: 2006/06/06 18:10:58 $
 * $Revision: 1.3 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SealKernel/Context.h"
#include "SealKernel/ComponentLoader.h"
#include "SealKernel/Exception.h"
#include "PluginManager/PluginManager.h"
#include "RelationalAccess/IConnectionService.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeSpecification.h"

using namespace seal;
using namespace coral;

#include "TROOT.h"
#include "TStyle.h"
#include "TPaveStats.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace cms;
using namespace edm;
using namespace std;

//class EBPedestalDb: public edm::EDAnalyzer{
class EBPedestalDb{

friend class EcalBarrelMonitorDbModule;

public:

/// Constructor
EBPedestalDb(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);

/// Destructor
virtual ~EBPedestalDb();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c, DaqMonitorBEInterface* dbe, ISessionProxy* isp);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void); 

// HtmlOutput
void htmlOutput(string htmlDir);

private:

int ievt_;

MonitorElement* mePed01_;
MonitorElement* mePed06_;
MonitorElement* mePed12_;

MonitorElement* meRms01_;
MonitorElement* meRms06_;
MonitorElement* meRms12_;

};

#endif
