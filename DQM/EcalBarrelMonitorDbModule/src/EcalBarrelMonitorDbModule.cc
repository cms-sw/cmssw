/*
 * \file EcalBarrelMonitorDbModule.cc
 * 
 * $Date: 2006/06/06 09:27:08 $
 * $Revision: 1.1 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorDbModule/interface/EcalBarrelMonitorDbModule.h>

EcalBarrelMonitorDbModule::EcalBarrelMonitorDbModule(const edm::ParameterSet& ps){

  dbName_ = ps.getUntrackedParameter<string>("dbName", "");
  dbHostName_ = ps.getUntrackedParameter<string>("dbHostName", "");
  dbUserName_ = ps.getUntrackedParameter<string>("dbUserName", "");
  dbPassword_ = ps.getUntrackedParameter<string>("dbPassword", "");

  cout << " DB output will come from"
       << " dbName = " << dbName_
       << " dbHostName = " << dbHostName_
       << " dbUserName = " << dbUserName_ << endl;

  dbe = 0;

  // get hold of back-end interface
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();

  // MonitorDaemon switch
  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", true);

  if ( enableMonitorDaemon_ ) {
    LogInfo("EcalBarrelMonitor") << " enableMonitorDaemon switch is ON";
    Service<MonitorDaemon> daemon;
    daemon.operator->();
  } else {
    LogInfo("EcalBarrelMonitor") << " enableMonitorDaemon switch is OFF";
  }
  
  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");
  if ( outputFile_.size() != 0 ) {
    cout << "Ecal Barrel Db Monitoring histograms will be saved to " << outputFile_.c_str() << endl;
  }

  // html output directory
  htmlDir_ = ps.getUntrackedParameter<string>("htmlDir", ".");

  if ( htmlDir_.size() != 0 ) {
    cout << " HTML output will go to"
         << " htmlDir = " << htmlDir_ << endl;
  } else {
    cout << " HTML output is disabled" << endl;
  }

  tempDb_ = new EBTemperatureDb(ps, dbe);

  if ( dbe ) dbe->showDirStructure();

  // this should give enough time to all the MEs to reach the Collector,
  // and then hopefully the clients
//  sleep(10);

}

EcalBarrelMonitorDbModule::~EcalBarrelMonitorDbModule(){

  if ( tempDb_ ) delete tempDb_;

}

void EcalBarrelMonitorDbModule::beginJob(const edm::EventSetup& c){

  icycle_ = 0;

  if ( tempDb_ ) tempDb_->beginJob(c);

}

void EcalBarrelMonitorDbModule::endJob(void) {

  if ( tempDb_ ) tempDb_->endJob();

  cout << "EcalBarrelMonitorDbModule: endJob, icycle = " << icycle_ << endl;

}

void EcalBarrelMonitorDbModule::analyze(const edm::Event& e, const edm::EventSetup& c){

  icycle_++;
//  if ( icycle_ % 10 == 0 )
    cout << "EcalBarrelMonitorDbModule: icycle = " << icycle_ << endl;

  // Creates the sessions
  ISessionProxy* readProxy = 0;

  try {
    Context* context = new Context;
    PluginManager* pm = PluginManager::get();
    pm->initialise ();
    seal::Handle<ComponentLoader> loader = new ComponentLoader(context);
    loader->load("CORAL/Services/ConnectionService");

    IHandle<IConnectionService> connectionService = context->query<IConnectionService>("CORAL/Services/ConnectionService");

    // Set configuration parameters
    IConnectionServiceConfiguration& config = connectionService->configuration();
    config.setNumberOfConnectionRetrials(2);

    readProxy = connectionService->connect("ECAL CondDB", ReadOnly);

    if ( tempDb_ ) tempDb_->analyze(e, c, dbe, readProxy);

  } catch (seal::Exception& se) {
    cerr << "Seal Exception : " << se.what() << endl;
  } catch (std::exception& e) {
    cerr << "Standard C++ exception : " << e.what() << endl;
  } catch (...) {
    cerr << "Exception caught (...)" << endl;
  }

  if ( htmlDir_.size() != 0 ) {

    tempDb_->htmlOutput(htmlDir_);

  }

  if ( outputFile_.size() != 0 && dbe ) dbe->save(outputFile_);

  sleep(10);

}

