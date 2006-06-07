/*
 * \file EcalBarrelMonitorDbModule.cc
 * 
 * $Date: 2006/06/07 08:36:56 $
 * $Revision: 1.4 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorDbModule/interface/EcalBarrelMonitorDbModule.h>

EcalBarrelMonitorDbModule::EcalBarrelMonitorDbModule(const ParameterSet& ps){

  dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  // MonitorDaemon switch
  enableMonitorDaemon_ = ps.getUntrackedParameter<bool>("enableMonitorDaemon", true);

  if ( enableMonitorDaemon_ ) {
    cout << " enableMonitorDaemon switch is ON" << endl;
    Service<MonitorDaemon> daemon;
    daemon.operator->();
  } else {
    cout << " enableMonitorDaemon switch is OFF" << endl;
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

  pedDb_ = new EBPedestalDb(ps, dbe);

  if ( dbe ) dbe->showDirStructure();

}

EcalBarrelMonitorDbModule::~EcalBarrelMonitorDbModule(){

  if ( tempDb_ ) delete tempDb_;

  if ( pedDb_ ) delete pedDb_;

}

void EcalBarrelMonitorDbModule::beginJob(const EventSetup& c){

  icycle_ = 0;

  if ( tempDb_ ) tempDb_->beginJob(c);

  if ( pedDb_ ) pedDb_->beginJob(c);

}

void EcalBarrelMonitorDbModule::endJob(void) {

  if ( tempDb_ ) tempDb_->endJob();

  if ( pedDb_ ) pedDb_->endJob();

  cout << "EcalBarrelMonitorDbModule: endJob, icycle = " << icycle_ << endl;

}

void EcalBarrelMonitorDbModule::analyze(const Event& e, const EventSetup& c){

  icycle_++;

  cout << "EcalBarrelMonitorDbModule: icycle = " << icycle_ << endl;

  // Creates the sessions
  ISessionProxy* readProxy = 0;

  try {
    seal::Handle<Context> context = new Context;
    PluginManager* pm = PluginManager::get();
    pm->initialise ();
    seal::Handle<ComponentLoader> loader = new ComponentLoader(context.get());

    loader->load("SEAL/Services/MessageService");

    vector<seal::Handle<IMessageService> > v_msgSvc;
    context->query(v_msgSvc);
    if ( ! v_msgSvc.empty() ) {
      seal::Handle<IMessageService>& msgSvc = v_msgSvc.front();
//      msgSvc->setOutputLevel(Msg::Error);
      msgSvc->setOutputLevel(Msg::Debug);
    }

    loader->load("CORAL/Services/ConnectionService");

    loader->load("CORAL/Services/EnvironmentAuthenticationService");

    IHandle<IConnectionService> connectionService = context->query<IConnectionService>("CORAL/Services/ConnectionService");

    loader->load("CORAL/RelationalPlugins/oracle");

    // Set configuration parameters
    IConnectionServiceConfiguration& config = connectionService->configuration();
    config.setNumberOfConnectionRetrials(2);

    readProxy = connectionService->connect("ECAL CondDB", ReadOnly);

    if ( tempDb_ ) tempDb_->analyze(e, c, dbe, readProxy);

    if ( pedDb_ ) pedDb_->analyze(e, c, dbe, readProxy);

  } catch (coral::Exception& se) {
    cerr << "CORAL Exception : " << se.what() << endl;
  } catch (std::exception& e) {
    cerr << "Standard C++ exception : " << e.what() << endl;
  } catch (...) {
    cerr << "Exception caught (...)" << endl;
  }

  if ( htmlDir_.size() != 0 ) {

    tempDb_->htmlOutput(htmlDir_);

    pedDb_->htmlOutput(htmlDir_);

  }

  if ( outputFile_.size() != 0 && dbe ) dbe->save(outputFile_);

  sleep(10);

}

