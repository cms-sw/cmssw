#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMOffline/Trigger/plugins/GenericTnPFitter.h"

using namespace edm;
using namespace dqmTnP;

class DQMGenericTnPClient : public edm::EDAnalyzer{
  public:
    DQMGenericTnPClient(const edm::ParameterSet& pset);
    virtual ~DQMGenericTnPClient();
    virtual void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {};
    virtual void endRun(const edm::Run &run, const edm::EventSetup &setup);
    void calculateEfficiency(const ParameterSet& pset);
  private:
    DQMStore * dqmStore;
    TFile * plots;
    string myDQMrootFolder;
    bool verbose;
    const VParameterSet efficiencies;
    GaussianPlusLinearFitter *GPLfitter;
    VoigtianPlusExponentialFitter *VPEfitter;
};

DQMGenericTnPClient::DQMGenericTnPClient(const edm::ParameterSet& pset):
  myDQMrootFolder( pset.getUntrackedParameter<string>("MyDQMrootFolder") ),
  verbose( pset.getUntrackedParameter<bool>("Verbose",false) ),
  efficiencies( pset.getUntrackedParameter<VParameterSet>("Efficiencies") )
{
  TString savePlotsInRootFileName = pset.getUntrackedParameter<string>("SavePlotsInRootFileName","");
  plots = savePlotsInRootFileName!="" ? new TFile(savePlotsInRootFileName,"recreate") : 0;
  GPLfitter = new GaussianPlusLinearFitter(verbose);
  VPEfitter = new VoigtianPlusExponentialFitter(verbose);
}

void DQMGenericTnPClient::endRun(const edm::Run &run, const edm::EventSetup &setup){
  dqmStore = Service<DQMStore>().operator->();
  if( !dqmStore ){
    LogError("DQMGenericTnPClient")<<"Could not find DQMStore service\n";
    return;
  }
  dqmStore->setCurrentFolder(myDQMrootFolder);
  //loop over all efficiency tasks
  for(VParameterSet::const_iterator pset = efficiencies.begin(); pset!=efficiencies.end(); pset++){
    calculateEfficiency(*pset);
  }
}
  
void DQMGenericTnPClient::calculateEfficiency(const ParameterSet& pset){
  //get hold of numerator and denominator histograms
  string allMEname = myDQMrootFolder+"/"+pset.getUntrackedParameter<string>("DenominatorMEname");
  string passMEname = myDQMrootFolder+"/"+pset.getUntrackedParameter<string>("NumeratorMEname");
  MonitorElement *allME = dqmStore->get(allMEname);
  MonitorElement *passME = dqmStore->get(passMEname);
  if(allME==0 || passME==0){
    LogDebug("DQMGenericTnPClient")<<"Could not find MEs: "<<allMEname<<" or "<<passMEname<<endl;
    return;
  }
  TH1 *all = allME->getTH1();
  TH1 *pass = passME->getTH1();
  //setup the fitter  
  string fitFunction = pset.getUntrackedParameter<string>("FitFunction");
  AbstractFitter *fitter = 0;
  if(fitFunction=="GaussianPlusLinear"){
    GPLfitter->setup(
      pset.getUntrackedParameter<double>("ExpectedMean"),
      pset.getUntrackedParameter<double>("FitRangeLow"),
      pset.getUntrackedParameter<double>("FitRangeHigh"),
      pset.getUntrackedParameter<double>("ExpectedSigma")
    );
    fitter = GPLfitter;
  }else if(fitFunction=="VoigtianPlusExponential"){
    VPEfitter->setup(
      pset.getUntrackedParameter<double>("ExpectedMean"),
      pset.getUntrackedParameter<double>("FitRangeLow"),
      pset.getUntrackedParameter<double>("FitRangeHigh"),
      pset.getUntrackedParameter<double>("ExpectedSigma"),
      pset.getUntrackedParameter<double>("FixedWidth")
    );
    fitter = VPEfitter;
  }else{
    LogError("DQMGenericTnPClient")<<"Fit function: "<<fitFunction<<" does not exist"<<endl;
    return;
  }
  //check dimensions
  int dimensions = all->GetDimension();
  int massDimension = pset.getUntrackedParameter<int>("MassDimension");
  if(massDimension>dimensions){
    LogError("DQMGenericTnPClient")<<"Monitoring Element "<<allMEname<<" has smaller dimension than "<<massDimension<<endl;
    return;
  }
  //figure out directory and efficiency names  
  string effName = pset.getUntrackedParameter<string>("EfficiencyMEname");
  string effDir = myDQMrootFolder;
  string::size_type slashPos = effName.rfind('/');
  if ( string::npos != slashPos ) {
    effDir += "/"+effName.substr(0, slashPos);
    effName.erase(0, slashPos+1);
  }
  dqmStore->setCurrentFolder(effDir);
  TString prefix(effDir.c_str());
  prefix.ReplaceAll('/','_');
  //calculate and book efficiency
  if(dimensions==2){
    TProfile* eff = 0;
    TProfile* effChi2 = 0;
    TString error = fitter->calculateEfficiency((TH2*)pass, (TH2*)all, massDimension, eff, effChi2, plots?prefix+effName.c_str():"");
    if(error!=""){
      LogError("DQMGenericTnPClient")<<error<<endl;
      return;
    }
    dqmStore->bookProfile(effName,eff);
    dqmStore->bookProfile(effName+"Chi2",effChi2);
    delete eff;
    delete effChi2;
  }else if(dimensions==3){
    TProfile2D* eff = 0;
    TProfile2D* effChi2 = 0;
    TString error = fitter->calculateEfficiency((TH3*)pass, (TH3*)all, massDimension, eff, effChi2, plots?prefix+effName.c_str():"");
    if(error!=""){
      LogError("DQMGenericTnPClient")<<error<<endl;
      return;
    }
    dqmStore->bookProfile2D(effName,eff);
    dqmStore->bookProfile2D(effName+"Chi2",effChi2);
    delete eff;
    delete effChi2;
  }
}

DQMGenericTnPClient::~DQMGenericTnPClient(){
  delete GPLfitter;
  if(plots){
    plots->Close();
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(DQMGenericTnPClient);
