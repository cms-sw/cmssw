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

#include<TString.h>
#include<TPRegexp.h>

using namespace edm;
using namespace dqmTnP;
using namespace std;


typedef std::vector<std::string> vstring;

class DQMGenericTnPClient : public edm::EDAnalyzer{
  public:
    DQMGenericTnPClient(const edm::ParameterSet& pset);
    virtual ~DQMGenericTnPClient();
    virtual void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override {};
    virtual void endRun(const edm::Run &run, const edm::EventSetup &setup) override;
  void calculateEfficiency(std::string dirName, const ParameterSet& pset);
    void findAllSubdirectories (std::string dir, std::set<std::string> * myList, TString pattern);
  private:
    DQMStore * dqmStore;
    TFile * plots;
    vstring subDirs;
    std::string myDQMrootFolder;
    bool verbose;
    const VParameterSet efficiencies;
    GaussianPlusLinearFitter *GPLfitter;
    VoigtianPlusExponentialFitter *VPEfitter;
};

DQMGenericTnPClient::DQMGenericTnPClient(const edm::ParameterSet& pset):
  subDirs( pset.getUntrackedParameter<vstring>("subDirs", vstring()) ),
  myDQMrootFolder( pset.getUntrackedParameter<std::string>("MyDQMrootFolder", "") ),
  verbose( pset.getUntrackedParameter<bool>("Verbose",false) ),
  efficiencies( pset.getUntrackedParameter<VParameterSet>("Efficiencies") )
{
  TString savePlotsInRootFileName = pset.getUntrackedParameter<string>("SavePlotsInRootFileName","");
  plots = savePlotsInRootFileName!="" ? new TFile(savePlotsInRootFileName,"recreate") : 0;
  GPLfitter = new GaussianPlusLinearFitter(verbose);
  VPEfitter = new VoigtianPlusExponentialFitter(verbose);
}

void DQMGenericTnPClient::endRun(const edm::Run &run, const edm::EventSetup &setup){

  TPRegexp metacharacters("[\\^\\$\\.\\*\\+\\?\\|\\(\\)\\{\\}\\[\\]]");

  dqmStore = Service<DQMStore>().operator->();
  if( !dqmStore ){
    LogError("DQMGenericTnPClient")<<"Could not find DQMStore service\n";
    return;
  }
  dqmStore->setCurrentFolder(myDQMrootFolder);

  set<std::string> subDirSet;
  
  if (myDQMrootFolder != "")
    subDirSet.insert(myDQMrootFolder);
  else {
    for(vstring::const_iterator iSubDir = subDirs.begin(); 
        iSubDir != subDirs.end(); ++iSubDir) {
      std::string subDir = *iSubDir;
      if ( subDir[subDir.size()-1] == '/' ) subDir.erase(subDir.size()-1);
      if ( TString(subDir).Contains(metacharacters) ) {
        const string::size_type shiftPos = subDir.rfind('/');
        const string searchPath = subDir.substr(0, shiftPos);
        const string pattern    = subDir.substr(shiftPos + 1, subDir.length());
        findAllSubdirectories (searchPath, &subDirSet, pattern);
      }
      else {
        subDirSet.insert(subDir);
      }
    }
  }

  for(set<string>::const_iterator iSubDir = subDirSet.begin();
      iSubDir != subDirSet.end(); ++iSubDir) {
    const string& dirName = *iSubDir;
    for(VParameterSet::const_iterator pset = efficiencies.begin(); 
        pset != efficiencies.end(); ++pset) {
      calculateEfficiency(dirName, *pset);
    }
  }

}
  
void DQMGenericTnPClient::calculateEfficiency(std::string dirName, const ParameterSet& pset){
  //get hold of numerator and denominator histograms
  string allMEname = dirName+"/"+pset.getUntrackedParameter<string>("DenominatorMEname");
  string passMEname = dirName+"/"+pset.getUntrackedParameter<string>("NumeratorMEname");
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
  string effDir = dirName;
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

void DQMGenericTnPClient::findAllSubdirectories (std::string dir, std::set<std::string> * myList, TString pattern = "") {
  if (!dqmStore->dirExists(dir)) {
    LogError("DQMGenericTnPClient") << " DQMGenericTnPClient::findAllSubdirectories ==> Missing folder " << dir << " !!!";
    return;
  }
  TPRegexp nonPerlWildcard("\\w\\*|^\\*");
  if (pattern != "") {
    if (pattern.Contains(nonPerlWildcard)) pattern.ReplaceAll("*",".*");
    TPRegexp regexp(pattern);
    dqmStore->cd(dir);
    vector <string> foundDirs = dqmStore->getSubdirs();
    for(vector<string>::const_iterator iDir = foundDirs.begin();
        iDir != foundDirs.end(); ++iDir) {
      TString dirName = iDir->substr(iDir->rfind('/') + 1, iDir->length());
      if (dirName.Contains(regexp))
        findAllSubdirectories ( *iDir, myList);
    }
  }
  else if (dqmStore->dirExists(dir)){
    myList->insert(dir);
    dqmStore->cd(dir);
    findAllSubdirectories (dir, myList, "*");
  } else {
    
    LogInfo ("DQMGenericClient") << "Trying to find sub-directories of " << dir
                                 << " failed because " << dir  << " does not exist";
                                 
  }
  return;
}


//define this as a plug-in
DEFINE_FWK_MODULE(DQMGenericTnPClient);
