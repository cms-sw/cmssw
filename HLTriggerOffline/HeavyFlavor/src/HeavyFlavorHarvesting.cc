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

#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "RVersion.h"

#if ROOT_VERSION_CODE >= ROOT_VERSION(5,27,0)
#include "TEfficiency.h"
#else
#include "TGraphAsymmErrors.h"
#endif
using namespace edm;
using namespace std;

#if ROOT_VERSION_CODE >= ROOT_VERSION(5,27,0)
class HeavyFlavorHarvesting : public edm::EDAnalyzer { //, public TGraphAsymmErrors{
#else
class HeavyFlavorHarvesting : public edm::EDAnalyzer , public TGraphAsymmErrors{
#endif
  public:
    HeavyFlavorHarvesting(const edm::ParameterSet& pset);
    virtual ~HeavyFlavorHarvesting();
    virtual void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {};
    virtual void endRun(const edm::Run &, const edm::EventSetup &);
  private:
    void calculateEfficiency(const ParameterSet& pset);
    void calculateEfficiency1D( TH1* num, TH1* den, string name );
    void calculateEfficiency2D( TH2F* num, TH2F* den, string name );
    DQMStore * dqmStore;
    string myDQMrootFolder;
    const VParameterSet efficiencies;
};

HeavyFlavorHarvesting::HeavyFlavorHarvesting(const edm::ParameterSet& pset):
  myDQMrootFolder( pset.getUntrackedParameter<string>("MyDQMrootFolder") ),
  efficiencies( pset.getUntrackedParameter<VParameterSet>("Efficiencies") )
{
}

void HeavyFlavorHarvesting::endRun(const edm::Run &, const edm::EventSetup &){
  dqmStore = Service<DQMStore>().operator->();
  if( !dqmStore ){
    LogError("HLTriggerOfflineHeavyFlavor") << "Could not find DQMStore service\n";
    return;
  }
  for(VParameterSet::const_iterator pset = efficiencies.begin(); pset!=efficiencies.end(); pset++){
    calculateEfficiency(*pset);
  }
}
  
void HeavyFlavorHarvesting::calculateEfficiency(const ParameterSet& pset){
//get hold of numerator and denominator histograms
  vector<string> numDenEffMEnames = pset.getUntrackedParameter<vector<string> >("NumDenEffMEnames");
  if(numDenEffMEnames.size()!=3){
    LogDebug("HLTriggerOfflineHeavyFlavor") << "NumDenEffMEnames must have three names"<<endl;
    return;
  }
  string denMEname = myDQMrootFolder+"/"+numDenEffMEnames[1];
  string numMEname = myDQMrootFolder+"/"+numDenEffMEnames[0];
  MonitorElement *denME = dqmStore->get(denMEname);
  MonitorElement *numME = dqmStore->get(numMEname);
  if(denME==0 || numME==0){
    LogDebug("HLTriggerOfflineHeavyFlavor") << "Could not find MEs: "<<denMEname<<" or "<<numMEname<<endl;
    return;
  }
  TH1 *den = denME->getTH1();
  TH1 *num = numME->getTH1();
  //check compatibility of the histograms  
  if( den->GetNbinsX() != num->GetNbinsX() || den->GetNbinsY() != num->GetNbinsY() ||  den->GetNbinsZ() != num->GetNbinsZ() ){
    LogDebug("HLTriggerOfflineHeavyFlavor") << "Monitoring elements "<<numMEname<<" and "<<denMEname<<"are incompatible"<<endl;
    return;
  }
  //figure out the directory and efficiency name  
  string effName = numDenEffMEnames[2];
  string effDir = myDQMrootFolder;
  string::size_type slashPos = effName.rfind('/');
  if ( string::npos != slashPos ) {
    effDir += "/"+effName.substr(0, slashPos);
    effName.erase(0, slashPos+1);
  }
  dqmStore->setCurrentFolder(effDir);
  //calculate the efficiencies
  int dimensions = num->GetDimension();
  if(dimensions==1){
    calculateEfficiency1D( num, den, effName );
  }else if(dimensions==2){
    calculateEfficiency2D( (TH2F*)num, (TH2F*)den, effName );
    TH1D* numX = ((TH2F*)num)->ProjectionX(); 
    TH1D* denX = ((TH2F*)den)->ProjectionX();
    calculateEfficiency1D( numX, denX, effName+"X" );
    delete numX;
    delete denX;
    TH1D* numY = ((TH2F*)num)->ProjectionY();
    TH1D* denY = ((TH2F*)den)->ProjectionY();
    calculateEfficiency1D( numY, denY, effName+"Y" );
    delete numY;
    delete denY;
  }else{
    return;
  }
}

void HeavyFlavorHarvesting::calculateEfficiency1D( TH1* num, TH1* den, string effName ){
  TProfile* eff;
  if(num->GetXaxis()->GetXbins()->GetSize()==0){
    eff = new TProfile(effName.c_str(),effName.c_str(),num->GetXaxis()->GetNbins(),num->GetXaxis()->GetXmin(),num->GetXaxis()->GetXmax());
  }else{
    eff = new TProfile(effName.c_str(),effName.c_str(),num->GetXaxis()->GetNbins(),num->GetXaxis()->GetXbins()->GetArray());
  }
  eff->SetTitle(effName.c_str());
  eff->SetXTitle( num->GetXaxis()->GetTitle() );
  eff->SetYTitle("Efficiency");
  eff->SetOption("PE");
  eff->SetLineColor(2);
  eff->SetLineWidth(2);
  eff->SetMarkerStyle(20);
  eff->SetMarkerSize(0.8);
  eff->GetYaxis()->SetRangeUser(-0.001,1.001);
  eff->SetStats(kFALSE);
  for(int i=1;i<=num->GetNbinsX();i++){
    double e, low, high;
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,27,0)
    if (int(den->GetBinContent(i))>0.) e= double(num->GetBinContent(i))/double(den->GetBinContent(i));
    else e=0.;
    low=TEfficiency::Wilson((double)den->GetBinContent(i),(double)num->GetBinContent(i),0.683,false);
    high=TEfficiency::Wilson((double)den->GetBinContent(i),(double)num->GetBinContent(i),0.683,true);
#else
    Efficiency( (double)num->GetBinContent(i), (double)den->GetBinContent(i), 0.683, e, low, high );
#endif
    double err = e-low>high-e ? e-low : high-e;
    //here is the trick to store info in TProfile:
    eff->SetBinContent( i, e );
    eff->SetBinEntries( i, 1 );
    eff->SetBinError( i, sqrt(e*e+err*err) );
  }
  dqmStore->bookProfile(effName,eff);
  delete eff;
}

void HeavyFlavorHarvesting::calculateEfficiency2D( TH2F* num, TH2F* den, string effName ){
  TProfile2D* eff;
  if(num->GetXaxis()->GetXbins()->GetSize()==0 && num->GetYaxis()->GetXbins()->GetSize()==0){
    eff = new TProfile2D(effName.c_str(),effName.c_str(),num->GetXaxis()->GetNbins(),num->GetXaxis()->GetXmin(),num->GetXaxis()->GetXmax(),num->GetYaxis()->GetNbins(),num->GetYaxis()->GetXmin(),num->GetYaxis()->GetXmax());
  }else if(num->GetXaxis()->GetXbins()->GetSize()!=0 && num->GetYaxis()->GetXbins()->GetSize()==0){
    eff = new TProfile2D(effName.c_str(),effName.c_str(),num->GetXaxis()->GetNbins(),num->GetXaxis()->GetXbins()->GetArray(),num->GetYaxis()->GetNbins(),num->GetYaxis()->GetXmin(),num->GetYaxis()->GetXmax());
  }else if(num->GetXaxis()->GetXbins()->GetSize()==0 && num->GetYaxis()->GetXbins()->GetSize()!=0){
    eff = new TProfile2D(effName.c_str(),effName.c_str(),num->GetXaxis()->GetNbins(),num->GetXaxis()->GetXmin(),num->GetXaxis()->GetXmax(),num->GetYaxis()->GetNbins(),num->GetYaxis()->GetXbins()->GetArray());
  }else{
    eff = new TProfile2D(effName.c_str(),effName.c_str(),num->GetXaxis()->GetNbins(),num->GetXaxis()->GetXbins()->GetArray(),num->GetYaxis()->GetNbins(),num->GetYaxis()->GetXbins()->GetArray());
  }
  eff->SetTitle(effName.c_str());
  eff->SetXTitle( num->GetXaxis()->GetTitle() );
  eff->SetYTitle( num->GetYaxis()->GetTitle() );
  eff->SetZTitle("Efficiency");
  eff->SetOption("colztexte");
  eff->GetZaxis()->SetRangeUser(-0.001,1.001);
  eff->SetStats(kFALSE);
  for(int i=0;i<num->GetSize();i++){
    double e, low, high;
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,27,0)
    if (int(den->GetBinContent(i))>0.) e= double(num->GetBinContent(i))/double(den->GetBinContent(i));
    else e=0.;
    low=TEfficiency::Wilson((double)den->GetBinContent(i),(double)num->GetBinContent(i),0.683,false);
    high=TEfficiency::Wilson((double)den->GetBinContent(i),(double)num->GetBinContent(i),0.683,true);
#else
    Efficiency( (double)num->GetBinContent(i), (double)den->GetBinContent(i), 0.683, e, low, high );
#endif
    double err = e-low>high-e ? e-low : high-e;
    //here is the trick to store info in TProfile:
    eff->SetBinContent( i, e );
    eff->SetBinEntries( i, 1 );
    eff->SetBinError( i, sqrt(e*e+err*err) );
  }
  dqmStore->bookProfile2D(effName,eff);
  delete eff;
}

HeavyFlavorHarvesting::~HeavyFlavorHarvesting(){
}

//define this as a plug-in
DEFINE_FWK_MODULE(HeavyFlavorHarvesting);
