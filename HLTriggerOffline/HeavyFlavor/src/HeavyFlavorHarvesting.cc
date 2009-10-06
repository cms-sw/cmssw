#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TGraphAsymmErrors.h"

using namespace edm;
using namespace std;

class HeavyFlavorHarvesting : public edm::EDAnalyzer, public TGraphAsymmErrors{
  public:
    HeavyFlavorHarvesting(const edm::ParameterSet& pset);
    virtual ~HeavyFlavorHarvesting();
    virtual void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {};
    virtual void endJob();
    void calculateEfficiency(const ParameterSet& pset);
  private:
    DQMStore * dqmStore;
    string myDQMrootFolder;
    const VParameterSet efficiencies;
};

HeavyFlavorHarvesting::HeavyFlavorHarvesting(const edm::ParameterSet& pset):
  myDQMrootFolder( pset.getUntrackedParameter<string>("MyDQMrootFolder") ),
  efficiencies( pset.getUntrackedParameter<VParameterSet>("Efficiencies") )
{
}

void HeavyFlavorHarvesting::endJob(){
  dqmStore = Service<DQMStore>().operator->();
  if( !dqmStore ){
    LogError("HLTriggerOfflineHeavyFlavor") << "Could not find DQMStore service\n";
    return;
  }
  dqmStore->setCurrentFolder(myDQMrootFolder);
  for(VParameterSet::const_iterator pset = efficiencies.begin(); pset!=efficiencies.end(); pset++){
    calculateEfficiency(*pset);
  }
}
  
void HeavyFlavorHarvesting::calculateEfficiency(const ParameterSet& pset){
//get hold of numerator and denominator histograms
  string denMEname = myDQMrootFolder+"/"+pset.getUntrackedParameter<string>("DenominatorMEname");
  string numMEname = myDQMrootFolder+"/"+pset.getUntrackedParameter<string>("NumeratorMEname");
  MonitorElement *denME = dqmStore->get(denMEname);
  MonitorElement *numME = dqmStore->get(numMEname);
  if(denME==0 || numME==0){
    LogDebug("HLTriggerOfflineHeavyFlavor") << "Could not find MEs: "<<denMEname<<" or "<<numMEname<<endl;
    return;
  }
  TH1 *den = denME->getTH1();
  TH1 *num = numME->getTH1();
//check dimensionality of the histograms  
  if( den->GetNbinsX() != num->GetNbinsX() || den->GetNbinsY() != num->GetNbinsY() ||  den->GetNbinsZ() != num->GetNbinsZ() ){
    LogError("HLTriggerOfflineHeavyFlavor") << "Monitoring elements "<<numMEname<<" and "<<denMEname<<"are incompatible"<<endl;
    return;
  }
//book the efficiency histogram  
  string effName = pset.getUntrackedParameter<string>("EfficiencyMEname");
  string effDir = myDQMrootFolder;
  string::size_type slashPos = effName.rfind('/');
  if ( string::npos != slashPos ) {
    effDir += "/"+effName.substr(0, slashPos);
    effName.erase(0, slashPos+1);
  }
  dqmStore->setCurrentFolder(effDir);
  MonitorElement * effME;
  int numberOfCells = 0;
  TH1 * eff = (TH1*)num->Clone(effName.c_str());
  eff->SetTitle(effName.c_str());
  int dimensions = eff->GetDimension();
  if(dimensions==1){
      numberOfCells = ((TH1F*)eff)->GetSize();
      effME = dqmStore->book1D(effName,(TH1F*)eff);
      delete eff;
      eff = effME->getTH1();
      eff->SetOption("PE");
      eff->SetLineColor(2);
      eff->SetLineWidth(2);
      eff->GetYaxis()->SetRangeUser(-0.001,1.001);
      eff->SetStats(kFALSE);
  }else if(dimensions==2){
      numberOfCells = ((TH2F*)eff)->GetSize();
      effME = dqmStore->book2D(effName,(TH2F*)eff);
      delete eff;
      eff = effME->getTH1();
      eff->SetOption("colztexte");
      eff->GetZaxis()->SetRangeUser(-0.001,1.001);
      eff->SetStats(kFALSE);

      TH1D* numX = ((TH2F*)num)->ProjectionX(); 
      TH1D* denX = ((TH2F*)den)->ProjectionX();
      TH1F* effX;
      if(numX->GetXaxis()->GetXbins()->GetSize()==0){
        effX = new TH1F((effName+"X").c_str(),(effName+"X").c_str(),numX->GetXaxis()->GetNbins(),numX->GetXaxis()->GetXmin(),numX->GetXaxis()->GetXmax());
      }else{
        effX = new TH1F((effName+"X").c_str(),(effName+"X").c_str(),numX->GetXaxis()->GetNbins(),numX->GetXaxis()->GetXbins()->GetArray());
      }
      effX->SetXTitle(num->GetXaxis()->GetTitle());
      effX->SetYTitle("Efficiency");
      MonitorElement *effMEX = dqmStore->book1D(effName+"X",effX);
      delete effX;
      effX = (TH1F*)effMEX->getTH1();
      effX->SetOption("PE");
      effX->SetLineColor(2);
      effX->SetLineWidth(2);
      effX->GetYaxis()->SetRangeUser(-0.001,1.001);
      effX->SetStats(kFALSE);
      for(int i=1;i<=numX->GetNbinsX();i++){
        double e;
        double low;
        double high;
        Efficiency((int)numX->GetBinContent(i),(int)denX->GetBinContent(i),0.683,e,low,high);
        effX->SetBinContent(i,e);
        effX->SetBinError( i, e-low>high-e ? e-low : high-e );
      }

      TH1D* numY = ((TH2F*)num)->ProjectionY();
      TH1D* denY = ((TH2F*)den)->ProjectionY();
      TH1F* effY;
      if(numY->GetXaxis()->GetXbins()->GetSize()==0){
        effY = new TH1F((effName+"Y").c_str(),(effName+"Y").c_str(),numY->GetXaxis()->GetNbins(),numY->GetXaxis()->GetXmin(),numY->GetXaxis()->GetXmax());
      }else{
        effY = new TH1F((effName+"Y").c_str(),(effName+"Y").c_str(),numY->GetXaxis()->GetNbins(),numY->GetXaxis()->GetXbins()->GetArray());
      }
      effY->SetXTitle(num->GetYaxis()->GetTitle());
      effY->SetYTitle("Efficiency");
      MonitorElement *effMEY = dqmStore->book1D(effName+"Y",effY);
      delete effY;
      effY = (TH1F*)effMEY->getTH1();
      effY->SetOption("PE");
      effY->SetLineColor(2);
      effY->SetLineWidth(2);
      effY->GetYaxis()->SetRangeUser(-0.001,1.001);
      effY->SetStats(kFALSE);
      for(int i=1;i<=numY->GetNbinsX();i++){
        double e;
        double low;
        double high;
        Efficiency((int)numY->GetBinContent(i),(int)denY->GetBinContent(i),0.683,e,low,high);
        effY->SetBinContent(i,e);
        effY->SetBinError( i, e-low>high-e ? e-low : high-e );
      }
/*  }else if(dimensions==3){
    numberOfCells = ((TH3F*)eff)->GetSize();
    effME = dqmStore->book3D(effName,(TH3F*)eff);
    delete eff;
    eff = effME->getTH1();*/
  }else{
    return;
  }
  for(int i=0;i<numberOfCells;i++){
    double e;
    double low;
    double high;
    Efficiency((int)num->GetBinContent(i),(int)den->GetBinContent(i),0.683,e,low,high);
    eff->SetBinContent(i,e);
    eff->SetBinError( i, e-low>high-e ? e-low : high-e );
  }
}

HeavyFlavorHarvesting::~HeavyFlavorHarvesting(){
}

//define this as a plug-in
DEFINE_FWK_MODULE(HeavyFlavorHarvesting);
