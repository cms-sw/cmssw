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

#include "RooGaussian.h"
#include "RooChebychev.h"
#include "RooRealVar.h"
#include "RooFormulaVar.h"
#include "RooDataHist.h"
#include "RooAddPdf.h"
#include "RooGlobalFunc.h"
#include "RooCategory.h"
#include "RooSimultaneous.h"
#include "RooChi2Var.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TCanvas.h"
#include "RooPlot.h"

using namespace RooFit;
using namespace edm;

class AbstractFitter{
  protected:
    RooRealVar mass;
    RooRealVar mean;
    double expectedMean;
    RooRealVar nSignalAll;
    RooRealVar efficiency;
    RooFormulaVar nSignalPass;
    RooRealVar nBackgroundAll;
    RooRealVar nBackgroundPass;
    RooCategory category;
    RooSimultaneous simPdf;
    RooDataHist *data;
    double chi2;
  public:
    AbstractFitter();
    virtual ~AbstractFitter(){};
    AbstractFitter(double massMean, double massLow, double massHigh):
      mass("mass","mass",massLow,massHigh,"GeV"),
      mean("mean","mean",massMean,massLow,massHigh,"GeV"),
      expectedMean(massMean),
      nSignalAll("nSignalAll","nSignalAll",0.,1e10),
      efficiency("efficiency","efficiency",0.5,0.0,1.0),
      nSignalPass("nSignalPass","nSignalAll*efficiency",RooArgList(nSignalAll,efficiency)),
      nBackgroundAll("nBackgroundAll","nBackgroundAll",0.,1e10),
      nBackgroundPass("nBackgroundPass","nBackgroundPass",0.,1e10),
      category("category","category"),
      simPdf("simPdf","simPdf",category)
    {
      category.defineType("pass");
      category.defineType("all");
    };
    virtual void fit(TH1* num, TH1* den) = 0;
    double getEfficiency(){ return efficiency.getVal(); }
    double getEfficiencyError(){ return efficiency.getError(); }
    double getChi2(){ return chi2; }
    RooPlot* savePlot(TString name){ 
      RooPlot* frame = mass.frame(Name(name), Title("All and Passing Probe Distributions"));
      data->plotOn(frame,Cut("category==category::all"),LineColor(kRed));
      data->plotOn(frame,Cut("category==category::pass"),LineColor(kGreen));
      simPdf.plotOn(frame,Slice(category,"all"),ProjWData(category,*data),LineColor(kRed));
      simPdf.plotOn(frame,Slice(category,"pass"),ProjWData(category,*data),LineColor(kGreen));
      simPdf.paramOn(frame,Layout(0.55));
      data->statOn(frame,Layout(0.65));
      return frame; 
    }
};

class GaussianPlusLinearFitter: public AbstractFitter{
  protected:
    double expectedSigma;
    RooRealVar sigma;
    RooGaussian gaussian;
    RooRealVar slopeAll;
    RooChebychev linearAll;
    RooRealVar slopePass;
    RooChebychev linearPass;
    RooAddPdf pdfAll;
    RooAddPdf pdfPass;
  public:
    GaussianPlusLinearFitter();
    ~GaussianPlusLinearFitter(){};
    GaussianPlusLinearFitter(double massMean, double massLow, double massHigh, double expectedSigma_):
      AbstractFitter(massMean, massLow, massHigh),
      expectedSigma(expectedSigma_),
      sigma("sigma","sigma",0.,10.*expectedSigma_,"GeV"),
      gaussian("gaussian","gaussian",mass,mean,sigma),
      slopeAll("slopeAll","slopeAll",0.,-1.,1.),
      linearAll("linearAll","linearAll",mass,slopeAll),
      slopePass("slopePass","slopePass",0.,-1.,1.),
      linearPass("linearPass","linearPass",mass,slopePass),
      pdfAll("pdfAll","pdfAll", RooArgList(gaussian,linearAll), RooArgList(nSignalAll,nBackgroundAll)),
      pdfPass("pdfPass","pdfPass", RooArgList(gaussian,linearPass), RooArgList(nSignalPass,nBackgroundPass))
    {
      simPdf.addPdf(pdfAll,"all");
      simPdf.addPdf(pdfPass,"pass");
    };
    void fit(TH1* pass, TH1* all){
      mean.setVal(expectedMean);
      sigma.setVal(expectedSigma);
      efficiency.setVal(pass->Integral()/all->Integral());
      nSignalAll.setVal(0.5*all->Integral());
      nBackgroundAll.setVal(0.5*all->Integral());
      nBackgroundPass.setVal(0.5*pass->Integral());
      slopeAll.setVal(0.);
      slopePass.setVal(0.);
//      delete data;
      data = new RooDataHist("data", "data", mass, Index(category), Import("all",*all), Import("pass",*pass) );
      simPdf.fitTo(*data);
      RooDataHist dataAll("all", "all", mass, all );
      RooDataHist dataPass("pass", "pass", mass, pass );
      chi2 = ( RooChi2Var("chi2All","chi2All",pdfAll,dataAll,DataError(RooAbsData::Poisson)).getVal()
        +RooChi2Var("chi2Pass","chi2Pass",pdfPass,dataPass,DataError(RooAbsData::Poisson)).getVal() )/(2*all->GetNbinsX()-8);
    }
};

class HeavyFlavorTnPHarvesting : public edm::EDAnalyzer{
  public:
    HeavyFlavorTnPHarvesting(const edm::ParameterSet& pset);
    virtual ~HeavyFlavorTnPHarvesting();
    virtual void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {};
    virtual void endJob();
  private:
    DQMStore * dqmStore;
    TFile * plots;
    string myDQMrootFolder;
    string numName;
    string denName;
    string effName;
    int massDimension;
    AbstractFitter *fitter;
};

HeavyFlavorTnPHarvesting::HeavyFlavorTnPHarvesting(const edm::ParameterSet& pset){
  myDQMrootFolder = pset.getUntrackedParameter<string>("MyDQMrootFolder");
  TString savePlotsInRootFileName = pset.getUntrackedParameter<string>("SavePlotsInRootFileName","");
  plots = savePlotsInRootFileName!="" ? new TFile(savePlotsInRootFileName,"recreate") : 0;
  numName = pset.getUntrackedParameter<string>("NumeratorMEname");
  denName = pset.getUntrackedParameter<string>("DenominatorMEname");
  effName = pset.getUntrackedParameter<string>("EfficiencyMEname");
  massDimension = pset.getUntrackedParameter<int>("MassDimension");
  string fitFunction = pset.getUntrackedParameter<string>("FitFunction");
  if(fitFunction=="GaussianPlusLinear"){
    fitter = new GaussianPlusLinearFitter(
      pset.getUntrackedParameter<double>("ExpectedMean"),
      pset.getUntrackedParameter<double>("FitRangeLow"),
      pset.getUntrackedParameter<double>("FitRangeHigh"),
      pset.getUntrackedParameter<double>("ExpectedSigma")
    );
  }else{
    LogError("HLTriggerOfflineHeavyFlavor") << "Fit function: "<<fitFunction<<" does not exist"<<endl;
  }
}

void HeavyFlavorTnPHarvesting::endJob(){
  dqmStore = Service<DQMStore>().operator->();
  if( !dqmStore ){
    LogError("HLTriggerOfflineHeavyFlavor") << "Could not find DQMStore service\n";
    return;
  }
  if(!fitter){
    return;
  }
  dqmStore->setCurrentFolder(myDQMrootFolder);
  TH1 *pass = dqmStore->get(myDQMrootFolder+"/"+numName)->getTH1();
  TH1 *all = dqmStore->get(myDQMrootFolder+"/"+denName)->getTH1();
  
  int dimensions = all->GetDimension();
  if( dimensions<2 || 3<dimensions ){
    LogError("HLTriggerOfflineHeavyFlavor") << "Monitoring Element "<<numName<<" has dimension "<<dimensions<<", different from 2 or 3"<<endl;
    return;
  }
  if(massDimension>dimensions){
    LogError("HLTriggerOfflineHeavyFlavor") << "Monitoring Element "<<numName<<" has smaller dimension than "<<massDimension<<endl;
    return;
  }
  TAxis *par1Axis, *par2Axis, *massAxis;
  int par1C, par2C, massC;
  switch(massDimension){
    case 1:
      massAxis = all->GetXaxis();
      massC = 1;
      par1Axis = all->GetYaxis();
      par1C = dimensions>1 ? massAxis->GetNbins()+2 : 0;
      par2Axis = all->GetZaxis();
      par2C = dimensions>2 ? (massAxis->GetNbins()+2)*(par1Axis->GetNbins()+2) : 0;
      break;
    case 2:
      par1Axis = all->GetXaxis();
      par1C = 1;
      massAxis = all->GetYaxis();
      massC = par1Axis->GetNbins()+2;
      par2Axis = all->GetZaxis();
      par2C = dimensions>2 ? (par1Axis->GetNbins()+2)*(massAxis->GetNbins()+2) : 0;
      break;
    case 3:
      par1Axis = all->GetXaxis();
      par1C = 1;
      par2Axis = all->GetYaxis();
      par2C = (par1Axis->GetNbins()+2);
      massAxis = all->GetZaxis();
      massC = (par1Axis->GetNbins()+2)*(par2Axis->GetNbins()+2);
      break;
    default:
      return;
  }
  int par2Cout = dimensions>2 ? par1Axis->GetNbins()+2 : 0;
  dqmStore->setCurrentFolder(myDQMrootFolder);
  MonitorElement * eff;
  MonitorElement * effChi2;
  switch(dimensions){
/*    case 1:
      TH1F* h1 = new TH1F("dummy","dummy",1,0,1);
      eff = dqmStore->book1D(effName,h1);
      effChi2 = dqmStore->book1D(effName+"Chi2",h1);
      delete h1;
      break;*/
    case 2:
      TH1F* h2;
      if(!par1Axis) return;
      if(par1Axis->GetXbins()->GetSize()==0){
        h2 = new TH1F("efficiency","efficiency",par1Axis->GetNbins(),par1Axis->GetXmin(),par1Axis->GetXmax());
      }else{
        h2 = new TH1F("efficiency","efficiency",par1Axis->GetNbins(),par1Axis->GetXbins()->GetArray());
      }
      eff = dqmStore->book1D(effName,h2);
      effChi2 = dqmStore->book1D(effName+"Chi2",h2);
      delete h2;
      break;
    case 3:
      TH2F* h3;
      if(!par1Axis || !par2Axis) return;
      if( par1Axis->GetXbins()->GetSize()==0 && par2Axis->GetXbins()->GetSize()==0 ){
        h3 = new TH2F("efficiency","efficiency",par1Axis->GetNbins(),par1Axis->GetXmin(),par1Axis->GetXmax(),par2Axis->GetNbins(),par2Axis->GetXmin(),par2Axis->GetXmax());
      }else if( par1Axis->GetXbins()->GetSize()==0 ){
        h3 = new TH2F("efficiency","efficiency",par1Axis->GetNbins(),par1Axis->GetXmin(),par1Axis->GetXmax(),par2Axis->GetNbins(),par2Axis->GetXbins()->GetArray());
      }else if( par2Axis->GetXbins()->GetSize()==0 ){
        h3 = new TH2F("efficiency","efficiency",par1Axis->GetNbins(),par1Axis->GetXbins()->GetArray(),par2Axis->GetNbins(),par2Axis->GetXmin(),par2Axis->GetXmax());
      }else{
        h3 = new TH2F("efficiency","efficiency",par1Axis->GetNbins(),par1Axis->GetXbins()->GetArray(),par2Axis->GetNbins(),par2Axis->GetXbins()->GetArray());
      }
      eff = dqmStore->book2D(effName,h3);
      effChi2 = dqmStore->book2D(effName+"Chi2",h3);
      delete h3;
      break;
    default:
      return;
  }
  TH1D * all1D;
  if( massAxis->GetXbins()->GetSize()==0 ){
    all1D = new TH1D("all1D","all1D",massAxis->GetNbins(),massAxis->GetXmin(),massAxis->GetXmax());
  }else{
    all1D = new TH1D("all1D","all1D",massAxis->GetNbins(),massAxis->GetXbins()->GetArray()); 
  }
  TH1D * pass1D = (TH1D *)all1D->Clone("pass1D");

  for(int par1=1; par1<=par1Axis->GetNbins(); par1++){
    for(int par2=1; par2<=par2Axis->GetNbins(); par2++){
      for(int mass=1; mass<=massAxis->GetNbins(); mass++){
        int index = par1*par1C + par2*par2C + mass*massC;
        all1D->SetBinContent(mass,all->GetBinContent(index));
        pass1D->SetBinContent(mass,pass->GetBinContent(index));
      }
      fitter->fit( pass1D, all1D );
      int index = par1 + par2*par2Cout;
      eff->setBinContent( index, fitter->getEfficiency() );
      eff->setBinError( index, fitter->getEfficiencyError() );
      effChi2->setBinContent( index, fitter->getChi2() );
      if(plots){
        plots->cd();
        fitter->savePlot( TString::Format("%s_%d_%d",effName.c_str(),par1,par2) );
      }
    }
  }
}

HeavyFlavorTnPHarvesting::~HeavyFlavorTnPHarvesting(){
  delete fitter;
  if(plots) plots->Write();
}

//define this as a plug-in
DEFINE_FWK_MODULE(HeavyFlavorTnPHarvesting);
