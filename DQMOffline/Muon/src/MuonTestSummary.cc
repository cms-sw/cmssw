
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/10/27 16:28:51 $
 *  $Revision: 1.1 $
 *  \author G. Mila - INFN Torino
 */


#include <DQMOffline/Muon/src/MuonTestSummary.h>

// Framework
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>

using namespace edm;
using namespace std;


MuonTestSummary::MuonTestSummary(const edm::ParameterSet& ps){

  dbe = Service<DQMStore>().operator->();

  // parameter initialization for kinematics test
  etaExpected = ps.getParameter<double>("etaExpected");
  phiExpected = ps.getParameter<double>("phiExpected");
  etaSpread = ps.getParameter<double>("etaSpread");
  phiSpread = ps.getParameter<double>("phiSpread");
  chi2Fraction = ps.getParameter<double>("chi2Fraction");
  chi2Spread = ps.getParameter<double>("chi2Spread");
  resEtaSpread_tkGlb = ps.getParameter<double>("resEtaSpread_tkGlb");
  resEtaSpread_glbSta = ps.getParameter<double>("resEtaSpread_glbSta");
  resPhiSpread_tkGlb = ps.getParameter<double>("resPhiSpread_tkGlb");
  resPhiSpread_glbSta = ps.getParameter<double>("resPhiSpread_glbSta");
  numMatchedExpected = ps.getParameter<double>("numMatchedExpected");
  sigmaResSegmTrackExp = ps.getParameter<double>("sigmaResSegmTrackExp");
}

MuonTestSummary::~MuonTestSummary(){}

void MuonTestSummary::beginJob(const edm::EventSetup& context){

  metname = "muonTestSummary";
  LogTrace(metname)<<"[MuonTestSummary] Histo booking";

  // book the summary histos
  dbe->setCurrentFolder("Muons/TestSummary"); 

  // kinematics test report
  kinematicsSummaryMap = dbe->book2D("kinematicsSummaryMap","Kinematics test summary",5,1,6,3,1,4);
  kinematicsSummaryMap->setAxisTitle("track monitored",1);
  kinematicsSummaryMap->setBinLabel(1,"GLB",1);
  kinematicsSummaryMap->setBinLabel(2,"TKfromGLB",1);
  kinematicsSummaryMap->setBinLabel(3,"STAfromGLB",1);
  kinematicsSummaryMap->setBinLabel(4,"TK",1);
  kinematicsSummaryMap->setBinLabel(5,"STA",1);
  kinematicsSummaryMap->setAxisTitle("parameter tested",2);
  kinematicsSummaryMap->setBinLabel(1,"#chi_{2}",2);
  kinematicsSummaryMap->setBinLabel(2,"#eta",2);
  kinematicsSummaryMap->setBinLabel(3,"#phi",2);

  // residuals test report
  residualsSummaryMap = dbe->book2D("residualsSummaryMap","Residuals test summary",3,1,4,2,1,3);
  residualsSummaryMap->setAxisTitle("residuals",1);
  residualsSummaryMap->setBinLabel(1,"TK-GLB",1);
  residualsSummaryMap->setBinLabel(2,"GLB-STA",1);
  residualsSummaryMap->setBinLabel(3,"TK-STA",1);
  residualsSummaryMap->setAxisTitle("parameter tested",2);
  residualsSummaryMap->setBinLabel(1,"#eta",2);
  residualsSummaryMap->setBinLabel(2,"#phi",2);

  // muonId test report
  muonIdSummaryMap = dbe->book1D("muonIdSummaryMap","muonId test summary",3,1,4);
  muonIdSummaryMap->setAxisTitle("test");
  muonIdSummaryMap->setBinLabel(1,"#matchCh");
  muonIdSummaryMap->setBinLabel(2,"#assSegm");
  muonIdSummaryMap->setBinLabel(3,"resTrackSegm");

}

void MuonTestSummary::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {


  // fill the kinematics report summary
  doKinematicsTests("GlbMuon_Glb_", 1, 1.0/3.0);
  doKinematicsTests("GlbMuon_Tk_", 2, 1.0/3.0);
  doKinematicsTests("GlbMuon_Sta_", 3, 1.0/3.0);
  doKinematicsTests("TkMuon_", 4, 1);
  doKinematicsTests("StaMuon_", 5, 1);

  // fill the residuals report summary
  doResidualsTests("TkGlb", "eta", 1);
  doResidualsTests("GlbSta", "eta", 2);
  doResidualsTests("TkSta", "eta", 3);
  doResidualsTests("TkGlb", "phi", 1);
  doResidualsTests("GlbSta", "phi", 2);
  doResidualsTests("TkSta", "phi", 3);

  // fill the muonID report summary
  doMuonIDTests();

 }


void MuonTestSummary::doKinematicsTests(string muonType, int bin, double weight){
  

  // chi2 test
  string path = "Muons/MuonRecoAnalyzer/" + muonType + "chi2OverDf";
  MonitorElement * chi2Histo = dbe->get(path);

  if(chi2Histo){

    TH1F * chi2Histo_root = chi2Histo->getTH1F();
    int maxBin = chi2Histo_root->GetMaximumBin();
    double fraction = double(chi2Histo_root->Integral(1,maxBin))/double(chi2Histo_root->Integral(maxBin+1,chi2Histo_root->GetNbinsX()));
    LogTrace(metname)<<"chi2 fraction for "<<muonType<<" : "<<fraction<<endl;
    if(fraction>(chi2Fraction-chi2Spread) && fraction<(chi2Fraction+chi2Spread))
      kinematicsSummaryMap->setBinContent(bin,1,weight);
    else
      kinematicsSummaryMap->setBinContent(bin,1,0);
  }



  // pseudorapidity test
  path = "Muons/MuonRecoAnalyzer/" + muonType + "eta";
  MonitorElement * etaHisto = dbe->get(path);
  
  if(etaHisto){

    TH1F * etaHisto_root = etaHisto->getTH1F();
    double binSize = (etaHisto_root->GetXaxis()->GetXmax()-etaHisto_root->GetXaxis()->GetXmin())/etaHisto_root->GetNbinsX();
    int binZero = int((0-etaHisto_root->GetXaxis()->GetXmin())/binSize);
    double symmetryFactor = 
      double(etaHisto_root->Integral(1,binZero-1)) / double(etaHisto_root->Integral(binZero,etaHisto_root->GetNbinsX()));
    LogTrace(metname)<<"eta symmetryFactor for "<<muonType<<" : "<<symmetryFactor<<endl;
    if (symmetryFactor>(etaExpected-etaSpread) && symmetryFactor<(etaExpected+etaSpread))
      kinematicsSummaryMap->setBinContent(bin,2,weight);
    else
      kinematicsSummaryMap->setBinContent(bin,2,0);
  }


  // phi test
  path = "Muons/MuonRecoAnalyzer/" + muonType + "phi";
  MonitorElement * phiHisto = dbe->get(path);

  if(phiHisto){

    TH1F * phiHisto_root = phiHisto->getTH1F();
    double binSize = (phiHisto_root->GetXaxis()->GetXmax()-phiHisto_root->GetXaxis()->GetXmin())/phiHisto_root->GetNbinsX();
    int binZero = int((0-phiHisto_root->GetXaxis()->GetXmin())/binSize);
    double symmetryFactor = 
      double(phiHisto_root->Integral(binZero+1,phiHisto_root->GetNbinsX())) / double(phiHisto_root->Integral(1,binZero));
    LogTrace(metname)<<"phi symmetryFactor for "<<muonType<<" : "<<symmetryFactor<<endl;
    if (symmetryFactor>(phiExpected-phiSpread) && symmetryFactor<(phiExpected+phiSpread))
      kinematicsSummaryMap->setBinContent(bin,3,weight);
    else
	kinematicsSummaryMap->setBinContent(bin,3,0);
  }

}




void MuonTestSummary::doResidualsTests(string type, string parameter, int bin){

  // residuals test
  string path = "Muons/MuonRecoAnalyzer/Res_" + type + "_" + parameter;
  MonitorElement * residualsHisto = dbe->get(path);
  
  if(residualsHisto){
  
    // Gaussian Fit
    float statMean = residualsHisto->getMean(1);
    float statSigma = residualsHisto->getRMS(1);
    Double_t mean = -1;
    Double_t sigma = -1;
    TH1F * histo_root = residualsHisto->getTH1F();
    if(histo_root->GetEntries()>20){
      TF1 *gfit = new TF1("Gaussian","gaus",(statMean-(2*statSigma)),(statMean+(2*statSigma)));
      try {
	histo_root->Fit(gfit);
      } catch (...) {
	edm::LogError (metname)<< "[MuonTestSummary]: Exception when fitting Res_"<<type<<"_"<<parameter;
      }
      if(gfit){
	mean = gfit->GetParameter(1); 
	sigma = gfit->GetParameter(2);
	LogTrace(metname)<<"mean: "<<mean<<endl;
	LogTrace(metname)<<"sigma: "<<sigma<<endl;
      }
    }
    else{
      LogTrace(metname) << "[MuonTestSummary]: Test of  Res_"<<type<<"_"<<parameter<< " not performed because # entries < 20 ";
    }

    if(sigma!=-1 && parameter=="eta" && type=="TkGlb" && sigma<resEtaSpread_tkGlb)
      residualsSummaryMap->setBinContent(bin, 1, 1.0/2.0);
    if(sigma!=-1 && parameter=="eta" && (type=="GlbSta" || type=="TkSta") && sigma<resEtaSpread_glbSta)
      residualsSummaryMap->setBinContent(bin, 1, 1.0/2.0);
    if(sigma!=-1 && parameter=="phi" && type=="TkGlb" && sigma<resPhiSpread_tkGlb)
      residualsSummaryMap->setBinContent(bin, 2, 1.0/2.0);     
    if(sigma!=-1 && parameter=="phi" && (type=="GlbSta" || type=="TkSta") && sigma<resPhiSpread_glbSta)
      residualsSummaryMap->setBinContent(bin, 2, 1.0/2.0); 
  }

}

void MuonTestSummary::doMuonIDTests(){

  // num matches test
  string path = "Muons/MuonIdDQM/TrackerMuons/hNumMatches";
  MonitorElement * matchesHisto = dbe->get(path);

  if(matchesHisto){
    TH1F * matchesHisto_root = matchesHisto->getTH1F();
    if(matchesHisto_root->GetMaximumBin() == numMatchedExpected || matchesHisto_root->GetMaximumBin() == numMatchedExpected+1)
      muonIdSummaryMap->setBinContent(1,1.0/3.0);
    else
      muonIdSummaryMap->setBinContent(1,0);
  }

  
  // num of associated segments (limits computed from simulated data)
  double numOneSegm_dt = 0;
  MonitorElement * DT1Histo = dbe->get("Muons/MuonIdDQM/TrackerMuons/hDT1NumSegments");
  if(DT1Histo) numOneSegm_dt+=DT1Histo->getBinContent(2);
  MonitorElement * DT2Histo = dbe->get("Muons/MuonIdDQM/TrackerMuons/hDT2NumSegments");
  if(DT2Histo) numOneSegm_dt+=DT2Histo->getBinContent(2);
  MonitorElement * DT3Histo = dbe->get("Muons/MuonIdDQM/TrackerMuons/hDT3NumSegments");
  if(DT3Histo) numOneSegm_dt+=DT3Histo->getBinContent(2);
  MonitorElement * DT4Histo = dbe->get("Muons/MuonIdDQM/TrackerMuons/hDT4NumSegments"); 
  if(DT4Histo) numOneSegm_dt+=DT4Histo->getBinContent(2);
  double fraction_dt = double(DT1Histo->getEntries())/numOneSegm_dt;
  LogTrace(metname)<<"numOneSegm_dt: "<<numOneSegm_dt<<endl;
  LogTrace(metname)<<"fraction_dt: "<<fraction_dt<<endl;
  
  double numOneSegm_csc = 0;
  MonitorElement * CSC1Histo = dbe->get("Muons/MuonIdDQM/TrackerMuons/hCSC1NumSegments");
  if(CSC1Histo) numOneSegm_csc+=CSC1Histo->getBinContent(2);
  MonitorElement * CSC2Histo = dbe->get("Muons/MuonIdDQM/TrackerMuons/hCSC2NumSegments");
  if(CSC2Histo) numOneSegm_csc+=CSC2Histo->getBinContent(2);
  MonitorElement * CSC3Histo = dbe->get("Muons/MuonIdDQM/TrackerMuons/hCSC3NumSegments");
  if(CSC3Histo) numOneSegm_csc+=CSC3Histo->getBinContent(2);
  MonitorElement * CSC4Histo = dbe->get("Muons/MuonIdDQM/TrackerMuons/hCSC4NumSegments");
  if(CSC4Histo) numOneSegm_csc+=CSC4Histo->getBinContent(2);
  double fraction_csc = double(CSC1Histo->getEntries())/numOneSegm_csc;
  LogTrace(metname)<<"numOneSegm_csc: "<<numOneSegm_csc<<endl;
  LogTrace(metname)<<"fraction_csc: "<<fraction_csc<<endl;

  if((fraction_dt>0.7 && fraction_dt<0.8) && (fraction_csc>0.57 && fraction_csc<0.67))
    muonIdSummaryMap->setBinContent(2,1.0/3.0);
  else{
    if((fraction_dt>0.7 && fraction_dt<0.8) || (fraction_csc>0.57 && fraction_csc<0.67))
      muonIdSummaryMap->setBinContent(2,1.0/6.0);
    else
      muonIdSummaryMap->setBinContent(2,0);
  }

  // residuals test
  vector<string> resHistos;
  resHistos.push_back("hDT1Pullx");
  resHistos.push_back("hDT2Pullx");
  resHistos.push_back("hDT3Pullx");
  resHistos.push_back("hDT4Pullx");
  resHistos.push_back("hDT1Pully");
  resHistos.push_back("hDT2Pully");
  resHistos.push_back("hDT3Pully");
  resHistos.push_back("hCSC1Pullx");
  resHistos.push_back("hCSC2Pullx");
  resHistos.push_back("hCSC3Pullx");
  resHistos.push_back("hCSC4Pullx");
  resHistos.push_back("hCSC1Pully");
  resHistos.push_back("hCSC2Pully");
  resHistos.push_back("hCSC3Pully");
  resHistos.push_back("hCSC4Pully");

  double dtSigma=0; 
  double cscSigma=0;
  for(int name=0; name<=14; name++){   
    MonitorElement * resHisto = dbe->get("Muons/MuonIdDQM/TrackerMuons/"+resHistos[name]);
    if(resHisto){
 
      TH1F * resHisto_root = resHisto->getTH1F();
      if(resHisto_root->GetEntries()>20){
	TF1 *gfit = new TF1("Gaussian","gaus",-2,2);
	try {
	  resHisto_root->Fit(gfit);
	} catch (...) {
	  edm::LogError (metname)<< "[MuonTestSummary]: Exception when fitting "<<resHistos[name];
	}
	if(gfit){
	  double mean = gfit->GetParameter(1); 
	  double sigma = gfit->GetParameter(2);
	  LogTrace(metname)<<"meanRes: "<<mean<<" for "<<resHistos[name]<<endl;
	  LogTrace(metname)<<"sigmaRes: "<<sigma<<" for "<<resHistos[name]<<endl;
	  if(name<=6) dtSigma+=sigma;
	  else cscSigma+=sigma;
 
	}
      }
      else{
      LogTrace(metname) << "[MuonTestSummary]: Test of "<<resHistos[name]<< " not performed because # entries < 20 ";
    }

    }
  } // loop over residuals histos
  
  if((dtSigma>((7*sigmaResSegmTrackExp)-0.5) && dtSigma<(7*sigmaResSegmTrackExp)+0.5) 
     && ((cscSigma>(8*sigmaResSegmTrackExp)-0.5) && cscSigma<(8*sigmaResSegmTrackExp)+0.5))
    muonIdSummaryMap->setBinContent(3,1.0/3.0);
  else{
    if((dtSigma>((7*sigmaResSegmTrackExp)-0.5) && dtSigma<(7*sigmaResSegmTrackExp)+0.5) 
     || ((cscSigma>(8*sigmaResSegmTrackExp)-0.5) && cscSigma<(8*sigmaResSegmTrackExp)+0.5))
      muonIdSummaryMap->setBinContent(3,1.0/6.0);
    else
      muonIdSummaryMap->setBinContent(3,0);
  }

}



  

  
