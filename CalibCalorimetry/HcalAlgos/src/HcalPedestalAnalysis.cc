
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"

#include <CalibCalorimetry/HcalAlgos/interface/HcalPedestalAnalysis.h>
#include "CalibCalorimetry/HcalAlgos/interface/HcalAlgoUtils.h"

using namespace std;

HcalPedestalAnalysis::HcalPedestalAnalysis(const edm::ParameterSet& ps)
  : pedCan (0),
    widthCan (0),
    pedCan_nominal (0),
    widthCan_nominal (0),
    meansper2caps (0),
    widthsper2caps (0)
{
  // init
  evt=0;
  sample=0;
  m_file=0;
  // output files
  for(int k=0;k<4;k++) state.push_back(true); // 4 capid
  m_outputFileMean = ps.getUntrackedParameter<string>("outputFileMeans", "");
  if ( m_outputFileMean.size() != 0 ) {
    cout << "Hcal pedestal means will be saved to " << m_outputFileMean.c_str() << endl;
  } 
  m_outputFileWidth = ps.getUntrackedParameter<string>("outputFileWidths", "");
  if ( m_outputFileWidth.size() != 0 ) {
    cout << "Hcal pedestal widths will be saved to " << m_outputFileWidth.c_str() << endl;
  } 
  m_outputFileROOT = ps.getUntrackedParameter<string>("outputFileHist", "");
  if ( m_outputFileROOT.size() != 0 ) {
    cout << "Hcal pedestal histograms will be saved to " << m_outputFileROOT.c_str() << endl;
  } 
  m_logFile.open("HcalPedestalAnalysis.log");
  m_startSample = ps.getUntrackedParameter<int>("firstSample", 0);
  if(m_startSample<0) m_startSample=0;
  m_endSample = ps.getUntrackedParameter<int>("lastSample", 19);

  // histogram booking
  hbHists.ALLPEDS = new TH1F("HB/HE All Pedestals","HB/HE All Peds",10,0,9);
  hbHists.PEDRMS= new TH1F("HB/HE All Pedestal RMS","HB/HE All Pedestal RMS",100,0,3);
  hbHists.PEDMEAN= new TH1F("HB/HE All Pedestal Means","HB/HE All Pedestal Means",100,0,9);
  hbHists.CHI2= new TH1F("HB/HE Chi2/ndf for whole range Gauss fit","HB/HE Chi2/ndf Gauss",200,0.,50.);

  hoHists.ALLPEDS = new TH1F("HO All Pedestals","HO All Peds",10,0,9);
  hoHists.PEDRMS= new TH1F("HO All Pedestal RMS","HO All Pedestal RMS",100,0,3);
  hoHists.PEDMEAN= new TH1F("HO All Pedestal Means","HO All Pedestal Means",100,0,9);
  hoHists.CHI2= new TH1F("HO Chi2/ndf for whole range Gauss fit","HO Chi2/ndf Gauss",200,0.,50.);

  hfHists.ALLPEDS = new TH1F("HF All Pedestals","HF All Peds",10,0,9);
  hfHists.PEDRMS= new TH1F("HF All Pedestal RMS","HF All Pedestal RMS",100,0,3);
  hfHists.PEDMEAN= new TH1F("HF All Pedestal Means","HF All Pedestal Means",100,0,9);
  hfHists.CHI2= new TH1F("HF Chi2/ndf for whole range Gauss fit","HF Chi2/ndf Gauss",200,0.,50.);
}

//-----------------------------------------------------------------------------
HcalPedestalAnalysis::~HcalPedestalAnalysis(){
  ///All done, clean up!!
  for(_meot=hbHists.PEDTRENDS.begin(); _meot!=hbHists.PEDTRENDS.end(); _meot++){
    for(int i=0; i<4; i++) _meot->second[i].first->Delete();
  }
  for(_meot=hoHists.PEDTRENDS.begin(); _meot!=hoHists.PEDTRENDS.end(); _meot++){
    for(int i=0; i<4; i++) _meot->second[i].first->Delete();
  }
  for(_meot=hfHists.PEDTRENDS.begin(); _meot!=hfHists.PEDTRENDS.end(); _meot++){
    for(int i=0; i<4; i++) _meot->second[i].first->Delete();
  }
  hbHists.ALLPEDS->Delete();
  hbHists.PEDRMS->Delete();
  hbHists.PEDMEAN->Delete();
  hbHists.CHI2->Delete();

  hoHists.ALLPEDS->Delete();
  hoHists.PEDRMS->Delete();
  hoHists.PEDMEAN->Delete();
  hoHists.CHI2->Delete();

  hfHists.ALLPEDS->Delete();
  hfHists.PEDRMS->Delete();
  hfHists.PEDMEAN->Delete();
  hfHists.CHI2->Delete();
}

//-----------------------------------------------------------------------------
void HcalPedestalAnalysis::setup(const std::string& m_outputFileROOT) {
  // open the histogram file, create directories within
  m_file=new TFile(m_outputFileROOT.c_str(),"RECREATE");
  m_file->mkdir("HB");
  m_file->cd();
  m_file->mkdir("HO");
  m_file->cd();
  m_file->mkdir("HF");
  m_file->cd();
}

//-----------------------------------------------------------------------------
void HcalPedestalAnalysis::GetPedConst(map<HcalDetId, map<int,PEDBUNCH> > &toolT){
  double mean; double rms; double param2[4]; double param1[4]; double dsigma;
  double dparam2[4]; double dparam1[4]; double chi2; double sigma;

  for(_meot=toolT.begin(); _meot!=toolT.end(); _meot++){
    for(int i=0; i<4; i++){
      if(fitflag>0){
        _meot->second[i].first->Fit("gaus","Q");
        TF1 *fit = _meot->second[i].first->GetFunction("gaus");
        chi2=0;
        if(fit->GetNDF()!=0){
        chi2=fit->GetChisquare()/fit->GetNDF();
        }
        if(chi2>10.){
          _meot->second[i].first->Fit("gaus","Q","",0.,fit->GetParameter(1)+fit->GetParameter(2));
        }
        fit = _meot->second[i].first->GetFunction("gaus");
        param1[i]=fit->GetParameter(1);
        dparam1[i]=fit->GetParError(1);
        param2[i]=fit->GetParameter(2);
        dparam2[i]=fit->GetParError(2);
      }
      else{
        param1[i] = _meot->second[i].first->GetMean();
        param2[i] = _meot->second[i].first->GetRMS();
//        dparam1[i] = param2[i]/sqrt(_meot->second[i].first->GetEntries());
        dparam1[i] = param2[i]/sqrt(nevt_ped*2.5);
        dparam2[i] = dparam1[i]*param2[i]/param1[i];
        chi2=0.;
      }
      _meot->second[i].first->GetXaxis()->SetTitle("Charge, fC");
      _meot->second[i].first->GetYaxis()->SetTitle("CapID samplings");
      _meot->second[i].first->Write();
      _meot->second[i].second.first[0].push_back(param1[i]);
      _meot->second[i].second.first[1].push_back(dparam1[i]);
      _meot->second[i].second.first[2].push_back(param2[i]);
      _meot->second[i].second.first[3].push_back(dparam2[i]);
      _meot->second[i].second.first[4].push_back(chi2);
    }
// Michal's addition
    for(int i=0; i<4; i++){
      _meot->second[i+4].first->GetXaxis()->SetTitle("Charge^2, fC^2");
      _meot->second[i+4].first->GetYaxis()->SetTitle("2-CapID samplings");
      _meot->second[i+4].first->Write();
      _meot->second[i+8].first->GetXaxis()->SetTitle("Charge^2, fC^2");
      _meot->second[i+8].first->GetYaxis()->SetTitle("2-CapID samplings");
      _meot->second[i+8].first->Write();
      _meot->second[i+12].first->GetXaxis()->SetTitle("Charge^2, fC^2");
      _meot->second[i+12].first->GetYaxis()->SetTitle("2-CapID samplings");
      _meot->second[i+12].first->Write();
// here calculate the correlation coefficients between cap-ids
// errors on coefficients are grossly approximative
      mean = _meot->second[i+4].first->GetMean();
      rms = _meot->second[i+4].first->GetRMS();
      sigma = mean-param1[i]*param1[(i+1)%4];
      float stats = _meot->second[i+4].first->GetEntries();
      dsigma = param1[i]*dparam1[(i+1)%4]*param1[i]*dparam1[(i+1)%4]+param1[(i+1)%4]*dparam1[i]*param1[(i+1)%4]*dparam1[i];      dsigma = 167./stats*sqrt(dsigma+rms/sqrt(stats)*rms/sqrt(stats));
      _meot->second[i].second.first[5].push_back(sigma);
      _meot->second[i].second.first[6].push_back(dsigma);
      mean = _meot->second[i+8].first->GetMean();
      rms = _meot->second[i+8].first->GetRMS();
      sigma = mean-param1[i]*param1[(i+2)%4];
      stats = _meot->second[i+8].first->GetEntries();
      dsigma = param1[i]*dparam1[(i+2)%4]*param1[i]*dparam1[(i+2)%4]+param1[(i+2)%4]*dparam1[i]*param1[(i+2)%4]*dparam1[i];      dsigma = 167./stats*sqrt(dsigma+rms/sqrt(stats)*rms/sqrt(stats));
      _meot->second[i].second.first[7].push_back(sigma);
      _meot->second[i].second.first[8].push_back(dsigma);
      mean = _meot->second[i+12].first->GetMean();
      rms = _meot->second[i+12].first->GetRMS();
      sigma = mean-param1[i]*param1[(i+3)%4];
      stats = _meot->second[i+12].first->GetEntries();
      dsigma = param1[i]*dparam1[(i+3)%4]*param1[i]*dparam1[(i+3)%4]+param1[(i+3)%4]*dparam1[i]*param1[(i+3)%4]*dparam1[i];      dsigma = 167./stats*sqrt(dsigma+rms/sqrt(stats)*rms/sqrt(stats));
      _meot->second[i].second.first[9].push_back(sigma);
      _meot->second[i].second.first[10].push_back(dsigma);
    }
  }
}

//-----------------------------------------------------------------------------
void HcalPedestalAnalysis::SampleAnalysis(){
  // it is called every nevt_ped events (a sample) and the end of run
  char PedSampleNum[20];

  sprintf(PedSampleNum,"PedSample_%d",sample);
  m_file->cd();
  m_file->mkdir(PedSampleNum);
  m_file->cd(PedSampleNum);

// Compute pedestal constants for each HB/HE, HO, HF
  GetPedConst(hbHists.PEDTRENDS);
  GetPedConst(hoHists.PEDTRENDS);
  GetPedConst(hfHists.PEDTRENDS);
}

//-----------------------------------------------------------------------------
void HcalPedestalAnalysis::Trendings(map<HcalDetId, map<int,PEDBUNCH> > &toolT,
TH1F* Chi2, TH1F* CapidAverage, TH1F* CapidChi2){
  map<int, std::vector<double> > AverageValues;

  for(_meot=toolT.begin(); _meot!=toolT.end(); _meot++){
    for(int i=0; i<4; i++){
      char name[1024];
      HcalDetId detid = _meot->first;
      sprintf(name,"Pedestal trend, ieta=%d iphi=%d depth=%d CAPID=%d",detid.ieta(),detid.iphi(),detid.depth(),i);
      int bins = _meot->second[i].second.first[0].size();       float lo =0.;       float hi = (float)bins;
      _meot->second[i].second.second.push_back(new TH1F(name,name,bins,lo,hi));
      sprintf(name,"Pedestal sigma trend, ieta=%d iphi=%d depth=%d CAPID=%d",detid.ieta(),detid.iphi(),detid.depth(),i);
      bins = _meot->second[i].second.first[2].size();
      hi = (float)bins;
      _meot->second[i].second.second.push_back(new TH1F(name,name,bins,lo,hi));
      sprintf(name,"Correlation coeff trend, ieta=%d iphi=%d depth=%d CAPIDs=%d*%d",detid.ieta(),detid.iphi(),detid.depth(),i,(i+1)%4);
      bins = _meot->second[i].second.first[5].size();
      hi = (float)bins;
      _meot->second[i].second.second.push_back(new TH1F(name,name,bins,lo,hi));
      sprintf(name,"Correlation coeff trend, ieta=%d iphi=%d depth=%d CAPIDs=%d*%d",detid.ieta(),detid.iphi(),detid.depth(),i,(i+2)%4);
      bins = _meot->second[i].second.first[7].size();
      hi = (float)bins;
      _meot->second[i].second.second.push_back(new TH1F(name,name,bins,lo,hi));
      sprintf(name,"Correlation coeff trend, ieta=%d iphi=%d depth=%d CAPIDs=%d*%d",detid.ieta(),detid.iphi(),detid.depth(),i,(i+3)%4);
      bins = _meot->second[i].second.first[9].size();
      hi = (float)bins;
      _meot->second[i].second.second.push_back(new TH1F(name,name,bins,lo,hi));
                                                                                
      std::vector<double>::iterator sample_it;
      // Pedestal mean - put content and errors
      int j=0;
      for(sample_it=_meot->second[i].second.first[0].begin();
          sample_it!=_meot->second[i].second.first[0].end();sample_it++){
        _meot->second[i].second.second[0]->SetBinContent(++j,*sample_it);
      }
      j=0;
      for(sample_it=_meot->second[i].second.first[1].begin();
          sample_it!=_meot->second[i].second.first[1].end();sample_it++){
        _meot->second[i].second.second[0]->SetBinError(++j,*sample_it);
      }
      // fit with a constant - extract parameters
      _meot->second[i].second.second[0]->Fit("pol0","Q");
      TF1 *fit = _meot->second[i].second.second[0]->GetFunction("pol0");
      AverageValues[0].push_back(fit->GetParameter(0));
      AverageValues[1].push_back(fit->GetParError(0));
      if(sample>1)
      AverageValues[2].push_back(fit->GetChisquare()/fit->GetNDF());
      else
      AverageValues[2].push_back(fit->GetChisquare());
      sprintf(name,"Samples (%d events nominal)",nevt_ped);
      _meot->second[i].second.second[0]->GetXaxis()->SetTitle(name);
      _meot->second[i].second.second[0]->GetYaxis()->SetTitle("Pedestal value");      _meot->second[i].second.second[0]->Write();
      // Pedestal sigma - put content and errors
      j=0;
      for(sample_it=_meot->second[i].second.first[2].begin();
          sample_it!=_meot->second[i].second.first[2].end();sample_it++){
        _meot->second[i].second.second[1]->SetBinContent(++j,*sample_it);
      }
      j=0;
      for(sample_it=_meot->second[i].second.first[3].begin();
          sample_it!=_meot->second[i].second.first[3].end();sample_it++){
        _meot->second[i].second.second[1]->SetBinError(++j,*sample_it);
      }
      _meot->second[i].second.second[1]->GetXaxis()->SetTitle(name);
      _meot->second[i].second.second[1]->GetYaxis()->SetTitle("Pedestal width");      _meot->second[i].second.second[1]->Write();
      // Correlation coeffs - put contents and errors (Michal)
      j=0;
      for(sample_it=_meot->second[i].second.first[5].begin();
          sample_it!=_meot->second[i].second.first[5].end();sample_it++){
        _meot->second[i].second.second[2]->SetBinContent(++j,*sample_it);
      }
      j=0;
      for(sample_it=_meot->second[i].second.first[6].begin();
          sample_it!=_meot->second[i].second.first[6].end();sample_it++){
        _meot->second[i].second.second[2]->SetBinError(++j,*sample_it);
      }
      _meot->second[i].second.second[2]->GetXaxis()->SetTitle(name);
      _meot->second[i].second.second[2]->GetYaxis()->SetTitle("Correlation");
      _meot->second[i].second.second[2]->Write();
      j=0;
      for(sample_it=_meot->second[i].second.first[7].begin();
          sample_it!=_meot->second[i].second.first[7].end();sample_it++){
        _meot->second[i].second.second[3]->SetBinContent(++j,*sample_it);
      }
      j=0;
      for(sample_it=_meot->second[i].second.first[8].begin();
          sample_it!=_meot->second[i].second.first[8].end();sample_it++){
        _meot->second[i].second.second[3]->SetBinError(++j,*sample_it);
      }
      _meot->second[i].second.second[3]->GetXaxis()->SetTitle(name);
      _meot->second[i].second.second[3]->GetYaxis()->SetTitle("Correlation");
      _meot->second[i].second.second[3]->Write();
      j=0;
      for(sample_it=_meot->second[i].second.first[9].begin();
          sample_it!=_meot->second[i].second.first[9].end();sample_it++){
        _meot->second[i].second.second[4]->SetBinContent(++j,*sample_it);
      }
      j=0;
      for(sample_it=_meot->second[i].second.first[10].begin();
          sample_it!=_meot->second[i].second.first[10].end();sample_it++){
        _meot->second[i].second.second[4]->SetBinError(++j,*sample_it);
      }
      _meot->second[i].second.second[4]->GetXaxis()->SetTitle(name);
      _meot->second[i].second.second[4]->GetYaxis()->SetTitle("Correlation");
      _meot->second[i].second.second[4]->Write();
      // chi2
      j=0;
      for(sample_it=_meot->second[i].second.first[4].begin();
          sample_it!=_meot->second[i].second.first[4].end();sample_it++){
        Chi2->Fill(*sample_it);
      }
    }
  }
  CapidAverage= new TH1F("HB/HE constant fit: Pedestal Values",
                         "HB/HE constant fit: Pedestal Values",
                         AverageValues[0].size(),0.,AverageValues[0].size());
  std::vector<double>::iterator sample_it;
  int j=0;
  for(sample_it=AverageValues[0].begin();
      sample_it!=AverageValues[0].end();sample_it++){
    CapidAverage->SetBinContent(++j,*sample_it);
  }
  j=0;
  for(sample_it=AverageValues[1].begin();
      sample_it!=AverageValues[1].end();sample_it++){
    CapidAverage->SetBinError(++j,*sample_it);
  }
  CapidChi2= new TH1F("HB/HE constant fit: Chi2/ndf",
                      "HB/HE constant fit: Chi2/ndf",
                      AverageValues[2].size(),0.,AverageValues[2].size());
  j=0;
  for(sample_it=AverageValues[2].begin();
      sample_it!=AverageValues[2].end();sample_it++){
    CapidChi2->SetBinContent(++j,*sample_it);
    //CapidChi2->SetBinError(++j,0);
  }
  Chi2->GetXaxis()->SetTitle("Chi2/ndf");
  Chi2->GetYaxis()->SetTitle("50 x [(16+2) x 4 x 4] `events`");
  Chi2->Write();
  CapidAverage->GetYaxis()->SetTitle("Pedestal value");
  CapidAverage->GetXaxis()->SetTitle("(16+2) x 4 x 4 `events`");
  CapidAverage->Write();
  CapidChi2->GetYaxis()->SetTitle("Chi2/ndf");
  CapidChi2->GetXaxis()->SetTitle("(16+2) x 4 x 4 `events`");
  CapidChi2->Write();

}

//-----------------------------------------------------------------------------
int HcalPedestalAnalysis::PedValidtn(map<HcalDetId, map<int,PEDBUNCH> > &toolT)
{
  int PedsOK=1;
  for(_meot=toolT.begin(); _meot!=toolT.end(); _meot++){
    float cap_new[4]; float sig_new[4][4];
    float dcap_new[4]; float dsig_new[4][4];
    float cap_nom[4]; float sig_nom[4][4];
    if(fitflag>0){
      cap_new[0]=_meot->second[0].first->GetFunction("gaus")->GetParameter(1);
      cap_new[1]=_meot->second[1].first->GetFunction("gaus")->GetParameter(1);
      cap_new[2]=_meot->second[2].first->GetFunction("gaus")->GetParameter(1);
      cap_new[3]=_meot->second[3].first->GetFunction("gaus")->GetParameter(1);
      dcap_new[0]=_meot->second[0].first->GetFunction("gaus")->GetParError(1);
      dcap_new[1]=_meot->second[1].first->GetFunction("gaus")->GetParError(1);
      dcap_new[2]=_meot->second[2].first->GetFunction("gaus")->GetParError(1);
      dcap_new[3]=_meot->second[3].first->GetFunction("gaus")->GetParError(1);
      sig_new[0][0]=_meot->second[0].first->GetFunction("gaus")->GetParameter(2);
      sig_new[1][1]=_meot->second[1].first->GetFunction("gaus")->GetParameter(2);
      sig_new[2][2]=_meot->second[2].first->GetFunction("gaus")->GetParameter(2);
      sig_new[3][3]=_meot->second[3].first->GetFunction("gaus")->GetParameter(2);
    }
    else{
      cap_new[0]=_meot->second[0].first->GetMean();
      cap_new[1]=_meot->second[1].first->GetMean();
      cap_new[2]=_meot->second[2].first->GetMean();
      cap_new[3]=_meot->second[3].first->GetMean();
      sig_new[0][0]=_meot->second[0].first->GetRMS();
      sig_new[1][1]=_meot->second[1].first->GetRMS();
      sig_new[2][2]=_meot->second[2].first->GetRMS();
      sig_new[3][3]=_meot->second[3].first->GetRMS();
    }
    sig_new[0][0]=sig_new[0][0]*sig_new[0][0];
    sig_new[1][1]=sig_new[1][1]*sig_new[1][1];
    sig_new[2][2]=sig_new[2][2]*sig_new[2][2];
    sig_new[3][3]=sig_new[3][3]*sig_new[3][3];
    dsig_new[0][0]=2*sig_new[0][0]*_meot->second[0].first->GetFunction("gaus")->GetParError(2);
    dsig_new[1][1]=2*sig_new[1][1]*_meot->second[1].first->GetFunction("gaus")->GetParError(2);
    dsig_new[2][2]=2*sig_new[2][2]*_meot->second[2].first->GetFunction("gaus")->GetParError(2);
    dsig_new[3][3]=2*sig_new[3][3]*_meot->second[3].first->GetFunction("gaus")->GetParError(2);
    sig_new[0][1]= _meot->second[4].first->GetMean()-cap_new[0]*cap_new[1];
    sig_new[0][2]= _meot->second[8].first->GetMean()-cap_new[0]*cap_new[2];
    sig_new[1][2]= _meot->second[5].first->GetMean()-cap_new[1]*cap_new[2];
    sig_new[1][3]= _meot->second[9].first->GetMean()-cap_new[1]*cap_new[3];
    sig_new[2][3]= _meot->second[6].first->GetMean()-cap_new[2]*cap_new[3];
    sig_new[3][0]= _meot->second[7].first->GetMean()-cap_new[3]*cap_new[0];
    if (pedCan) pedCan->addValue(_meot->first,cap_new[0],cap_new[1],cap_new[2],cap_new[3]);
    if (widthCan) {
      HcalPedestalWidth* widthsp = widthCan->setWidth(_meot->first);
      widthsp->setSigma(1,1,sig_new[0][0]);
      widthsp->setSigma(1,2,sig_new[0][1]);
      widthsp->setSigma(1,3,sig_new[0][2]);
      widthsp->setSigma(2,2,sig_new[1][1]);
      widthsp->setSigma(2,3,sig_new[1][2]);
      widthsp->setSigma(2,4,sig_new[1][3]);
      widthsp->setSigma(3,3,sig_new[2][2]);
      widthsp->setSigma(3,4,sig_new[2][3]);
      widthsp->setSigma(4,4,sig_new[3][3]);
      widthsp->setSigma(4,1,sig_new[3][0]);
    }
    if(pedValflag>0 && pedCan_nominal && widthCan_nominal){
      cap_nom[0]=pedCan_nominal->getValue(_meot->first,1);
      cap_nom[1]=pedCan_nominal->getValue(_meot->first,2);
      cap_nom[2]=pedCan_nominal->getValue(_meot->first,3);
      cap_nom[3]=pedCan_nominal->getValue(_meot->first,4);
      sig_nom[0][0]=widthCan_nominal->getWidth(_meot->first,1);
      sig_nom[0][1]=widthCan_nominal->getSigma(_meot->first,1,2);
      sig_nom[0][2]=widthCan_nominal->getSigma(_meot->first,1,3);
      sig_nom[1][1]=widthCan_nominal->getWidth(_meot->first,2);
      sig_nom[1][2]=widthCan_nominal->getSigma(_meot->first,2,3);
      sig_nom[1][3]=widthCan_nominal->getSigma(_meot->first,2,4);
      sig_nom[2][2]=widthCan_nominal->getWidth(_meot->first,3);
      sig_nom[2][3]=widthCan_nominal->getSigma(_meot->first,3,4);
      sig_nom[3][3]=widthCan_nominal->getWidth(_meot->first,4);
      sig_nom[3][0]=widthCan_nominal->getSigma(_meot->first,4,1);
    }
// here compute and store the quantities useful for physics analysis:
// means and widths in pairs of adjacent cap-ids; pairs are numbered
// after the first paired cap-id.
//---> F.R. I believe it is not used, thus I've commented it out. Otherwise objects must be created first
//   meansper2caps.addValue(_meot->first,cap_new[0]+cap_new[1],cap_new[1]+cap_new[2],cap_new[2]+cap_new[3],cap_new[3]+cap_new[0]);
//   widthsper2caps.addValue(_meot->first,sqrt(sig_new[0][0]+sig_new[1][1]+2*sig_new[0][1]),sqrt(sig_new[1][1]+sig_new[2][2]+2*sig_new[1][2]),sqrt(sig_new[2][2]+sig_new[3][3]+2*sig_new[2][3]),sqrt(sig_new[3][3]+sig_new[0][0]+2*sig_new[3][0]));
                                                                                
// here should go code that compares new values against nominal ones,
// in the present implementation this is somewhat simplistic: an update
// of the DB is deemed necessary if any mean pedestal deviates from its
// nominal by more than its nominal width plus the stat. error on its
// current measurement (no need to update if the new measurement is just
// junky!), or when a width or sigma coefficient deviates by more than 3
// times its measurement error from its nominal value
    if(pedValflag>0){
      for(int i=0; i<4; i++){
        if(cap_new[i]>0 && abs(cap_new[i]-cap_nom[i])>sig_nom[i][i]+dcap_new[i])PedsOK=0;
        if(sig_new[i][i]>0 && abs(sig_new[i][i]-sig_nom[i][i])>3.*dsig_new[i][i])PedsOK=0;
        for(int j=0; j<4; j++){
// error on off-diagonal elements are assumed the same as for diagonal
          if(abs(sig_new[i][j]-sig_nom[i][j])>3.*dsig_new[i][i])PedsOK=0;
        }
      }
    }
  }
  return PedsOK;
}

//-----------------------------------------------------------------------------
void HcalPedestalAnalysis::done(const HcalPedestals* fInputPedestals, 
				const HcalPedestalWidths* fInputPedestalWidths,
				HcalPedestals* fOutputPedestals, 
				HcalPedestalWidths* fOutputPedestalWidths) {
//  char PedSampleNum[20];
  map<int, std::vector<double> > AverageValues;

// First process the last sample (remaining events).
  if(evt%nevt_ped!=0) SampleAnalysis();

// Now do the end of run analysis: trending histos
  m_file->cd();
  m_file->cd("HB");
  Trendings(hbHists.PEDTRENDS,hbHists.CHI2,hbHists.CAPID_AVERAGE,hbHists.CAPID_CHI2);
  m_file->cd();
  m_file->cd("HO");
  Trendings(hoHists.PEDTRENDS,hoHists.CHI2,hoHists.CAPID_AVERAGE,hoHists.CAPID_CHI2);
  m_file->cd();
  m_file->cd("HF");
  Trendings(hfHists.PEDTRENDS,hfHists.CHI2,hfHists.CAPID_AVERAGE,hfHists.CAPID_CHI2);

//  Pedestal validation (Michal)
//  nominal values from DB
  // inputs...
  pedCan_nominal = fInputPedestals;
  widthCan_nominal = fInputPedestalWidths;
  // outputs...
  pedCan = fOutputPedestals;
  widthCan = fOutputPedestalWidths;

  int HBPedsOK=PedValidtn(hbHists.PEDTRENDS);
  int HOPedsOK=PedValidtn(hoHists.PEDTRENDS);
  int HFPedsOK=PedValidtn(hfHists.PEDTRENDS);

  if (pedCan && widthCan) {
    pedCan->sort();
    widthCan->sort();
  }
  // Write other histograms.
  // HB
  m_file->cd();
  m_file->cd("HB");
  hbHists.ALLPEDS->Write();
  hbHists.PEDRMS->Write();
  hbHists.PEDMEAN->Write();
  // HO
  m_file->cd();
  m_file->cd("HO");
  hoHists.ALLPEDS->Write();
  hoHists.PEDRMS->Write();
  hoHists.PEDMEAN->Write();
  // HF
  m_file->cd();
  m_file->cd("HF");
  hfHists.ALLPEDS->Write();
  hfHists.PEDRMS->Write();
  hfHists.PEDMEAN->Write();

  // Write the histo file and close it
  m_file->Write();
  m_file->Close();
  cout << "Hcal histograms written to " << m_outputFileROOT.c_str() << endl;
}

//-----------------------------------------------------------------------------
void HcalPedestalAnalysis::processEvent(const HBHEDigiCollection& hbhe,
					const HODigiCollection& ho,
					const HFDigiCollection& hf,
					const HcalDbService& cond){
  evt++;
  sample = (evt-1)/nevt_ped +1;
  m_shape = cond.getHcalShape();
  // Get data for every CAPID.
  // HB
  try{
    if(!hbhe.size()) throw (int)hbhe.size();
    for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); j++){
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      m_coder = cond.getHcalCoder(digi.id());
      for(int k=0; k<(int)state.size();k++) state[k]=true;
// here we loop over pairs of time slices, it is more convenient
// in order to extract the correlation matrix (Michal)
      for (int i=m_startSample; i<digi.size() && i<m_endSample; i++) {
        for(int flag=0; flag<4; flag++){
          if(i+flag<digi.size() && i+flag<m_endSample){
            per2CapsHists(flag,0,digi.id(),digi.sample(i),digi.sample(i+flag),hbHists.PEDTRENDS);
          }
        }
      }
    }
  }
  catch (int i ) {
    m_logFile<< "Event with " << i<<" HBHE Digis passed." << std::endl;
  } 
  // HO
  try{
    if(!ho.size()) throw (int)ho.size();
    for (HODigiCollection::const_iterator j=ho.begin(); j!=ho.end(); j++){
      const HODataFrame digi = (const HODataFrame)(*j);
      m_coder = cond.getHcalCoder(digi.id());
      for (int i=m_startSample; i<digi.size() && i<m_endSample; i++) {	   
        for(int flag=0; flag<4; flag++){
          if(i+flag<digi.size() && i+flag<m_endSample){
            per2CapsHists(flag,0,digi.id(),digi.sample(i),digi.sample(i+flag),hoHists.PEDTRENDS);
          }
        }
      }
    }        
  } 
  catch (int i ) {
//    m_logFile << "Event with " << i<<" HO Digis passed." << std::endl;
  } 
  // HF
  try{
    if(!hf.size()) throw (int)hf.size();
    for (HFDigiCollection::const_iterator j=hf.begin(); j!=hf.end(); j++){
      const HFDataFrame digi = (const HFDataFrame)(*j);
      m_coder = cond.getHcalCoder(digi.id());
      for (int i=m_startSample; i<digi.size() && i<m_endSample; i++) {
        for(int flag=0; flag<4; flag++){
          if(i+flag<digi.size() && i+flag<m_endSample){
            per2CapsHists(flag,0,digi.id(),digi.sample(i),digi.sample(i+flag),hfHists.PEDTRENDS);
          }
        }
      }
    }
  } 
  catch (int i ) {
//    m_logFile << "Event with " << i<<" HF Digis passed." << std::endl;
  } 
  // Call the function every nevt_ped events
  if(evt%nevt_ped==0) SampleAnalysis();
}
//-----------------------------------------------------------------------------
void HcalPedestalAnalysis::per2CapsHists(int flag, int id, const HcalDetId detid, const HcalQIESample& qie1, const HcalQIESample& qie2, map<HcalDetId, map<int,PEDBUNCH> > &toolT) {

// this function is due to be called for every time slice, it fills a histo
// for the current cap and for the correlation with another cap (Michal)

  static const int bins=10;
  static const int bins2=100;
  float lo=-1; float hi=9; float lo2=-1; float hi2=9;
  float slope[4]; float offset[4];
  map<int,PEDBUNCH> _mei;
  string type = "HB/HE";

  if(id==0) { 
    type = "HB/HE"; 
  }
  else if(id==1) { 
    type = "HO";
  }
  else if(id==2) { 
    type = "HF"; 
  }  

  _meot = toolT.find(detid);
  if (_meot!=toolT.end()){
// if histos for these particular cap-ids already exist, just fill them
    _mei = _meot->second;
    if(flag==0){
      if((evt-1)%nevt_ped==0 && state[qie1.capid()]){
        state[qie1.capid()]=false; 
        _mei[qie1.capid()].first->Reset();
        _mei[qie1.capid()+4].first->Reset();
        _mei[qie1.capid()+8].first->Reset();
        _mei[qie1.capid()+12].first->Reset();
      }
      if (qie1.adc()<bins){
        _mei[qie1.capid()].first->AddBinContent(qie1.adc()+1,1);
      }
      else if(qie1.adc()>=bins){
        _mei[qie1.capid()].first->AddBinContent(bins+1,1);
      }
//      m_logFile << qie1.capid()<<" "<<qie1.adc() <<std::endl;
    }
// unfortunately need to redo adc->fC every time,
// this is not very intelligent
    if(flag>0){
      getLinearizedADC(*m_shape,m_coder,bins,qie1.capid(),lo,hi);
      slope[qie1.capid()]=(hi-lo)/bins;
      offset[qie1.capid()]=lo+0.5;
      float charge1=qie1.adc()*slope[qie1.capid()]+offset[qie1.capid()];
      getLinearizedADC(*m_shape,m_coder,bins,qie2.capid(),lo2,hi2);
      slope[qie2.capid()]=(hi2-lo2)/bins;
      offset[qie2.capid()]=lo2+0.5;
      float charge2=qie2.adc()*slope[qie2.capid()]+offset[qie2.capid()];
      if (charge1*charge2<bins2){
        _mei[qie1.capid()+4*flag].first->Fill(charge1*charge2);
      }
      else{
        _mei[qie1.capid()+4*flag].first->Fill(bins2);
      }
    }
  }
  else{
// if histos for these cap-ids do not exist, first create them
// and then do all the same things - this is a bit silly as well
    map<int,PEDBUNCH> insert;
    for(int i=0; i<4; i++){
      char name[1024];
      sprintf(name,"%s Pedestal Value, ieta=%d iphi=%d depth=%d CAPID=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth(),i);  
      getLinearizedADC(*m_shape,m_coder,bins,i,lo,hi);
           // printf("Linearized: lo: %f, hi: %f\n",lo,hi);
      insert[i].first =  new TH1F(name,name,bins,lo,hi);
      slope[i]=(hi-lo)/bins;
      offset[i]=lo+0.5;
      sprintf(name,"%s Product, ieta=%d iphi=%d depth=%d CAPIDs=%d*%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth(),i,(i+1)%4);  
      insert[4+i].first = new TH1F(name,name,bins2,0.,100.);
      sprintf(name,"%s Product, ieta=%d iphi=%d depth=%d CAPIDs=%d*%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth(),i,(i+2)%4);  
      insert[8+i].first = new TH1F(name,name,bins2,0.,100.);
      sprintf(name,"%s Product, ieta=%d iphi=%d depth=%d CAPIDs=%d*%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth(),i,(i+3)%4);  
      insert[12+i].first = new TH1F(name,name,bins2,0.,100.);
    }
    if(flag==0){
      if((evt-1)%nevt_ped==0 && state[qie1.capid()]){
        state[qie1.capid()]=false; 
        insert[qie1.capid()].first->Reset();
        insert[qie1.capid()+4].first->Reset();
        insert[qie1.capid()+8].first->Reset();
        insert[qie1.capid()+12].first->Reset();
      }
      if (qie1.adc()<bins){
        insert[qie1.capid()].first->AddBinContent(qie1.adc()+1,1);
      }
      else{
        insert[qie1.capid()].first->AddBinContent(bins+1,1);
      }
//      m_logFile << qie1.capid()<<" "<<qie1.adc() <<" 2"<<std::endl;
    }
    if(flag>0){
      float charge1=qie1.adc()*slope[qie1.capid()]+offset[qie1.capid()];
      float charge2=qie2.adc()*slope[qie2.capid()]+offset[qie2.capid()];
      if (charge1*charge2<bins2){
        insert[qie1.capid()+4*flag].first->Fill(charge1*charge2);
      }
      else{
        insert[qie1.capid()+4*flag].first->Fill(bins2);
      }
    }
    toolT[detid] = insert;
  }

  if(flag==0){
    if(id==0) {
      hbHists.ALLPEDS->Fill(qie1.adc());
    }
    else if(id==1){ 
      hoHists.ALLPEDS->Fill(qie1.adc());
    }
    else if(id==2){
      hfHists.ALLPEDS->Fill(qie1.adc());
    }
  }
}
