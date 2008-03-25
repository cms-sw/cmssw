
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalLedAnalysis.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalAlgoUtils.h"
#include <TFile.h>

using namespace std;

HcalLedAnalysis::HcalLedAnalysis(const edm::ParameterSet& ps)
  : pedCan (0)
{
  // init
  evt=0;
  sample=0;
  m_file=0;
  // output files
  for(int k=0;k<4;k++) state.push_back(true); // 4 cap-ids (do we care?)
  m_outputFileText = ps.getUntrackedParameter<string>("outputFileText", "");
  if ( m_outputFileText.size() != 0 ) {
    cout << "Hcal LED results will be saved to " << m_outputFileText.c_str() << endl;
    m_outFile.open(m_outputFileText.c_str());
  } 
  m_outputFileROOT = ps.getUntrackedParameter<string>("outputFileHist", "");
  if ( m_outputFileROOT.size() != 0 ) {
    cout << "Hcal LED histograms will be saved to " << m_outputFileROOT.c_str() << endl;
  } 
  m_nevtsample = ps.getUntrackedParameter<int>("nevtsample",9999999);
  if(m_nevtsample<1)m_nevtsample=9999999;
  m_hiSaveflag = ps.getUntrackedParameter<int>("hiSaveflag",0);
  if(m_hiSaveflag<0)m_hiSaveflag=0;
  if(m_hiSaveflag>0)m_hiSaveflag=1;
  m_fitflag = ps.getUntrackedParameter<int>("analysisflag",2);
  if(m_fitflag<0)m_fitflag=0;
  if(m_fitflag>4)m_fitflag=4;
  m_startTS = ps.getUntrackedParameter<int>("firstTS", 0);
  if(m_startTS<0) m_startTS=0;
  m_endTS = ps.getUntrackedParameter<int>("lastTS", 9);
  m_logFile.open("HcalLedAnalysis.log");

  // histogram booking
  hbHists.ALLLEDS = new TH1F("HB/HE All LEDs","HB/HE All Leds",10,0,9);
  hbHists.LEDRMS= new TH1F("HB/HE All LED RMS","HB/HE All LED RMS",100,0,3);
  hbHists.LEDMEAN= new TH1F("HB/HE All LED Means","HB/HE All LED Means",100,0,9);
  hbHists.CHI2= new TH1F("HB/HE Chi2/ndf for Landau fit","HB/HE Chi2/ndf Gauss",200,0.,50.);

  hoHists.ALLLEDS = new TH1F("HO All LEDs","HO All Leds",10,0,9);
  hoHists.LEDRMS= new TH1F("HO All LED RMS","HO All LED RMS",100,0,3);
  hoHists.LEDMEAN= new TH1F("HO All LED Means","HO All LED Means",100,0,9);
  hoHists.CHI2= new TH1F("HO Chi2/ndf for Landau fit","HO Chi2/ndf Gauss",200,0.,50.);

  hfHists.ALLLEDS = new TH1F("HF All LEDs","HF All Leds",10,0,9);
  hfHists.LEDRMS= new TH1F("HF All LED RMS","HF All LED RMS",100,0,3);
  hfHists.LEDMEAN= new TH1F("HF All LED Means","HF All LED Means",100,0,9);
  hfHists.CHI2= new TH1F("HF Chi2/ndf for Landau fit","HF Chi2/ndf Gauss",200,0.,50.);
}

//-----------------------------------------------------------------------------
HcalLedAnalysis::~HcalLedAnalysis(){
  ///All done, clean up!!
  for(_meol=hbHists.LEDTRENDS.begin(); _meol!=hbHists.LEDTRENDS.end(); _meol++){
    for(int i=0; i<15; i++) _meol->second[i].first->Delete();
  }
  for(_meol=hoHists.LEDTRENDS.begin(); _meol!=hoHists.LEDTRENDS.end(); _meol++){
    for(int i=0; i<15; i++) _meol->second[i].first->Delete();
  }
  for(_meol=hfHists.LEDTRENDS.begin(); _meol!=hfHists.LEDTRENDS.end(); _meol++){
    for(int i=0; i<15; i++) _meol->second[i].first->Delete();
  }
  hbHists.ALLLEDS->Delete();
  hbHists.LEDRMS->Delete();
  hbHists.LEDMEAN->Delete();
  hbHists.CHI2->Delete();

  hoHists.ALLLEDS->Delete();
  hoHists.LEDRMS->Delete();
  hoHists.LEDMEAN->Delete();
  hoHists.CHI2->Delete();

  hfHists.ALLLEDS->Delete();
  hfHists.LEDRMS->Delete();
  hfHists.LEDMEAN->Delete();
  hfHists.CHI2->Delete();
}

//-----------------------------------------------------------------------------
void HcalLedAnalysis::LedSetup(const std::string& m_outputFileROOT) {
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
void HcalLedAnalysis::doPeds(const HcalPedestals* fInputPedestals){
// put all pedestals in a map m_AllPedVals, to be used in processLedEvent -
// sorry, this is the only way I was able to implement pedestal subtraction
  HcalDetId detid;
  map<int,float> PedVals;
  pedCan = fInputPedestals;
  if(pedCan){
    std::vector<DetId> Channs=pedCan->getAllChannels();
    for (int i=0; i<(int)Channs.size(); i++){
      detid=HcalDetId (Channs[i]);
      for (int icap=0; icap<4; icap++) PedVals[icap]=pedCan->getValue(detid,icap);
      m_AllPedVals[detid]=PedVals;
    }
  }
}

//-----------------------------------------------------------------------------
void HcalLedAnalysis::GetLedConst(map<HcalDetId, map<int,LEDBUNCH> > &toolT){
  double time2=0; double time1=0; double time3=0; double time4=0;
  double dtime2=0; double dtime1=0; double dtime3=0; double dtime4=0;

  if (m_outputFileText!=""){
    if(m_fitflag==0 || m_fitflag==2) m_outFile<<"Det Eta,Phi,D   Mean    Error"<<std::endl;
    else if(m_fitflag==1 || m_fitflag==3) m_outFile<<"Det Eta,Phi,D   Peak    Error"<<std::endl;
    else if(m_fitflag==4) m_outFile<<"Det Eta,Phi,D   Mean    Error      Peak    Error       MeanEv  Error       PeakEv  Error"<<std::endl;
  }
  for(_meol=toolT.begin(); _meol!=toolT.end(); _meol++){
// scale the LED pulse to 1 event
    _meol->second[10].first->Scale(1./evt_curr);
    if(m_fitflag==0 || m_fitflag==4){
      time1 = _meol->second[10].first->GetMean();
      dtime1 = _meol->second[10].first->GetRMS()/sqrt((float)evt_curr*(m_endTS-m_startTS+1));
    }
    if(m_fitflag==1 || m_fitflag==4){
// put proper errors
      for(int j=0; j<10; j++) _meol->second[10].first->SetBinError(j+1,_meol->second[j].first->GetRMS()/sqrt((float)evt_curr));
    }
    if(m_fitflag==1 || m_fitflag==3 || m_fitflag==4){
      _meol->second[10].first->Fit("landau","Q");
//      _meol->second[10].first->Fit("gaus","Q");
      TF1 *fit = _meol->second[10].first->GetFunction("landau");
//      TF1 *fit = _meol->second[10].first->GetFunction("gaus");
      time2=fit->GetParameter(1);
      dtime2=fit->GetParError(1);
    }
    if(m_fitflag==2 || m_fitflag==4){
      time3 = _meol->second[12].first->GetMean();
      dtime3 = _meol->second[12].first->GetRMS()/sqrt((float)_meol->second[12].first->GetEntries());
    }
    if(m_fitflag==3 || m_fitflag==4){
      time4 = _meol->second[13].first->GetMean();
      dtime4 = _meol->second[13].first->GetRMS()/sqrt((float)_meol->second[13].first->GetEntries());
    }
    for (int i=0; i<10; i++){
      _meol->second[i].first->GetXaxis()->SetTitle("Pulse height (fC)");
      _meol->second[i].first->GetYaxis()->SetTitle("Counts");
//      if(m_hiSaveflag>0)_meol->second[i].first->Write();
    }
    _meol->second[10].first->GetXaxis()->SetTitle("Time slice");
    _meol->second[10].first->GetYaxis()->SetTitle("Averaged pulse (fC)");
    if(m_hiSaveflag>0)_meol->second[10].first->Write();
    _meol->second[10].second.first[0].push_back(time1);
    _meol->second[10].second.first[1].push_back(dtime1);
    _meol->second[11].second.first[0].push_back(time2);
    _meol->second[11].second.first[1].push_back(dtime2);
    _meol->second[12].first->GetXaxis()->SetTitle("Mean TS");
    _meol->second[12].first->GetYaxis()->SetTitle("Counts");
    if(m_fitflag==2 && m_hiSaveflag>0)_meol->second[12].first->Write();
    _meol->second[12].second.first[0].push_back(time3);
    _meol->second[12].second.first[1].push_back(dtime3);
    _meol->second[13].first->GetXaxis()->SetTitle("Peak TS");
    _meol->second[13].first->GetYaxis()->SetTitle("Counts");
    if(m_fitflag>2 && m_hiSaveflag>0)_meol->second[13].first->Write();
    _meol->second[13].second.first[0].push_back(time4);
    _meol->second[13].second.first[1].push_back(dtime4);
    _meol->second[14].first->GetXaxis()->SetTitle("Peak TS error");
    _meol->second[14].first->GetYaxis()->SetTitle("Counts");
    if(m_fitflag>2 && m_hiSaveflag>0)_meol->second[14].first->Write();
    _meol->second[15].first->GetXaxis()->SetTitle("Chi2/NDF");
    _meol->second[15].first->GetYaxis()->SetTitle("Counts");
    if(m_fitflag>2 && m_hiSaveflag>0)_meol->second[15].first->Write();

// Ascii printout
    HcalDetId detid = _meol->first;
    if (m_outputFileText!=""){
      if(m_fitflag==0) m_outFile<<detid<<"   "<<time1<<" "<<dtime1<<std::endl;
      else if(m_fitflag==1) m_outFile<<detid<<"   "<<time2<<" "<<dtime2<<std::endl;
      else if(m_fitflag==2) m_outFile<<detid<<"   "<<time3<<" "<<dtime3<<std::endl;
      else if(m_fitflag==3) m_outFile<<detid<<"   "<<time4<<" "<<dtime4<<std::endl;
      else if(m_fitflag==4) m_outFile<<detid<<"   "<<time1<<" "<<dtime1<<"   "<<time2<<" "<<dtime2<<"   "<<time3<<" "<<dtime3<<"   "<<time4<<" "<<dtime4<<std::endl;
    }
  }
}

//-----------------------------------------------------------------------------
void HcalLedAnalysis::LedSampleAnalysis(){
  // it is called every m_nevtsample events (a sample) and the end of run
  char LedSampleNum[20];

  sprintf(LedSampleNum,"LedSample_%d",sample);
  m_file->cd();
  m_file->mkdir(LedSampleNum);
  m_file->cd(LedSampleNum);

// Compute LED constants for each HB/HE, HO, HF
  GetLedConst(hbHists.LEDTRENDS);
  GetLedConst(hoHists.LEDTRENDS);
  GetLedConst(hfHists.LEDTRENDS);
}

//-----------------------------------------------------------------------------
void HcalLedAnalysis::LedTrendings(map<HcalDetId, map<int,LEDBUNCH> > &toolT)
{

  for(_meol=toolT.begin(); _meol!=toolT.end(); _meol++){
    char name[1024];
    HcalDetId detid = _meol->first;
    sprintf(name,"LED timing trend, eta=%d phi=%d depth=%d",detid.ieta(),detid.iphi(),detid.depth());
    int bins = _meol->second[10+m_fitflag].second.first[0].size();
    float lo =0.5;
    float hi = (float)bins+0.5;
    _meol->second[10+m_fitflag].second.second.push_back(new TH1F(name,name,bins,lo,hi));

    std::vector<double>::iterator sample_it;
// LED timing - put content and errors
    int j=0;
    for(sample_it=_meol->second[10+m_fitflag].second.first[0].begin();
        sample_it!=_meol->second[10+m_fitflag].second.first[0].end();sample_it++){
      _meol->second[10+m_fitflag].second.second[0]->SetBinContent(++j,*sample_it);
    }
    j=0;
    for(sample_it=_meol->second[10+m_fitflag].second.first[1].begin();
        sample_it!=_meol->second[10+m_fitflag].second.first[1].end();sample_it++){
      _meol->second[10+m_fitflag].second.second[0]->SetBinError(++j,*sample_it);
    }
    sprintf(name,"Sample (%d events)",m_nevtsample);
    _meol->second[10+m_fitflag].second.second[0]->GetXaxis()->SetTitle(name);
    _meol->second[10+m_fitflag].second.second[0]->GetYaxis()->SetTitle("Peak position");
    _meol->second[10+m_fitflag].second.second[0]->Write();
  }
}

//-----------------------------------------------------------------------------
void HcalLedAnalysis::LedDone() 
{

// First process the last sample (remaining events).
  if(evt%m_nevtsample!=0) LedSampleAnalysis();

// Now do the end of run analysis: trending histos
  if(sample>1 && m_fitflag!=4){
    m_file->cd();
    m_file->cd("HB");
    LedTrendings(hbHists.LEDTRENDS);
    m_file->cd();
    m_file->cd("HO");
    LedTrendings(hoHists.LEDTRENDS);
    m_file->cd();
    m_file->cd("HF");
    LedTrendings(hfHists.LEDTRENDS);
  }

  // Write other histograms.
  // HB
  m_file->cd();
  m_file->cd("HB");
  hbHists.ALLLEDS->Write();
  hbHists.LEDRMS->Write();
  hbHists.LEDMEAN->Write();
  // HO
  m_file->cd();
  m_file->cd("HO");
  hoHists.ALLLEDS->Write();
  hoHists.LEDRMS->Write();
  hoHists.LEDMEAN->Write();
  // HF
  m_file->cd();
  m_file->cd("HF");
  hfHists.ALLLEDS->Write();
  hfHists.LEDRMS->Write();
  hfHists.LEDMEAN->Write();

  // Write the histo file and close it
//  m_file->Write();
  m_file->Close();
  cout << "Hcal histograms written to " << m_outputFileROOT.c_str() << endl;
}

//-----------------------------------------------------------------------------
void HcalLedAnalysis::processLedEvent(const HBHEDigiCollection& hbhe,
					const HODigiCollection& ho,
					const HFDigiCollection& hf,
					const HcalDbService& cond)
{
  evt++;
  sample = (evt-1)/m_nevtsample +1;
  evt_curr = evt%m_nevtsample;
  if(evt_curr==0)evt_curr=m_nevtsample;

  m_shape = cond.getHcalShape();

  // Get data from every time slice
  // HB
  try{
    if(!hbhe.size()) throw (int)hbhe.size();
// this is effectively a loop over electronic channels
    for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); j++){
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      m_coder = cond.getHcalCoder(digi.id());
      for(int k=0; k<(int)state.size();k++) state[k]=true;
// here we loop over time slices
      for (int i=m_startTS; i<digi.size() && i<=m_endTS; i++) {
        const HcalQIESample& qiedata=digi.sample(i);
        float pedval=0;
        for (_meee=m_AllPedVals.begin(); _meee!=m_AllPedVals.end(); _meee++){
          HcalDetId detid=_meee->first;
          if(detid==digi.id()) pedval=_meee->second[qiedata.capid()];
        }
        LedTSHists(0,digi.id(),i,digi.sample(i),hbHists.LEDTRENDS,pedval);
      }
    }
  }
  catch (int i ) {
//    m_logFile<< "Event with " << i<<" HBHE Digis passed." << std::endl;
  } 
  // HO
  try{
    if(!ho.size()) throw (int)ho.size();
    for (HODigiCollection::const_iterator j=ho.begin(); j!=ho.end(); j++){
      const HODataFrame digi = (const HODataFrame)(*j);
      m_coder = cond.getHcalCoder(digi.id());
      for (int i=m_startTS; i<digi.size() && i<=m_endTS; i++) {	   
        const HcalQIESample& qiedata=digi.sample(i);
        float pedval=0;
        for (_meee=m_AllPedVals.begin(); _meee!=m_AllPedVals.end(); _meee++){
          HcalDetId detid=_meee->first;
          if(detid==digi.id())pedval=_meee->second[qiedata.capid()];
        }
        LedTSHists(1,digi.id(),i,digi.sample(i),hoHists.LEDTRENDS,pedval);
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
      for (int i=m_startTS; i<digi.size() && i<=m_endTS; i++) {
        const HcalQIESample& qiedata=digi.sample(i);
        float pedval=0;
        for (_meee=m_AllPedVals.begin(); _meee!=m_AllPedVals.end(); _meee++){
          HcalDetId detid=_meee->first;
          if(detid==digi.id())pedval=_meee->second[qiedata.capid()];
        }
        LedTSHists(2,digi.id(),i,digi.sample(i),hfHists.LEDTRENDS,pedval);
      }
    }
  } 
  catch (int i ) {
//    m_logFile << "Event with " << i<<" HF Digis passed." << std::endl;
  } 
  // Call the function every m_nevtsample events
  if(evt%m_nevtsample==0) LedSampleAnalysis();
}
//-----------------------------------------------------------------------------
void HcalLedAnalysis::LedTSHists(int id, const HcalDetId detid, int TS, const HcalQIESample& qie1, map<HcalDetId, map<int,LEDBUNCH> > &toolT, float pedestal) {

// this function is due to be called for every time slice

  map<int,LEDBUNCH> _mei;
  string type = "HB/HE";

  if(id==1) type = "HO";
  if(id==2) type = "HF"; 

// and this is just a very dumb way of doing the adc->fc conversion in the
// full range (and is the same for all channels and cap-ids)
  static const float adc2fc[128]={-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 15., 17., 19., 21., 23., 25., 27., 29.5,
 32.5, 35.5, 38.5, 42., 46., 50., 54.5, 59.5, 64.5, 59.5, 64.5, 69.5, 74.5,
 79.5, 84.5, 89.5, 94.5, 99.5, 104.5, 109.5, 114.5, 119.5, 124.5, 129.5, 137.,
 147., 157., 167., 177., 187., 197., 209.5, 224.5, 239.5, 254.5, 272., 292.,
 312., 334.5, 359.5, 384.5, 359.5, 384.5, 409.5, 434.5, 459.5, 484.5, 509.5,
 534.5, 559.5, 584.5, 609.5, 634.5, 659.5, 684.5, 709.5, 747., 797., 847.,
 897., 947., 997., 1047., 1109.5, 1184.5, 1259.5, 1334.5, 1422., 1522., 1622.,
 1734.5, 1859.5, 1984.5, 1859.5, 1984.5, 2109.5, 2234.5, 2359.5, 2484.5,
 2609.5, 2734.5, 2859.5, 2984.5, 3109.5, 3234.5, 3359.5, 3484.5, 3609.5, 3797.,
 4047., 4297., 4547., 4797., 5047., 5297., 5609.5, 5984.5, 6359.5, 6734.5,
 7172., 7672., 8172., 8734.5, 9359.5, 9984.5};

  _meol = toolT.find(detid);
  if (_meol==toolT.end()){
// if histos for this channel do not exist, first create them
    map<int,LEDBUNCH> insert;
    char name[1024];
    for(int i=0; i<10; i++){
      sprintf(name,"%s Pulse height, eta=%d phi=%d depth=%d TS=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth(),i);  
      insert[i].first =  new TH1F(name,name,200,0.,2000.);
    }
    sprintf(name,"%s LED Mean pulse, eta=%d phi=%d depth=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth());  
    insert[10].first =  new TH1F(name,name,10,-0.5,9.5);
    sprintf(name,"%s LED Pulse, eta=%d phi=%d depth=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth());  
    insert[11].first =  new TH1F(name,name,10,-0.5,9.5);
    sprintf(name,"%s Mean TS, eta=%d phi=%d depth=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth());  
    insert[12].first =  new TH1F(name,name,200,0.,10.);
    sprintf(name,"%s Peak TS, eta=%d phi=%d depth=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth());  
    insert[13].first =  new TH1F(name,name,200,0.,10.);
    sprintf(name,"%s Peak TS error, eta=%d phi=%d depth=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth());  
    insert[14].first =  new TH1F(name,name,200,0.,0.05);
    sprintf(name,"%s Fit chi2, eta=%d phi=%d depth=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth());  
    insert[15].first =  new TH1F(name,name,100,0.,50.);

    toolT[detid] = insert;
    _meol = toolT.find(detid);
  }
  _mei = _meol->second;
  if((evt-1)%m_nevtsample==0 && state[0]){
    for(int k=0; k<(int)state.size();k++) state[k]=false;
    for(int i=0; i<15; i++) _mei[i].first->Reset();
  }
  _mei[TS].first->Fill(adc2fc[qie1.adc()]-pedestal);
  _mei[10].first->AddBinContent(TS+1,adc2fc[qie1.adc()]-pedestal);
  if(m_fitflag>1){
    if(TS==m_startTS)_mei[11].first->Reset();
    _mei[11].first->SetBinContent(TS+1,adc2fc[qie1.adc()]-pedestal);
    float fcgap;
// size of errors should compensate for the TS width, this
// certainly needs to be improved in future versions
    if(qie1.adc()==0)fcgap=adc2fc[1]-adc2fc[0];
    else if(qie1.adc()==127)fcgap=adc2fc[127]-adc2fc[126];
    else fcgap=0.5*(adc2fc[qie1.adc()+1]-adc2fc[qie1.adc()-1]);
    _mei[11].first->SetBinError(TS+1,fcgap);
    if(TS==m_endTS){
      float sum=0.;
      for(int i=0; i<10; i++)sum=sum+_mei[11].first->GetBinContent(i+1);
      if(sum>100){
        if(m_fitflag==2 || m_fitflag==4){
          float timmean=_mei[11].first->GetMean();
          float timmeancorr=BinsizeCorr(timmean);
          _mei[12].first->Fill(timmeancorr);
        }
        if(m_fitflag==3 || m_fitflag==4){
          _mei[11].first->Fit("landau","Q");
//          _mei[11].first->Fit("gaus","Q");
          TF1 *fit = _mei[11].first->GetFunction("landau");
//          TF1 *fit = _mei[11].first->GetFunction("gaus");
            _mei[13].first->Fill(fit->GetParameter(1));
            _mei[14].first->Fill(fit->GetParError(1));
          _mei[15].first->Fill(fit->GetChisquare()/fit->GetNDF());
        }
      }
    }
  }

  if(id==0) hbHists.ALLLEDS->Fill(qie1.adc());
  else if(id==1) hoHists.ALLLEDS->Fill(qie1.adc());
  else if(id==2) hfHists.ALLLEDS->Fill(qie1.adc());
}

//-----------------------------------------------------------------------------
float HcalLedAnalysis::BinsizeCorr(float time) {

// this is the bin size correction to be applied for laser data (from Andy),
// it comes from a pulse shape measured from TB04 data (from Jordan)

  float corrtime=0.;
  static const float tstrue[32]={0.003, 0.03425, 0.06548, 0.09675, 0.128,
 0.15925, 0.1905, 0.22175, 0.253, 0.28425, 0.3155, 0.34675, 0.378, 0.40925,
 0.4405, 0.47175, 0.503, 0.53425, 0.5655, 0.59675, 0.628, 0.65925, 0.6905,
 0.72175, 0.753, 0.78425, 0.8155, 0.84675, 0.878, 0.90925, 0.9405, 0.97175};
  static const float tsreco[32]={-0.00422, 0.01815, 0.04409, 0.07346, 0.09799,
 0.12192, 0.15072, 0.18158, 0.21397, 0.24865, 0.28448, 0.31973, 0.35449,
 0.39208, 0.43282, 0.47244, 0.5105, 0.55008, 0.58827, 0.62828, 0.6717, 0.70966,
 0.74086, 0.77496, 0.80843, 0.83472, 0.86044, 0.8843, 0.90674, 0.92982,
 0.95072, 0.9726};

 int inttime=(int)time;
 float restime=time-inttime;
 for(int i=0; i<=32; i++) {
   float lolim=0.; float uplim=1.; float tsdown; float tsup;
   if(i>0){
     lolim=tsreco[i-1];
     tsdown=tstrue[i-1];
   }
   else tsdown=tstrue[31]-1.;
   if(i<32){
     uplim=tsreco[i];
     tsup=tstrue[i];
   }
   else tsup=tstrue[0]+1.;
   if(restime>=lolim && restime<uplim){
      corrtime=(tsdown*(uplim-restime)+tsup*(restime-lolim)) / (uplim-lolim);
    }
  }
  corrtime+=inttime;

 return corrtime;
}
