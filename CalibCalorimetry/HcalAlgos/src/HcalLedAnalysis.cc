
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalLedAnalysis.h"
#include "TFile.h"
#include <math.h>
using namespace std;


HcalLedAnalysis::HcalLedAnalysis(const edm::ParameterSet& ps)
{
  // init

  m_coder = 0;
  m_ped   = 0;
  m_shape = 0;
  evt=0;
  sample=0;
  m_file=0;
  // output files
  for(int k=0;k<4;k++) state.push_back(true); // 4 cap-ids (do we care?)
  m_outputFileText = ps.getUntrackedParameter<string>("outputFileText", "");
  m_outputFileX = ps.getUntrackedParameter<string>("outputFileXML","");
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
  m_usecalib = ps.getUntrackedParameter<bool>("usecalib",false);
  m_logFile.open("HcalLedAnalysis.log");

  int runNum = ps.getUntrackedParameter<int>("runNumber",999999);

  // histogram booking
  hbHists.ALLLEDS = new TH1F("HBHE All LEDs","HB/HE All Leds",10,0,9);
  hbHists.LEDRMS= new TH1F("HBHE All LED RMS","HB/HE All LED RMS",100,0,3);
  hbHists.LEDMEAN= new TH1F("HBHE All LED Means","HB/HE All LED Means",100,0,9);
  hbHists.CHI2= new TH1F("HBHE Chi2 by ndf for Landau fit","HB/HE Chi2/ndf Landau",200,0.,50.);

  hoHists.ALLLEDS = new TH1F("HO All LEDs","HO All Leds",10,0,9);
  hoHists.LEDRMS= new TH1F("HO All LED RMS","HO All LED RMS",100,0,3);
  hoHists.LEDMEAN= new TH1F("HO All LED Means","HO All LED Means",100,0,9);
  hoHists.CHI2= new TH1F("HO Chi2 by ndf for Landau fit","HO Chi2/ndf Landau",200,0.,50.);

  hfHists.ALLLEDS = new TH1F("HF All LEDs","HF All Leds",10,0,9);
  hfHists.LEDRMS= new TH1F("HF All LED RMS","HF All LED RMS",100,0,3);
  hfHists.LEDMEAN= new TH1F("HF All LED Means","HF All LED Means",100,0,9);
  hfHists.CHI2= new TH1F("HF Chi2 by ndf for Landau fit","HF Chi2/ndf Landau",200,0.,50.);


  //XML file header
  m_outputFileXML.open(m_outputFileX.c_str());

  m_outputFileXML << "<?xml version='1.0' encoding='UTF-8'?>" << endl;

  m_outputFileXML << "<ROOT>" << endl;

  m_outputFileXML << "  <HEADER>" << endl;

  m_outputFileXML << "    <TYPE>" << endl;

  m_outputFileXML << "      <EXTENSION_TABLE_NAME>HCAL_LED_TIMING</EXTENSION_TABLE_NAME>" << endl;

  m_outputFileXML << "      <NAME>HCAL LED Timing</NAME>" << endl;

  m_outputFileXML << "    </TYPE>" << endl;

  m_outputFileXML << "    <RUN>" << endl;

  m_outputFileXML << "      <RUN_TYPE>hcal-led-timing-test</RUN_TYPE>" << endl;

  sprintf(output, "      <RUN_NUMBER>%06i</RUN_NUMBER>", runNum);
  m_outputFileXML << output << endl;

  m_outputFileXML << "      <RUN_BEGIN_TIMESTAMP>2007-07-09 00:00:00.0</RUN_BEGIN_TIMESTAMP>" << endl;

  m_outputFileXML << "      <COMMENT_DESCRIPTION></COMMENT_DESCRIPTION>" << endl;

  m_outputFileXML << "    </RUN>" << endl;

  m_outputFileXML << "  </HEADER>" << endl;

  m_outputFileXML << "<!-- Tags secton -->" << endl;

  m_outputFileXML <<  "  <ELEMENTS>" << endl;

  m_outputFileXML <<  "    <DATA_SET id='-1'/>" << endl;

  m_outputFileXML << "      <IOV id='1'>" << endl;

  m_outputFileXML << "        <INTERVAL_OF_VALIDITY_BEGIN>2147483647</INTERVAL_OF_VALIDITY_BEGIN>" << endl;

  m_outputFileXML <<  "        <INTERVAL_OF_VALIDITY_END>0</INTERVAL_OF_VALIDITY_END>" << endl;

  m_outputFileXML << "      </IOV>" << endl;

  m_outputFileXML << "      <TAG id='2' mode='auto'>" << endl;

  sprintf(output, "        <TAG_NAME>laser_led_%06i<TAG_NAME>", runNum);
  m_outputFileXML << output << endl;

  m_outputFileXML << "        <DETECTOR_NAME>HCAL</DETECTOR_NAME>" << endl;

  m_outputFileXML << "        <COMMENT_DESCRIPTION></COMMENT_DESCRIPTION>" << endl;

  m_outputFileXML << "      </TAG>" << endl;

  m_outputFileXML << "  </ELEMENTS>" << endl;

  m_outputFileXML << "  <MAPS>" << endl;

  m_outputFileXML << "      <TAG idref ='2'>" << endl;

  m_outputFileXML << "        <IOV idref='1'>" << endl;

  m_outputFileXML << "          <DATA_SET idref='-1' />" << endl;

  m_outputFileXML << "        </IOV>" << endl;

  m_outputFileXML << "      </TAG>" << endl;

  m_outputFileXML <<   "  </MAPS>" << endl;

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
  m_file->mkdir("HBHE");
  m_file->cd();
  m_file->mkdir("HO");
  m_file->cd();
  m_file->mkdir("HF");
  m_file->cd();
  m_file->mkdir("Calib");
  m_file->cd();
}

//-----------------------------------------------------------------------------
/*
void HcalLedAnalysis::doPeds(const HcalPedestal* fInputPedestals){
// put all pedestals in a map m_AllPedVals, to be used in processLedEvent -
// sorry, this is the only way I was able to implement pedestal subtraction

// DEPRECATED
// This is no longer useful, better ways of doing it -A
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
*/
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
    _meol->second[16].first->GetXaxis()->SetTitle("Integrated Signal");
    _meol->second[16].first->Write();


// Ascii printout (need to modify to include new info)
    HcalDetId detid = _meol->first;

    if (m_outputFileText!=""){
      if(m_fitflag==0) {
	m_outFile<<detid<<"   "<<time1<<" "<<dtime1<<std::endl;
        m_outputFileXML << "  <DATA_SET>" << endl;
        m_outputFileXML << "    <VERSION>version:1</VERSION>" << endl;
        m_outputFileXML << "    <CHANNEL>" << endl;
        m_outputFileXML << "      <EXTENSION_TABLE_NAME>HCAL_CHANNELS</EXTENSION_TABLE_NAME>" << endl;
        sprintf(output, "      <ETA>%2i</ETA>", detid.ietaAbs() );
        m_outputFileXML << output << endl;
        sprintf(output, "      <PHI>%2i</PHI>", detid.iphi() );
        m_outputFileXML << output << endl;
        sprintf(output, "      <DEPTH>%2i</DEPTH>", detid.depth() );
        m_outputFileXML << output << endl;
        sprintf(output, "      <Z>%2i</Z>", detid.zside() );
        m_outputFileXML << output << endl;

        if(detid.subdet() == 1)  m_outputFileXML << "      <DETECTOR_NAME>HB</DETECTOR_NAME>" << endl;
        if(detid.subdet() == 2)  m_outputFileXML << "      <DETECTOR_NAME>HE</DETECTOR_NAME>" << endl;
        if(detid.subdet() == 3)  m_outputFileXML << "      <DETECTOR_NAME>HO</DETECTOR_NAME>" << endl;
        if(detid.subdet() == 4)  m_outputFileXML << "      <DETECTOR_NAME>HF</DETECTOR_NAME>" << endl;
        sprintf(output, "      <HCAL_CHANNEL_ID>%10i</HCAL_CHANNEL_ID>", detid.rawId() );
        m_outputFileXML << output << endl;
        m_outputFileXML << "    </CHANNEL>" << endl;
        m_outputFileXML << "    <DATA>" << endl;
        sprintf(output, "      <MEAN_TIME>%7f</MEAN_TIME>", time1);
        m_outputFileXML << output << endl;
        m_outputFileXML << "      <OFFSET_TIME> 0</OFFSET_TIME>" << endl;
        sprintf(output, "      <ERROR_STAT>%7f</ERROR_STAT>", dtime1);
        m_outputFileXML << output << endl;
        sprintf(output, "      <ANALYSIS_FLAG>%2i</ANALYSIS_FLAG>", m_fitflag+1);
        m_outputFileXML << output << endl;
        m_outputFileXML << "      <STATUS_WORD>  0</STATUS_WORD>" << endl;
        m_outputFileXML << "    </DATA>" << endl;
        m_outputFileXML << "  </DATA_SET>" << endl;

	}
      else if(m_fitflag==1){
	m_outFile<<detid<<"   "<<time2<<" "<<dtime2<<std::endl;
        m_outputFileXML << "  <DATA_SET>" << endl;
        m_outputFileXML << "    <VERSION>version:1</VERSION>" << endl;
        m_outputFileXML << "    <CHANNEL>" << endl;
        m_outputFileXML << "      <EXTENSION_TABLE_NAME>HCAL_CHANNELS</EXTENSION_TABLE_NAME>" << endl;
        sprintf(output, "      <ETA>%2i</ETA>", detid.ietaAbs() );
        m_outputFileXML << output << endl;
        sprintf(output, "      <PHI>%2i</PHI>", detid.iphi() );
        m_outputFileXML << output << endl;
        sprintf(output, "      <DEPTH>%2i</DEPTH>", detid.depth() );
        m_outputFileXML << output << endl;
        sprintf(output, "      <Z>%2i</Z>", detid.zside() );
        m_outputFileXML << output << endl;
        if(detid.subdet() == 1)  m_outputFileXML << "      <DETECTOR_NAME>HB</DETECTOR_NAME>"<< endl;
        if(detid.subdet() == 2)  m_outputFileXML << "      <DETECTOR_NAME>HE</DETECTOR_NAME>"<< endl;
        if(detid.subdet() == 3)  m_outputFileXML << "      <DETECTOR_NAME>HO</DETECTOR_NAME>"<< endl;
        if(detid.subdet() == 4)  m_outputFileXML << "      <DETECTOR_NAME>HF</DETECTOR_NAME>"<< endl;
        sprintf(output, "      <HCAL_CHANNEL_ID>%10i</HCAL_CHANNEL_ID>", detid.rawId() );
        m_outputFileXML << output << endl;
        m_outputFileXML << "    </CHANNEL>" << endl;
        m_outputFileXML << "    <DATA>" << endl;
        sprintf(output, "      <MEAN_TIME>%7f</MEAN_TIME>", time2);
        m_outputFileXML << output << endl;
        m_outputFileXML << "      <OFFSET_TIME> 0</OFFSET_TIME>" << endl;
        sprintf(output, "      <ERROR_STAT>%7f</ERROR_STAT>", dtime2);
        m_outputFileXML << output << endl;
        sprintf(output, "      <ANALYSIS_FLAG>%2i</ANALYSIS_FLAG>", m_fitflag+1);
        m_outputFileXML << output << endl;
        m_outputFileXML << "      <STATUS_WORD>  0</STATUS_WORD>" << endl;
        m_outputFileXML << "    </DATA>" << endl;
        m_outputFileXML << "  </DATA_SET>" << endl;
        }

      else if(m_fitflag==2){
	m_outFile<<detid<<"   "<<time3<<" "<<dtime3<<std::endl;
        m_outputFileXML << "  <DATA_SET>" << endl;
        m_outputFileXML << "    <VERSION>version:1</VERSION>" << endl;
        m_outputFileXML << "    <CHANNEL>" << endl;
        m_outputFileXML << "      <EXTENSION_TABLE_NAME>HCAL_CHANNELS</EXTENSION_TABLE_NAME>" << endl;
        sprintf(output, "      <ETA>%2i</ETA>", detid.ietaAbs() );
        m_outputFileXML << output << endl;
        sprintf(output, "      <PHI>%2i</PHI>", detid.iphi() );
        m_outputFileXML << output << endl;
        sprintf(output, "      <DEPTH>%2i</DEPTH>", detid.depth() );
        m_outputFileXML << output << endl;
        sprintf(output, "      <Z>%2i</Z>", detid.zside() );
        m_outputFileXML << output << endl;
	if(detid.subdet() == 1) m_outputFileXML << "      <DETECTOR_NAME>HB</DETECTOR_NAME>" << endl;
        if(detid.subdet() == 2) m_outputFileXML << "      <DETECTOR_NAME>HE</DETECTOR_NAME>" << endl;
        if(detid.subdet() == 3) m_outputFileXML << "      <DETECTOR_NAME>HO</DETECTOR_NAME>" << endl;
        if(detid.subdet() == 4) m_outputFileXML << "      <DETECTOR_NAME>HF</DETECTOR_NAME>" << endl;
	sprintf(output, "      <HCAL_CHANNEL_ID>%10i</HCAL_CHANNEL_ID>", detid.rawId() );
        m_outputFileXML << output << endl;
        m_outputFileXML << "    </CHANNEL>" << endl;
        m_outputFileXML << "    <DATA>" << endl;
        sprintf(output, "      <MEAN_TIME>%7f</MEAN_TIME>", time3);
        m_outputFileXML << output << endl;
        m_outputFileXML << "      <OFFSET_TIME> 0</OFFSET_TIME>" << endl;
        sprintf(output, "      <ERROR_STAT>%7f</ERROR_STAT>", dtime3);
        m_outputFileXML << output << endl;
	sprintf(output, "      <ANALYSIS_FLAG>%2i</ANALYSIS_FLAG>", m_fitflag+1);
	m_outputFileXML << output << endl;
  	m_outputFileXML << "      <STATUS_WORD>  0</STATUS_WORD>" << endl;
        m_outputFileXML << "    </DATA>" << endl;
        m_outputFileXML << "  </DATA_SET>" << endl;
        }
      else if(m_fitflag==3){
	m_outFile<<detid<<"   "<<time4<<" "<<dtime4<<std::endl;
        m_outputFileXML << "  <DATA_SET>" << endl;
        m_outputFileXML <<"    <VERSION>version:1</VERSION>" << endl;
        m_outputFileXML << "    <CHANNEL>" << endl;
        m_outputFileXML << "      <EXTENSION_TABLE_NAME>HCAL_CHANNELS</EXTENSION_TABLE_NAME>" << endl;
        sprintf(output, "      <ETA>%2i</ETA>", detid.ietaAbs() );
        m_outputFileXML << output << endl;
        sprintf(output, "      <PHI>%2i</PHI>", detid.iphi() );
        m_outputFileXML << output << endl;
        sprintf(output, "      <DEPTH>%2i</DEPTH>", detid.depth() );
        m_outputFileXML << output << endl;
        sprintf(output, "      <Z>%2i</Z>", detid.zside() );
        m_outputFileXML << output << endl;
        if(detid.subdet() == 1) m_outputFileXML <<"      <DETECTOR_NAME>HB</DETECTOR_NAME>" << endl;
        if(detid.subdet() == 2) m_outputFileXML <<"      <DETECTOR_NAME>HE</DETECTOR_NAME>" << endl;
        if(detid.subdet() == 3) m_outputFileXML <<"      <DETECTOR_NAME>HO</DETECTOR_NAME>" << endl;
        if(detid.subdet() == 4) m_outputFileXML <<"      <DETECTOR_NAME>HF</DETECTOR_NAME>" << endl;
        sprintf(output, "      <HCAL_CHANNEL_ID>%10i</HCAL_CHANNEL_ID>", detid.rawId() );
        m_outputFileXML << output << endl;
        m_outputFileXML << "    </CHANNEL>" << endl;
        m_outputFileXML << "    <DATA>" << endl;
        sprintf(output, "      <MEAN_TIME>%7f</MEAN_TIME>", time4);
        m_outputFileXML << output << endl;
        m_outputFileXML << "      <OFFSET_TIME> 0</OFFSET_TIME>" << endl;
        sprintf(output, "      <ERROR_STAT>%7f</ERROR_STAT>", dtime4);
        m_outputFileXML << output << endl;
        sprintf(output, "      <ANALYSIS_FLAG>%2i</ANALYSIS_FLAG>", m_fitflag+1);
        m_outputFileXML << output << endl;
        m_outputFileXML << "      <STATUS_WORD>  0</STATUS_WORD>" << endl;
        m_outputFileXML << "    </DATA>" << endl;
        m_outputFileXML << "  </DATA_SET>" << endl;
        }

      else if(m_fitflag==4){
	m_outFile<<detid<<"   "<<time1<<" "<<dtime1<<"   "<<time2<<" "<<dtime2<<"   "<<time3<<" "<<dtime3<<"   "<<time4<<" "<<dtime4<<std::endl;
	}
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
    m_file->cd("HBHE");
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
  m_file->cd("HBHE");
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
  // Calib
  m_file->cd();
  m_file->cd("Calib");
  for(_meca=calibHists.begin(); _meca!=calibHists.end(); _meca++){
    _meca->second.avePulse->Write();
    _meca->second.integPulse->Write();
  }

  // Write the histo file and close it
//  m_file->Write();
  m_file->Close();
  cout << "Hcal histograms written to " << m_outputFileROOT.c_str() << endl;
}

//-----------------------------------------------------------------------------
void HcalLedAnalysis::processLedEvent(const HBHEDigiCollection& hbhe,
					const HODigiCollection& ho,
					const HFDigiCollection& hf,
                                        const HcalCalibDigiCollection calib,
					const HcalDbService& cond)
{
  evt++;
  sample = (evt-1)/m_nevtsample +1;
  evt_curr = evt%m_nevtsample;
  if(evt_curr==0)evt_curr=m_nevtsample;

  // Calib

  if (m_usecalib){
    try{
      if(!calib.size()) throw (int)calib.size();
      // this is effectively a loop over electronic channels
      for (HcalCalibDigiCollection::const_iterator j=calib.begin(); j!=calib.end(); j++){
        const HcalCalibDataFrame digi = (const HcalCalibDataFrame)(*j);   
        HcalElectronicsId elecId = digi.elecId();
        HcalCalibDetId calibId = digi.id();
        ProcessCalibEvent(elecId.fiberChanId(),calibId,digi);  //Shouldn't depend on anything in elecId but not sure how else to do it 
      }
    }
    catch (int i ) {
    //  m_logFile<< "Event with " << i<<" Calib Digis passed." << std::endl;
    }
  }


  // HB + HE
  try{
    if(!hbhe.size()) throw (int)hbhe.size();
// this is effectively a loop over electronic channels
    for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); j++){
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      for(int k=0; k<(int)state.size();k++) state[k]=true;
      // See if histos exist for this channel, and if not, create them
      _meol = hbHists.LEDTRENDS.find(digi.id());
      if (_meol==hbHists.LEDTRENDS.end()){
        SetupLEDHists(0,digi.id(),hbHists.LEDTRENDS);
      }
      LedHBHEHists(digi.id(),digi,hbHists.LEDTRENDS,cond);
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
      _meol = hoHists.LEDTRENDS.find(digi.id());
      if (_meol==hoHists.LEDTRENDS.end()){
        SetupLEDHists(1,digi.id(),hoHists.LEDTRENDS);
      }
      LedHOHists(digi.id(),digi,hoHists.LEDTRENDS,cond);
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
      _meol = hfHists.LEDTRENDS.find(digi.id());
      if (_meol==hfHists.LEDTRENDS.end()){
        SetupLEDHists(2,digi.id(),hfHists.LEDTRENDS);
      }
      LedHFHists(digi.id(),digi,hfHists.LEDTRENDS,cond);
    }
  } 
  catch (int i ) {
//    m_logFile << "Event with " << i<<" HF Digis passed." << std::endl;
  } 

  // Call the function every m_nevtsample events
  if(evt%m_nevtsample==0) LedSampleAnalysis();

}
//----------------------------------------------------------------------------
void HcalLedAnalysis::SetupLEDHists(int id, const HcalDetId detid, map<HcalDetId, map<int,LEDBUNCH> > &toolT) {

  string type = "HBHE";
  if(id==1) type = "HO";
  if(id==2) type = "HF";

  _meol = toolT.find(detid);
  if (_meol==toolT.end()){
// if histos for this channel do not exist, create them
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
    sprintf(name,"%s Integrated Signal, eta=%d phi=%d depth=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth());
    insert[16].first =  new TH1F(name,name,500,0.,5000.);

    toolT[detid] = insert;
  }
}
//-----------------------------------------------------------------------------
void HcalLedAnalysis::LedHBHEHists(const HcalDetId& detid, const HBHEDataFrame& ledDigi, map<HcalDetId, map<int,LEDBUNCH> > &toolT, const HcalDbService& cond){

  map<int,LEDBUNCH> _mei;
  _meol = toolT.find(detid);
  _mei = _meol->second;

  // Reset the histos if we're at the end of a 'bunch'
  if((evt-1)%m_nevtsample==0 && state[0]){
    for(int k=0; k<(int)state.size();k++) state[k]=false;
    for(int i=0; i<16; i++) _mei[i].first->Reset();
  }

  // Most of this is borrowed from HcalSimpleReconstructor, so thanks Jeremy/Phil


  //  int maxTS = -1;
  float max_fC = 0;
  float ta = 0;
  m_coder = cond.getHcalCoder(detid);
  m_ped = cond.getPedestal(detid);
  m_shape = cond.getHcalShape(m_coder);
  for (int TS = m_startTS; TS < m_endTS && TS < ledDigi.size(); TS++){
    int capid = ledDigi[TS].capid();
    int adc = ledDigi[TS].adc();
    double fC = m_coder->charge(*m_shape,adc,capid);
    ta = (fC - m_ped->getValue(capid));
    //cout << "DetID: " << detid << "  CapID: " << capid << "  ADC: " << adc << "  fC: " << fC << endl;
    _mei[TS].first->Fill(ta);
    _mei[10].first->AddBinContent(TS+1,ta);  // This is average pulse, could probably do better (Profile?)
    if(m_fitflag>1){
      if(TS==m_startTS)_mei[11].first->Reset();
      _mei[11].first->SetBinContent(TS+1,ta);
    }
    // keep track of max TS and max amplitude (in fC)
    if (ta > max_fC){
      max_fC = ta;
      //      maxTS = TS;
    }
  }

  // Now we have a sample with pedestals subtracted and in units of fC
  // If we are using a weighted mean (m_fitflag = 2) to extraxt timing
  // we now want to use Phil's timing correction.  This is not necessary
  // if we are performing a Landau fit (m_fitflag = 3)

  float sum=0.;
  for(int i=0; i<10; i++)sum=sum+_mei[11].first->GetBinContent(i+1);
  if(sum>100){
    if(m_fitflag==2 || m_fitflag==4){
      float timmean=_mei[11].first->GetMean();  // let's use Phil's way instead
      float timmeancorr=BinsizeCorr(timmean);
      _mei[12].first->Fill(timmeancorr);
    }
    _mei[16].first->Fill(_mei[11].first->Integral()); // Integrated charge (may be more usfull to convert to Energy first?)
    if(m_fitflag==3 || m_fitflag==4){
      _mei[11].first->Fit("landau","Q");
      TF1 *fit = _mei[11].first->GetFunction("landau");
      _mei[13].first->Fill(fit->GetParameter(1));
      _mei[14].first->Fill(fit->GetParError(1));
      _mei[15].first->Fill(fit->GetChisquare()/fit->GetNDF());
    }
  }

}

//-----------------------------------------------------------------------------
void HcalLedAnalysis::LedHOHists(const HcalDetId& detid, const HODataFrame& ledDigi, map<HcalDetId, map<int,LEDBUNCH> > &toolT, const HcalDbService& cond) {

  map<int,LEDBUNCH> _mei;
  _meol = toolT.find(detid);
  _mei = _meol->second;
  // Rest the histos if we're at the end of a 'bunch'
  if((evt-1)%m_nevtsample==0 && state[0]){
    for(int k=0; k<(int)state.size();k++) state[k]=false;
    for(int i=0; i<16; i++) _mei[i].first->Reset();
  }

  // now we have the signal in fC, let's get rid of that darn pedestal
  // Most of this is borrowed from HcalSimpleReconstructor, so thanks Jeremy/Phil

  //  int maxTS = -1;
  float max_fC = 0;
  float ta = 0;
  m_coder = cond.getHcalCoder(detid);
  m_ped = cond.getPedestal(detid);
  m_shape = cond.getHcalShape(m_coder);
  for (int TS = m_startTS; TS < m_endTS && TS < ledDigi.size(); TS++){
    int capid = ledDigi[TS].capid();
    int adc = ledDigi[TS].adc();
    double fC = m_coder->charge(*m_shape,adc,capid);
    ta = (fC - m_ped->getValue(capid));
    _mei[TS].first->Fill(ta);
    _mei[10].first->AddBinContent(TS+1,ta);  // This is average pulse, could probably do better (Profile?)
    if(m_fitflag>1){
      if(TS==m_startTS)_mei[11].first->Reset();
      _mei[11].first->SetBinContent(TS+1,ta);
    }
    // keep track of max TS and max amplitude (in fC)
    if (ta > max_fC){
      max_fC = ta;
      //      maxTS = TS;
    }
  }

  // Now we have a sample with pedestals subtracted and in units of fC
  // If we are using a weighted mean (m_fitflag = 2) to extraxt timing
  // we now want to use Phil's timing correction.  This is not necessary
  // if we are performing a Landau fit (m_fitflag = 3)

  float sum=0.;
  for(int i=0; i<10; i++)sum=sum+_mei[11].first->GetBinContent(i+1);
  if(sum>100){
    if(m_fitflag==2 || m_fitflag==4){
      float timmean=_mei[11].first->GetMean();  // let's use Phil's way instead
      float timmeancorr=BinsizeCorr(timmean);
      _mei[12].first->Fill(timmeancorr);
    }
    _mei[16].first->Fill(_mei[11].first->Integral()); // Integrated charge (may be more usfull to convert to Energy first?)
    if(m_fitflag==3 || m_fitflag==4){
      _mei[11].first->Fit("landau","Q");
      TF1 *fit = _mei[11].first->GetFunction("landau");
      _mei[13].first->Fill(fit->GetParameter(1));
      _mei[14].first->Fill(fit->GetParError(1));
      _mei[15].first->Fill(fit->GetChisquare()/fit->GetNDF());
    }
  }

}

//-----------------------------------------------------------------------------
void HcalLedAnalysis::LedHFHists(const HcalDetId& detid, const HFDataFrame& ledDigi, map<HcalDetId, map<int,LEDBUNCH> > &toolT, const HcalDbService& cond) {

  map<int,LEDBUNCH> _mei;
  _meol = toolT.find(detid);
  _mei = _meol->second;
  // Rest the histos if we're at the end of a 'bunch'
  if((evt-1)%m_nevtsample==0 && state[0]){
    for(int k=0; k<(int)state.size();k++) state[k]=false;
    for(int i=0; i<16; i++) _mei[i].first->Reset();
  }

  // now we have the signal in fC, let's get rid of that darn pedestal
  // Most of this is borrowed from HcalSimpleReconstructor, so thanks Jeremy/Phil

  //  int maxTS = -1;
  float max_fC = 0;
  float ta = 0;
  m_coder = cond.getHcalCoder(detid);
  m_ped = cond.getPedestal(detid);
  m_shape = cond.getHcalShape(m_coder);
  //cout << "New Digi!!!!!!!!!!!!!!!!!!!!!!" << endl;
  for (int TS = m_startTS; TS < m_endTS && TS < ledDigi.size(); TS++){
    int capid = ledDigi[TS].capid();
    // BE CAREFUL: this is assuming peds are stored in ADCs
    int adc = (int)(ledDigi[TS].adc() - m_ped->getValue(capid));
    if (adc < 0){ adc = 0; }  // to prevent negative adcs after ped subtraction, which should really only happen
                              // if you're using the wrong peds.
    double fC = m_coder->charge(*m_shape,adc,capid);
    //ta = (fC - m_ped->getValue(capid));
    ta = fC;
    //cout << "DetID: " << detid << "  CapID: " << capid << "  ADC: " << adc << "  Ped: " << m_ped->getValue(capid) << "  fC: " << fC << endl;
    _mei[TS].first->Fill(ta);
    _mei[10].first->AddBinContent(TS+1,ta);  // This is average pulse, could probably do better (Profile?)
    if(m_fitflag>1){
      if(TS==m_startTS)_mei[11].first->Reset();
      _mei[11].first->SetBinContent(TS+1,ta);
    }

    // keep track of max TS and max amplitude (in fC)
    if (ta > max_fC){
      max_fC = ta;
      //      maxTS = TS;
    }
  }

  // Now we have a sample with pedestals subtracted and in units of fC
  // If we are using a weighted mean (m_fitflag = 2) to extraxt timing
  // we now want to use Phil's timing correction.  This is not necessary
  // if we are performing a Landau fit (m_fitflag = 3)

  float sum=0.;
  for(int i=0; i<10; i++)sum=sum+_mei[11].first->GetBinContent(i+1);
  if(sum>100){
    if(m_fitflag==2 || m_fitflag==4){
      float timmean=_mei[11].first->GetMean();  // let's use Phil's way instead
      float timmeancorr=BinsizeCorr(timmean);
      _mei[12].first->Fill(timmeancorr);
    }
    _mei[16].first->Fill(_mei[11].first->Integral()); // Integrated charge (may be more usfull to convert to Energy first?)
    if(m_fitflag==3 || m_fitflag==4){
      _mei[11].first->Fit("landau","Q");
      TF1 *fit = _mei[11].first->GetFunction("landau");
      _mei[13].first->Fill(fit->GetParameter(1));
      _mei[14].first->Fill(fit->GetParError(1));
      _mei[15].first->Fill(fit->GetChisquare()/fit->GetNDF());
    }
  }



}


//-----------------------------------------------------------------------------
float HcalLedAnalysis::BinsizeCorr(float time) {

// this is the bin size correction to be applied for laser data (from Andy),
// it comes from a pulse shape measured from TB04 data (from Jordan)
// This should eventually be replaced with the more thorough treatment from Phil

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
//-----------------------------------------------------------------------------

// Will try to implement Phil's time slew correction here at some point


//-----------------------------------------------------------------------------
void HcalLedAnalysis::ProcessCalibEvent(int fiberChan, HcalCalibDetId calibId, const HcalCalibDataFrame digi){

  _meca = calibHists.find(calibId);
  if (_meca==calibHists.end()){
  // if histos for this channel do not exist, first create them
    char name[1024];
    std::string prefix;
    if (calibId.calibFlavor()==HcalCalibDetId::CalibrationBox) {
      std::string sector=(calibId.hcalSubdet()==HcalBarrel)?("HB"):
	(calibId.hcalSubdet()==HcalEndcap)?("HE"):
	(calibId.hcalSubdet()==HcalOuter)?("HO"):
	(calibId.hcalSubdet()==HcalForward)?("HF"):"";
      sprintf(name,"%s %+d iphi=%d %s",sector.c_str(),calibId.ieta(),calibId.iphi(),calibId.cboxChannelString().c_str());
      prefix=name;
    }
    
    sprintf(name,"%s Pin Diode Mean",prefix.c_str());
    calibHists[calibId].avePulse = new TProfile(name,name,10,-0.5,9.5,0,1000);
    sprintf(name,"%s Pin Diode Current Pulse",prefix.c_str());
    calibHists[calibId].thisPulse = new TH1F(name,name,10,-0.5,9.5);
    sprintf(name,"%s Pin Diode Integrated Pulse",prefix.c_str());
    calibHists[calibId].integPulse = new TH1F(name,name,200,0,500);    
  }
  else {
    for (int i=m_startTS; i<digi.size() && i<=m_endTS; i++) {
      calibHists[calibId].avePulse->Fill(i,digi.sample(i).adc());
      calibHists[calibId].thisPulse->SetBinContent(i+1,digi.sample(i).adc());
    }
    calibHists[calibId].integPulse->Fill(calibHists[calibId].thisPulse->Integral());
  }
}

