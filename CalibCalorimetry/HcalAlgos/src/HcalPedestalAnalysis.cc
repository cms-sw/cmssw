#include <CalibCalorimetry/HcalAlgos/interface/HcalPedestalAnalysis.h>
#include "CalibCalorimetry/HcalAlgos/interface/HcalAlgoUtils.h"
#include <iostream>
#include <fstream>


/*
 * \file HcalPedestalAnalysis.cc
 * 
 * $Date: 2006/01/05 19:55:32 $
 * $Revision: 1.1 $
 * \author W Fisher
 *
*/

HcalPedestalAnalysis::HcalPedestalAnalysis(const edm::ParameterSet& ps){

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

  m_startSample = ps.getUntrackedParameter<int>("firstSample", 0);
  if(m_startSample<0) m_startSample=0;
  m_endSample = ps.getUntrackedParameter<int>("lastSample", 10);

  hbHists.ALLPEDS = new TH1F("HB/HE All Pedestals","HB/HE All Pedestals",10,0,9);
  hbHists.PEDRMS= new TH1F("HB/HE All Pedestal RMS","HB/HE All Pedestal RMS",100,0,3);
  hbHists.PEDMEAN= new TH1F("HB/HE All Pedestal Means","HB/HE All Pedestal Means",100,0,9);
  hbHists.CAPIDRMS= new TH1F("HB/HE Per-Channel CAPID RMS Deviation","HB/HE Per-Channel CAPID RMS Deviation",50,0,0.3);
  hbHists.CAPIDMEAN= new TH1F("HB/HE Per-Channel CAPID Mean Deviation","HB/HE Per-Channel CAPID Mean Deviation",50,0,3);

  hoHists.ALLPEDS = new TH1F("HO All Pedestals","HO All Pedestals",10,0,9);
  hoHists.PEDRMS= new TH1F("HO All Pedestal RMS","HO All Pedestal RMS",100,0,3);
  hoHists.PEDMEAN= new TH1F("HO All Pedestal Means","HO All Pedestal Means",100,0,9);
  hoHists.CAPIDRMS= new TH1F("HO Per-Channel CAPID RMS Deviation","HO Per-Channel CAPID RMS Deviation",50,0,0.3);
  hoHists.CAPIDMEAN= new TH1F("HO Per-Channel CAPID Mean Deviation","HO Per-Channel CAPID Mean Deviation",50,0,3);

  hfHists.ALLPEDS = new TH1F("HF All Pedestals","HF All Pedestals",10,0,9);
  hfHists.PEDRMS= new TH1F("HF All Pedestal RMS","HF All Pedestal RMS",100,0,3);
  hfHists.PEDMEAN= new TH1F("HF All Pedestal Means","HF All Pedestal Means",100,0,9);
  hfHists.CAPIDRMS= new TH1F("HF Per-Channel CAPID RMS Deviation","HF Per-Channel CAPID RMS Deviation",50,0,0.3);
  hfHists.CAPIDMEAN= new TH1F("HF Per-Channel CAPID Mean Deviation","HF Per-Channel CAPID Mean Deviation",50,0,3);

}

HcalPedestalAnalysis::~HcalPedestalAnalysis(){
  ///All done, clean up!!
  for(_meo=hbHists.PEDVALS.begin(); _meo!=hbHists.PEDVALS.end(); _meo++){
    for(int i=0; i<4; i++) _meo->second[i]->Delete();
  }
  for(_meo=hoHists.PEDVALS.begin(); _meo!=hoHists.PEDVALS.end(); _meo++){
    for(int i=0; i<4; i++) _meo->second[i]->Delete();
  }
  for(_meo=hfHists.PEDVALS.begin(); _meo!=hfHists.PEDVALS.end(); _meo++){
    for(int i=0; i<4; i++) _meo->second[i]->Delete();
  }
  hbHists.ALLPEDS->Delete();
  hbHists.PEDRMS->Delete();
  hbHists.PEDMEAN->Delete();
  hbHists.CAPIDRMS->Delete();
  hbHists.CAPIDMEAN->Delete();

  hoHists.ALLPEDS->Delete();
  hoHists.PEDRMS->Delete();
  hoHists.PEDMEAN->Delete();
  hoHists.CAPIDRMS->Delete();
  hoHists.CAPIDMEAN->Delete();

  hfHists.ALLPEDS->Delete();
  hfHists.PEDRMS->Delete();
  hfHists.PEDMEAN->Delete();
  hfHists.CAPIDRMS->Delete();
  hfHists.CAPIDMEAN->Delete();

}

void HcalPedestalAnalysis::done() {

  HcalPedestals pedCan;
  HcalPedestalWidths widthCan;

  for(_meo=hbHists.PEDVALS.begin(); _meo!=hbHists.PEDVALS.end(); _meo++){
    pedCan.addValue(_meo->first, _meo->second[0]->GetMean(),_meo->second[1]->GetMean(),
		    _meo->second[2]->GetMean(),_meo->second[3]->GetMean());      
    widthCan.addValue(_meo->first, _meo->second[0]->GetRMS(),_meo->second[1]->GetRMS(),
		      _meo->second[2]->GetRMS(),_meo->second[3]->GetRMS());    
    for(int i=0; i<4; i++){
      hbHists.PEDMEAN->Fill(_meo->second[i]->GetMean());
      hbHists.PEDRMS->Fill(_meo->second[i]->GetRMS());
    }
    hbHists.CAPIDMEAN->Fill(maxDiff(_meo->second[0]->GetMean(),_meo->second[1]->GetMean(),
				   _meo->second[2]->GetMean(),_meo->second[3]->GetMean()));
    hbHists.CAPIDRMS->Fill(maxDiff(_meo->second[0]->GetRMS(),_meo->second[1]->GetRMS(),
				   _meo->second[2]->GetRMS(),_meo->second[3]->GetRMS()));
  }
  for(_meo=hoHists.PEDVALS.begin(); _meo!=hoHists.PEDVALS.end(); _meo++){
    pedCan.addValue(_meo->first, _meo->second[0]->GetMean(),_meo->second[1]->GetMean(),
		    _meo->second[2]->GetMean(),_meo->second[3]->GetMean()); 
    widthCan.addValue(_meo->first, _meo->second[0]->GetRMS(),_meo->second[1]->GetRMS(),
		      _meo->second[2]->GetRMS(),_meo->second[3]->GetRMS());
    for(int i=0; i<4; i++){
      hoHists.PEDMEAN->Fill(_meo->second[i]->GetMean());
      hoHists.PEDRMS->Fill(_meo->second[i]->GetRMS());
    }
    hoHists.CAPIDMEAN->Fill(maxDiff(_meo->second[0]->GetMean(),_meo->second[1]->GetMean(),
				   _meo->second[2]->GetMean(),_meo->second[3]->GetMean()));
    hoHists.CAPIDRMS->Fill(maxDiff(_meo->second[0]->GetRMS(),_meo->second[1]->GetRMS(),
				   _meo->second[2]->GetRMS(),_meo->second[3]->GetRMS()));
  }
  for(_meo=hfHists.PEDVALS.begin(); _meo!=hfHists.PEDVALS.end(); _meo++){
    pedCan.addValue(_meo->first, _meo->second[0]->GetMean(),_meo->second[1]->GetMean(),
		    _meo->second[2]->GetMean(),_meo->second[3]->GetMean());   
    widthCan.addValue(_meo->first, _meo->second[0]->GetRMS(),_meo->second[1]->GetRMS(),
		      _meo->second[2]->GetRMS(),_meo->second[3]->GetRMS());
    for(int i=0; i<4; i++){
      hfHists.PEDMEAN->Fill(_meo->second[i]->GetMean());
      hfHists.PEDRMS->Fill(_meo->second[i]->GetRMS());
    }
    hfHists.CAPIDMEAN->Fill(maxDiff(_meo->second[0]->GetMean(),_meo->second[1]->GetMean(),
				   _meo->second[2]->GetMean(),_meo->second[3]->GetMean()));
    hfHists.CAPIDRMS->Fill(maxDiff(_meo->second[0]->GetRMS(),_meo->second[1]->GetRMS(),
				   _meo->second[2]->GetRMS(),_meo->second[3]->GetRMS()));
  }
  pedCan.sort();
  widthCan.sort();
  
  if(m_outputFileMean.size() != 0 ) {
    filebuf fb;
    fb.open (m_outputFileMean.c_str(),ios::out);
    ostream os(&fb);
    HcalDbASCIIIO::dumpObject(os,pedCan);
    fb.close();
    cout << "Hcal pedestal means written to " << m_outputFileMean.c_str() << endl;
  }

  if(m_outputFileWidth.size() != 0 ) {
    filebuf fb;
    fb.open (m_outputFileWidth.c_str(),ios::out);
    ostream os(&fb);
    HcalDbASCIIIO::dumpObject(os,widthCan);
    fb.close();
    cout << "Hcal pedestal means written to " << m_outputFileWidth.c_str() << endl;
  }
  
  if(m_outputFileROOT.size() != 0 ) {
    TFile out(m_outputFileROOT.c_str(),"RECREATE");
    for(_meo=hbHists.PEDVALS.begin(); _meo!=hbHists.PEDVALS.end(); _meo++){
      for(int i=0; i<4; i++) _meo->second[i]->Write();
    }
    for(_meo=hoHists.PEDVALS.begin(); _meo!=hoHists.PEDVALS.end(); _meo++){
      for(int i=0; i<4; i++) _meo->second[i]->Write();
    }
    for(_meo=hfHists.PEDVALS.begin(); _meo!=hfHists.PEDVALS.end(); _meo++){
      for(int i=0; i<4; i++) _meo->second[i]->Write();
    }

    hbHists.ALLPEDS->Write();
    hbHists.PEDRMS->Write();
    hbHists.PEDMEAN->Write();
    hbHists.CAPIDRMS->Write();
    hbHists.CAPIDMEAN->Write();
    
    hoHists.ALLPEDS->Write();
    hoHists.PEDRMS->Write();
    hoHists.PEDMEAN->Write();
    hoHists.CAPIDRMS->Write();
    hoHists.CAPIDMEAN->Write();
    
    hfHists.ALLPEDS->Write();
    hfHists.PEDRMS->Write();
    hfHists.PEDMEAN->Write();
    hfHists.CAPIDRMS->Write();
    hfHists.CAPIDMEAN->Write();
        
    out.Write();
    out.Close();
    cout << "Hcal histograms written to " << m_outputFileROOT.c_str() << endl;
  }

}

void HcalPedestalAnalysis::processEvent(const HBHEDigiCollection& hbhe,
					const HODigiCollection& ho,
					const HFDigiCollection& hf,
					const HcalDbService& cond){
  
  m_shape = cond.getHcalShape();
  
  try{
    for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); j++){
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      m_coder = cond.getHcalCoder(digi.id());
      for (int i=m_startSample; i<digi.size() && i<m_endSample; i++) {
	perChanHists(0,digi.id(),digi.sample(i),hbHists.PEDVALS);
      }
    }
  } catch (...) {
    printf("HcalPedestalAnalysis::processEvent  No HB/HE Digis.\n");
  }
  
  try{
    for (HODigiCollection::const_iterator j=ho.begin(); j!=ho.end(); j++){
      const HODataFrame digi = (const HODataFrame)(*j);
      m_coder = cond.getHcalCoder(digi.id());
      for (int i=m_startSample; i<digi.size() && i<m_endSample; i++) {	   
	perChanHists(1,digi.id(),digi.sample(i),hoHists.PEDVALS);
      }
    }        
  } catch (...) {
    cout << "HcalPedestalAnalysis::processEvent  No HO Digis." << endl;
  }
  
  try{
    for (HFDigiCollection::const_iterator j=hf.begin(); j!=hf.end(); j++){
      const HFDataFrame digi = (const HFDataFrame)(*j);
      m_coder = cond.getHcalCoder(digi.id());
      for (int i=m_startSample; i<digi.size() && i<m_endSample; i++) {
	perChanHists(2,digi.id(),digi.sample(i),hfHists.PEDVALS);
      }
    }
  } catch (...) {
    cout << "HcalPedestalAnalysis::processEvent  No HF Digis." << endl;
  }
  
}

void HcalPedestalAnalysis::perChanHists(int id, const HcalDetId detid, const HcalQIESample& qie, map<HcalDetId, map<int, TH1F*> > &tool) {
  static const int bins=10;

  map<int,TH1F*> _mei;
  
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

  ///outer iteration
  _meo = tool.find(detid);
  if (_meo!=tool.end()){
    //inner iteration
    _mei = _meo->second;
    if(_mei[qie.capid()]==NULL) printf("HcalPedestalAnalysis::perChanHists  This histo is NULL!!??\n");
    else if (qie.adc()<bins) _mei[qie.capid()]->AddBinContent(qie.adc()+1,1);
  }
  else{
    map<int,TH1F*> insert;
    float lo=3; float hi=4;
    for(int i=0; i<4; i++){
      char name[1024];
      sprintf(name,"%s Pedestal Value, ieta=%d iphi=%d depth=%d CAPID=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth(),i);     
      getLinearizedADC(*m_shape,m_coder,bins,i,lo,hi);
      //      printf("Linearized: lo: %f, hi: %f\n",lo,hi);
      insert[i] =  new TH1F(name,name,bins,lo,hi); 
    }
    if (qie.adc()<bins) insert[qie.capid()]->AddBinContent(qie.adc()+1,1);
    tool[detid] = insert;
  }

  if(id==0) hbHists.ALLPEDS->Fill(qie.adc());
  else if(id==1) hoHists.ALLPEDS->Fill(qie.adc());
  else if(id==2) hfHists.ALLPEDS->Fill(qie.adc());

}
