//-------------------------------------------------
//
/**  \class DTTrigFineSync
 *
 *   Analyzer used to generate BTI fine sync
 *   parameters
 *
 *
 *   $Date: 2006/09/12 $
 *   $Revision: 1.1 $
 *
 *   \author C. Battilana
 */
//
//--------------------------------------------------

// This class's header
#include <L1Trigger/DTTrigger/interface/DTTrigFineSync.h>

// C++ headers
#include <iostream>
#include <math.h>

const double DTTrigFineSync::myTtoTDC = 32./25.;

DTTrigFineSync::DTTrigFineSync (const ParameterSet& pset){ 
  
   MyTrig = new DTTrig();
   string rootfilename = pset.getUntrackedParameter<string>("rootFileName");
   string txtfilename = pset.getUntrackedParameter<string>("outputFileName");
   string cfgfilename = pset.getUntrackedParameter<string>("cfgFileName");
   CorrectBX = pset.getUntrackedParameter<int>("correctBX");  // Default is 16
   FTStep = pset.getUntrackedParameter<double>("offsetStep");       // Default is 0.104 ns
   string rootext = ".root";
   cout << "****Opening/Creating files" << endl;
   rootfile = new TFile((rootfilename+rootext).c_str(),"RECREATE");
   txtfile = new fstream;
   txtfile->open(txtfilename.c_str(),ios::out|ios::trunc);
   cfgfile = new fstream;
   cfgfile->open(cfgfilename.c_str(),ios::out|ios::trunc);
   cout << "****Introducing steps in Config" << endl;
  for (int i=1;i<25;i++){
    stringstream myos;
    myos<< i*10.;
    //MyTrig->config()->setParamValue("BTI Fine sync delay",myos.str(),i*10.);
    float time = i*10*FTStep* myTtoTDC; 
    MyTrig->config()->setParamValue("BTI setup time",myos.str(),time);
  }
  double syncdelay = pset.getUntrackedParameter<double>("syncDelay");  // Default is 500 ns
  stringstream myos;
  myos << syncdelay;
  MyTrig->config()->setParam("Programmable Dealy",myos.str());
  //     MyTrig->config()->setParam("Debugging level","fullTRACO");
  //     cout << "Step Value: " << MyTrig->config()->FTStep() << endl;
  //     cout << "Step Width: " << MyTrig->config()->FTStepW() << endl;
  
}


DTTrigFineSync::~DTTrigFineSync(){ 
  
  delete MyTrig;
  delete rootfile;
  delete txtfile;
  
}

void DTTrigFineSync::endJob(){

  TObjArray Hlist(0);
  float hbx=24.95*.5;
  // Creating histograms (all stations sum)
  TH1F *HH_all       = new TH1F ("all_HH","HH distibution for All Chambers",25,-.5,24.5);          // HH Histo
  TH1F *HL_all       = new TH1F ("all_HL","HL distibution for All Chambers",25,-.5,24.5);          // HL Histo
  TH1F *HLoverHH_all = new TH1F ("all_HLoverHH","HL/HH distibution for All Chambers",25,-.5,24.5); // HHoverHL Histo
  // Writing first part of cfg file
  (*cfgfile) << "untracked PSet L1DTFineSync = {" << endl;
  
  for (DelayIterator iQual=QualMap.begin(); iQual!=QualMap.end(); iQual++){
    // Creating histograms (one for each station)
    stringstream os;
    stringstream osHH;
    stringstream osHL;
    stringstream osHLoverHH;
    stringstream osBX;
    os<< (*iQual).first.wheel() << "sect" << (*iQual).first.sector() << "st" << (*iQual).first.station();  
    osHH << "HH_wh" <<os.str();
    string HHname = osHH.str();      
    osHL << "HL_wh" <<os.str(); 
    string HLname = osHL.str();
    osHLoverHH << "HLoverHH_wh" <<os.str();
    string HLoverHHname = osHLoverHH.str();
    osBX << "best_delay_BX_wh" <<os.str();
    string BXname = osBX.str();
    TH1F *HH       = new TH1F (HHname.c_str(),HHname.c_str(),25,-.5,24.5);               // HH Histo
    TH1F *HL       = new TH1F (HLname.c_str(),HLname.c_str(),25,-.5,24.5);               // HL Histo
    TH1F *HLoverHH = new TH1F (HLoverHHname.c_str(),HLoverHHname.c_str(),25,-.5,24.5);   // HHoverHL Histo
    TH1F *BX       = new TH1F (BXname.c_str(),BXname.c_str(),25,-.5,24.5);               // BX Histo for HH segments at correct delay
    
    // Filling sigle station histograms
    for (int i=0;i<25;i++){
      int   iHH       = (*iQual).second.nHH[i];
      int   iHL       = (*iQual).second.nHL[i];
      HH->SetBinContent(HH->GetBin(i+1),iHH);
      HH_all->AddBinContent(HH_all->GetBin(i+1),iHH);
      HL->SetBinContent(HL->GetBin(i+1),iHL);
      HL_all->AddBinContent(HL_all->GetBin(i+1),iHL);
      if ((*iQual).second.nHH[i]>0) {
	float iHLoverHH = iHL/(float)iHH;
	HLoverHH->SetBinContent(HLoverHH->GetBin(i+1),iHLoverHH);
	HLoverHH_all->AddBinContent(HLoverHH_all->GetBin(i+1),iHLoverHH);
      }
      else  {
	HLoverHH->SetBinContent(HLoverHH->GetBin(i+1),1);
	HLoverHH_all->AddBinContent(HLoverHH_all->GetBin(i+1),1);
      }
    }
    
    // Adding histogram to root Hlist and writing single station record to output file
    Hlist.Add(HH);  
    Hlist.Add(HL); 
    Hlist.Add(HLoverHH);
    (*txtfile) << (*iQual).first.wheel() 
	       << "\t" << (*iQual).first.sector() 
	       << "\t" <<  (*iQual).first.station(); 
    float posMax =  HLoverHH->GetBinCenter(HLoverHH->GetMaximumBin());
    posMax *= FTStep*10;
    if ( posMax < hbx ) posMax += hbx; // Finding sync delay
    else posMax -= hbx;                // inside the BX
    posMax = ((int)(posMax/FTStep))*FTStep;
    (*txtfile) << "\t" << posMax ; 
    for (int i=0;i<25;i++){
      BX->SetBinContent(BX->GetBin(i+1),(*iQual).second.nBX[(int)posMax][i]);	
    }
    float BXMax =  BX->GetBinCenter(BX->GetMaximumBin());
    float BXcorr = ((int)(CorrectBX-BXMax))*hbx*2;  
    (*cfgfile) << "\t double wh" <<(*iQual).first.wheel() 
	       << "se" <<  (*iQual).first.sector() 
	       << "st" <<  (*iQual).first.station() 
	       << " = " << (BXcorr+posMax)
	       << " // time in ns " << BXcorr+posMax << endl;
    (*txtfile) << "\t" << BXcorr << endl ;
    Hlist.Add(BX);
  }
  
  // Adding histograms of all station to Hlist and writing all stations record to output file
  Hlist.Add(HH_all);
  Hlist.Add(HL_all);
  Hlist.Add(HLoverHH_all);
  float posMax =  HLoverHH_all->GetBinCenter(HLoverHH_all->GetMaximumBin());    
  posMax *= FTStep*10;
  if ( posMax < hbx ) posMax += hbx;
  else posMax -= hbx;
  posMax = ((int)(posMax/FTStep))*FTStep;
  
  (*txtfile) << "All Stations :" << posMax  << endl;
  (*cfgfile) << "}" << endl;
  cout << "****Writing Histograms and Closing Files" << endl;
  Hlist.Write();
  rootfile->Close();
  txtfile->close();
  
}


void DTTrigFineSync::beginJob(const EventSetup & iEventSetup){
  
  cout << "****Creating TU's" << endl;
  MyTrig->createTUs(iEventSetup);
  QualArr myQA;
  for (int i=0;i<25;i++){
    myQA.nHH[i]=0;
    myQA.nHL[i]=0;
    for (int j=0;j<25;j++)myQA.nBX[i][j]=0;
  }
  cout << "****Populating synchronization map" << endl;
  // Populate the synchronization quality map
  edm::ESHandle<DTGeometry>pDD;
  iEventSetup.get<MuonGeometryRecord>().get(pDD);
  for (std::vector<DTChamber*>::const_iterator ich=pDD->chambers().begin(); ich!=pDD->chambers().end();ich++){      
    DTChamber* chamb = (*ich);
    DTChamberId chid = chamb->id();
    QualMap[chid] = myQA;
  }
  
}
  

void DTTrigFineSync::analyze(const Event & iEvent, const EventSetup& iEventSetup){
  // Runnig trigger algorithm for all the possible delays 
  for (int istep=0;istep<25;istep++){
    stringstream os;
    os << istep*10.;
    MyTrig->config()->setParam("BTI setup time",os.str());
    //  cout << "Delay Value: " << MyTrig->config()->FTStep() << endl;
    MyTrig->triggerReco(iEvent,iEventSetup);
    vector<DTChambPhSegm> phitr = MyTrig->TSPhTrigs();
    //cout << phitr.size() << " phi triggers found" << endl;
    for (vector<DTChambPhSegm>::const_iterator iphitr=phitr.begin();iphitr!=phitr.end();iphitr++){
      // Filling synchronization map
      if (iphitr->code()==6 && iphitr->isFirst()==1)  {
	QualMap[iphitr->ChamberId()].nHH[istep]++; 
	QualMap[iphitr->ChamberId()].nBX[istep][iphitr->step()]++;
      }
      if (iphitr->code()==5 && iphitr->isFirst()==1) QualMap[iphitr->ChamberId()].nHL[istep]++; 
    }
  }

}


