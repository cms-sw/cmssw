/*
  displays timing monitoring histograms
   produced by BxTiming DQM source module
  skips empty histograms
  nuno.leonardo@cern.ch 08.03
*/

const int nspr_=3;
const int NSYS =9;
const int nfed=1000;
const int nttype_ = 6; 
const int nbit=1;
int listGtBits_[nbit] = {0};

std::string SysLabel[NSYS] = {
  "ECAL", "HCAL", "GCT", "CSCTPG", "CSCTF", "DTTPG", "DTTF", "RPC", "GT"
}
std::string spreadLabel[nspr_] = {"Spread","Min", "Max"};
Color_t     spreadColor[nspr_] = {'',kRed, kBlue};

TProfile* bxDiffAllFed;
TProfile* bxDiffSysFed[NSYS];
TH1F* bxDiffAllFedSpread[nspr_];
TH1F* bxOccyAllFedSpread[nspr_];
TH1F* bxOccyAllFed;
TH1F** bxOccyOneFed; //[nfed];

TH1F* bxOccyGtTrigType[nttype_];
TH1F**bxOccyTrigBit[NSYS]; //[ntbit]

bool empty(TH1* h) {return h->GetEntries()==0;}
bool empty(TProfile* h) {return h->GetEntries()==0;}

void Print(TH1F* h, TCanvas *cvs, TString of) {
  if(empty(h)) return; 
  h->Draw();
  cvs->Print(of);
}
void Print(TProfile* h, TCanvas *cvs, TString of) {
  if(empty(h)) return; 
  h->Draw();
  cvs->Print(of);
}

void plotBx(TString finput = "l1timing.root", bool saveps = true ) {

  TFile *infile = new TFile(finput);
  TDirectory* tdir = infile->GetDirectory("DQMData/L1T/BXSynch");
  gInterpreter->ExecuteMacro("/afs/cern.ch/user/n/nuno/public/ScanStyle.C");

  std::string lbl("");

  bxDiffAllFed = (TProfile*) tdir->Get("BxDiffAllFed");
  bxOccyAllFed = (TH1F*) tdir->Get("BxOccyAllFed");

  for(int i=0; i<NSYS; i++) {
      lbl.clear();lbl+=SysLabel[i];lbl+="FedBxDiff"; 
      bxDiffSysFed[i] = (TProfile*) tdir->Get(lbl.data());
  }
  
  for(int i=0; i<nspr_; i++) {
    lbl.clear();lbl+="BxDiffAllFed";lbl+=spreadLabel[i];
    bxDiffAllFedSpread[i] = (TH1F*) tdir->Get(lbl.data());
    lbl.clear();lbl+="BxOccyAllFed";lbl+=spreadLabel[i];
    bxOccyAllFedSpread[i] = (TH1F*) tdir->Get(lbl.data());
  }

  TDirectory* tdirs 
    = infile->GetDirectory("DQMData/L1T/BXSynch/SingleFed/");
  bxOccyOneFed = new TH1F*[nfed];
  for(int i=0; i<nfed; i++) {
    lbl.clear(); lbl+="BxOccyOneFed";
    char *ii = new char[1000]; std::sprintf(ii,"%d",i);lbl+=ii;
    bxOccyOneFed[i] = tdirs->Get(lbl.data());
  }

  for(int i=0; i<nttype_; i++) {
    lbl.clear();lbl+="BxOccyGtTrigType";
    char *ii = new char[10]; std::sprintf(ii,"%d",i+1);lbl+=ii;
    bxOccyGtTrigType[i] = (TH1F*) tdir->Get(lbl.data());
  }  


  TDirectory* tdirb
    = infile->GetDirectory("DQMData/L1T/BXSynch/SingleBit/");
  for(int i=0; i<NSYS; i++) {
    bxOccyTrigBit[i] = new TH1F*[nbit];
    for(int j=0; j<nbit; j++) {
      lbl.clear();lbl+=SysLabel[i];lbl+="BxOccyGtBit"; 
      char *ii = new char[1000]; std::sprintf(ii,"%d",listGtBits_[j]); lbl+=ii;
      bxOccyTrigBit[i][j] = tdirb->Get(lbl.data());
    }
  }

  TCanvas *cvsallfed = new TCanvas("all fed","all fed",0,0,500,500);
  cvsallfed->Divide(3,3);
  TCanvas *cvsdiff = new TCanvas("bx diff","bx diff",500,0,500,500);
  cvsdiff->Divide(3,3);
  TCanvas *cvsonefed = new TCanvas("one fed","one fed",1000,0,500,1000);
  cvsonefed->Divide(5,10);
  TCanvas *cvsttype = new TCanvas("ttype","ttype",0,100,500,500);
  cvsttype->Divide(3,2);
  TCanvas *cvstbit = new TCanvas("tbit","tbit",0,100,500,500);
  cvstbit->Divide(9,2);

  for(int i=0; i<NSYS; i++) {
    cvsdiff->cd(i+1);
    bxDiffSysFed[i]->Draw();
  }

  cvsallfed->cd(1);
  bxDiffAllFed->Draw();
  cvsallfed->cd(2);
  bxOccyAllFed->Draw();

  for(int i=0; i<nspr_; i++) {
    cvsallfed->cd(4+i);
    bxDiffAllFedSpread[i]->SetLineColor(spreadColor[i]);
    bxDiffAllFedSpread[i]->Draw();
  }
  for(int i=0; i<nspr_; i++) {
    cvsallfed->cd(7+i);
    bxOccyAllFedSpread[i]->SetLineColor(spreadColor[i]);
    bxOccyAllFedSpread[i]->Draw();
  }

  int ki=0;
  for(int i=0; i<nfed; i++) {
    if(empty(bxOccyOneFed[i])) continue;
    cvsonefed->cd(++ki);
    bxOccyOneFed[i]->Draw("p");
  }

  ki=0;
  for(int i=0; i<nttype_; i++) {
    cvsttype->cd(++ki);
    bxOccyGtTrigType[i]->Draw();
  }  
  
  ki=0;
  for(int i=0; i<NSYS; i++) {
    for(int j=0; j<nbit; j++) {
      cvstbit->cd(++ki);
      bxOccyTrigBit[i][j]->Draw();
    }
  }

  if(!saveps)
    return;

  cout << "printing histos...\n" << flush;
  
  TString dir("figures/");
  dir += "timingplots";
  TString ofile(dir); ofile += ".ps";

  TCanvas *cvs = new TCanvas("teste","teste",0,0,500,450);
  cvs->Print(TString(ofile+"["));

  Print ( bxDiffAllFed ,cvs,ofile);
  for(int i=0; i<NSYS; i++)
    Print ( bxDiffSysFed[i] ,cvs,ofile);

  Print ( bxOccyAllFed ,cvs,ofile);
  for(int i=0; i<nspr_; i++)
    Print ( bxDiffAllFedSpread[i],cvs,ofile);
  for(int i=0; i<nspr_; i++)
    Print ( bxOccyAllFedSpread[i],cvs,ofile);
  for(int i=0; i<nttype_; i++)
    Print (bxOccyGtTrigType[i],cvs,ofile);
  for(int i=0; i<NSYS; i++)
    for(int j=0; j<nbit; j++)
      Print (bxOccyTrigBit[i][j],cvs,ofile);
  for(int i=0; i<nfed; i++)
      Print (bxOccyOneFed[i],cvs,ofile);
  
  cvs->Print(TString(ofile+"]"));

  cout << "done printing histos.\n" << flush;

  return;
}

