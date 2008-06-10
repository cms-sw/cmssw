/*
  \description
    displays l1 trigger monitoring histograms
    discards empty histograms
  \author nuno.leonardo@cern.ch 07.10

  ---
  macro input and flags:
    - input file name (argument)
    - run number
    - save_histos = 0: do not save histos
                    1: append to single file
    - save_single = 0: do not save individual histos
                    1: save individual histos as .eps
                    2: save individual histos as .gif
                    3: save individual histos as .jpg

                    saves into assumed directory structure
		    figures/run-<x>/<sub-system>
		    figures/run-<x>/CORR
*/

bool save_histos = 1;
int  save_single = 0;  // 0.none, 1.eps, 2.gif, 3.jpg
//TString run("grumm38414,38478");
const int run = 0;//38414;

const int DEnsys = 12;

enum syslist {
  ETP, HTP, RCT, GCT, DTP, DTF, 
  CTP, CTF, RPC, LTC, GMT, GLT
};

const std::string SystLabelExt[DEnsys] = {
  "ECAL",   "HCAL",  "RCT", "GCT", "DTTPG", "DTTF", 
  "CSCTPG", "CSCTF", "RPC", "LTC", "GMT",   "GT"
};

const std::string SystLabel[DEnsys] = {
  "ETP", "HTP", "RCT", "GCT", "DTP", "DTF", 
  "CTP", "CTF", "RPC", "LTC", "GMT", "GLT"
};


TH1F* rates;
TH1F* ncand [DEnsys];

TH1F* eta   [DEnsys];
TH1F* phi   [DEnsys];
TH1F* x3    [DEnsys];
TH2F* etaphi[DEnsys];

TH1F* etaData[DEnsys];
TH1F* phiData[DEnsys];
TH1F*  x3Data[DEnsys];
TH1F* rnkData[DEnsys];

TH1F* dword [DEnsys];
TH1F* eword [DEnsys];
TH1F* deword[DEnsys];
TH1F* masked[DEnsys];

TH1F* error;
TH1F* errors[DEnsys];

const int ncorr = 3;
TH2F* CORR[DEnsys][DEnsys][ncorr];

const int nhist = 8+4;
TAxis *xaxis[nhist][DEnsys], *yaxis[nhist][DEnsys];

// ------------------

void plotDE(TString finput = "l1demon.root") {

  TString dir("figures/");
  dir += "run-";
  dir += run;

  TFile *infile = new TFile(finput);
  TDirectory* tdir = infile->GetDirectory("DQMData/L1TEMU/");
  //gInterpreter->ExecuteMacro("/cmsnfshome0/nfshome0/nuno/sty/ScanStyle.C");
  gInterpreter->ExecuteMacro("/afs/cern.ch/user/n/nuno/style/ScanStyle.C");


  TDirectory* tdircom = tdir->GetDirectory("common/");
  rates = (TH1F*) tdircom->Get("sysrates");
  error = (TH1F*) tdircom->Get("errorflag");

  //gDirectory->ls();
  //cvs->SetLogy();
  //error->Draw();
  //return;

  for(int j=0; j<2; j++) {
    std::string lbl("");
    lbl += "sysncand";
    lbl += (j==0?"Data":"Emul");
    ncand [j] = (TH1F*) tdircom->Get(lbl.data());
  }


  cout << "getting histos\t";
  for(int j=0; j<DEnsys; j++) {

    cout << " ." << flush;
    
    TDirectory* tdirsys = tdir->GetDirectory(SystLabelExt[j].data());
    
    std::string lbl("");
    
    lbl.clear();
    lbl+=SystLabel[j];lbl+="eta"; 
    eta   [j] = (TH1F*) tdirsys->Get(lbl.data());

    lbl.clear();
    lbl+=SystLabel[j];lbl+="phi"; 
    phi   [j] = (TH1F*) tdirsys->Get(lbl.data());
    lbl.clear();
    lbl+=SystLabel[j];lbl+="x3"; 
    x3    [j] = (TH1F*) tdirsys->Get(lbl.data());

    lbl.clear();
    lbl+=SystLabel[j];lbl+="etaphi"; 
    etaphi[j] = (TH2F*) tdirsys->Get(lbl.data());

    lbl.clear();
    lbl+=SystLabel[j];lbl+="dword"; 
    dword [j] = (TH1F*) tdirsys->Get(lbl.data());

    lbl.clear();
    lbl+=SystLabel[j];lbl+="eta"; lbl+="Data"; 
    etaData[j] = (TH1F*) tdirsys->Get(lbl.data());
    lbl.clear();
    lbl+=SystLabel[j];lbl+="phi"; lbl+="Data";  
    phiData[j] = (TH1F*) tdirsys->Get(lbl.data());
    lbl.clear();
    lbl+=SystLabel[j];lbl+="x3"; lbl+="Data";  
    x3Data [j] = (TH1F*) tdirsys->Get(lbl.data());
    lbl.clear();
    lbl+=SystLabel[j];lbl+="rank"; lbl+="Data";
    rnkData[j] = (TH1F*) tdirsys->Get(lbl.data());


    lbl.clear();
    lbl+=SystLabel[j];lbl+="eword"; 
    eword [j] = (TH1F*) tdirsys->Get(lbl.data());
    lbl.clear();
    lbl+=SystLabel[j];lbl+="deword"; 
    deword[j] = (TH1F*) tdirsys->Get(lbl.data());
    lbl.clear();
    lbl+=SystLabel[j];lbl+="masked"; 
    masked[j] = (TH1F*) tdirsys->Get(lbl.data());
    lbl.clear();
    lbl+=SystLabel[j];lbl+="ErrorFlag"; 
    errors[j] = (TH1F*) tdirsys->Get(lbl.data());

  }
  cout << "\tdone.\n";

  TString corrl[ncorr] = {"phi","eta","rank"};

  TDirectory* tdirsys = tdir->GetDirectory("xcorr");
  
  for(int i=0; i<DEnsys; i++) {
    for(int j=0; j<DEnsys; j++) {
      if(i>j) continue;
      for(int k=0; k<ncorr; k++) {
	std::string lbl("");
	lbl.clear(); lbl+=SystLabel[i]; lbl+=SystLabel[j]; lbl+=corrl[k]; 
	CORR[i][j][k] = (TH2F*) tdirsys->Get(lbl.data());
      }
    }
  }

  TAxis *xcorr[DEnsys][DEnsys][ncorr], *ycorr[DEnsys][DEnsys][ncorr];
  for(int i=0; i<DEnsys; i++) {
    for(int j=0; j<DEnsys; j++) {
      if(i>j) continue;
      for(int k=0; k<ncorr; k++) {
	std::string xlbl("");xlbl+=SystLabel[i];xlbl+=corrl[k];
	std::string ylbl("");ylbl+=SystLabel[j];ylbl+=corrl[k];
	axis2F(CORR[i][j][k], xcorr[i][j][k],ycorr[i][j][k],xlbl,ylbl);
      }
    }
  }
  
  const int nglobal = 4;
  TAxis *xglb[nglobal], *yglb[nglobal];

  axis1F(rates,    xglb[0],yglb[0],"L1 trigger subsystem","error rate");
  axis1F(ncand[0], xglb[1],yglb[1],"L1 trigger subsystem","");
  axis1F(ncand[1], xglb[2],yglb[2],"L1 trigger subsystem","");
  axis1F(error   , xglb[3],yglb[3],"error flag", "");
  for(int j=0; j<3; j++)
    for(int i=0; i<DEnsys+1; i++) 
      xglb[j]->SetBinLabel(i+1,SystLabel[i].c_str());
  
  const int nerr=5;
  TString errLabel[nerr]= {
    "Agree", "Loc. Agree", "L.Disagree", "Data only", "Emul only"
  }
  TString errLabelB[nerr]= {
    "Agree", "Loc. Agree", "Loc. Disagree", "D only", "E only"
  }
  for(int i=0; i<nerr; i++)
    xglb[3]->SetBinLabel(i+1,errLabel[i].Data());


  TAxis *xerr[DEnsys], *yerr[DEnsys];
  axis1F(errors[ 0],    xerr[ 0],yerr[ 0],"","");
  axis1F(errors[ 1],    xerr[ 1],yerr[ 1],"","");
  axis1F(errors[ 2],    xerr[ 2],yerr[ 2],"","");
  axis1F(errors[ 3],    xerr[ 3],yerr[ 3],"","");
  axis1F(errors[ 4],    xerr[ 4],yerr[ 4],"","");
  axis1F(errors[ 5],    xerr[ 5],yerr[ 5],"","");
  axis1F(errors[ 6],    xerr[ 6],yerr[ 6],"","");
  axis1F(errors[ 7],    xerr[ 7],yerr[ 7],"","");
  axis1F(errors[ 8],    xerr[ 8],yerr[ 8],"","");
  axis1F(errors[ 9],    xerr[ 9],yerr[ 9],"","");
  axis1F(errors[10],    xerr[10],yerr[10],"","");
  for(int j=0; j<10; j++) 
    for(int i=0; i<nerr; i++)
      xerr[j]->SetBinLabel(i+1,errLabelB[i].Data());

  TString xlabel[nhist] = {
    "eta",      	//eta 
    "phi",      	//phi   
    "x3",       	//x3    
    "eta",      	//etaphi
    "eta",      	//eta data
    "phi",      	//phi data  
    "x3",       	//x3  data    
    "rank",       	//rnk data    
    "trigger data word bit", 	//dword 
    "trigger data word bit", 	//eword 
    "trigger data word bit", 	//deword
    "trigger data word bit"  	//masked
  };

  TString ylabel[nhist] = {
    "",	//eta   
    "",	//phi   
    "",	//x3    
    "phi",	//etaphi
    "",	//eta data  
    "",	//phi data  
    "",	//x3  data  
    "",	//rnk data  
    "",	//dword 
    "",	//eword 
    "",	//deword
    "" 	//masked
   };
 

  cout << "setting axis\t" << flush;
  for(int j=0; j<DEnsys; j++) {
    cout << " ." << flush;
    axis1F(eta    [j]	, xaxis[ 0][j],yaxis[ 0][j],xlabel[ 0].Data(),ylabel[ 0].Data());
    axis1F(phi    [j]	, xaxis[ 1][j],yaxis[ 1][j],xlabel[ 1].Data(),ylabel[ 1].Data());
    axis1F(x3     [j]	, xaxis[ 2][j],yaxis[ 2][j],xlabel[ 2].Data(),ylabel[ 2].Data());
    axis2F(etaphi [j]	, xaxis[ 3][j],yaxis[ 3][j],xlabel[ 3].Data(),ylabel[ 3].Data());
    axis1F(etaData[j]	, xaxis[ 4][j],yaxis[ 4][j],xlabel[ 4].Data(),ylabel[ 4].Data());
    axis1F(phiData[j]	, xaxis[ 5][j],yaxis[ 5][j],xlabel[ 5].Data(),ylabel[ 5].Data());
    axis1F( x3Data[j]	, xaxis[ 6][j],yaxis[ 6][j],xlabel[ 6].Data(),ylabel[ 6].Data());
    axis1F(rnkData[j]	, xaxis[ 7][j],yaxis[ 7][j],xlabel[ 7].Data(),ylabel[ 7].Data());
    axis1F(dword  [j]	, xaxis[ 8][j],yaxis[ 8][j],xlabel[ 8].Data(),ylabel[ 8].Data());
    axis1F(eword  [j]	, xaxis[ 9][j],yaxis[ 9][j],xlabel[ 9].Data(),ylabel[ 9].Data());
    axis1F(deword [j]	, xaxis[10][j],yaxis[10][j],xlabel[10].Data(),ylabel[10].Data());
    axis1F(masked [j]	, xaxis[11][j],yaxis[11][j],xlabel[11].Data(),ylabel[11].Data());
  }
  cout << "\tdone.\n";

  if(!save_histos)
    return;

  cout << "printing histos...\n" << flush;
  
  TString ofile(dir); ofile += ".ps";

  //TCanvas *log = new TCanvas("testl","testl",0,0,500,450);
  //log->SetLogy();
  TCanvas *cvs = new TCanvas("teste","teste",0,0,500,450);

  cvs->Print(TString(ofile+"["));
  Print (rates   ,cvs,ofile);
  Print (ncand[0],cvs,ofile);
  Print (ncand[1],cvs,ofile);
  //cvs->GetPad(0)->SetLogy(1);
  Print (error   ,cvs,ofile);
  //cvs->GetPad(0)->SetLogy(0);
  Print2(rates   ,cvs,dir);

  Print2(ncand[0],cvs,dir);
  Print2(ncand[1],cvs,dir);
  Print2(error   ,cvs,dir);
  for(int j=0; j<DEnsys; j++) {
    //Print(errors  [j],log,ofile);
    Print(eta     [j],cvs,ofile);
    Print(phi     [j],cvs,ofile);
    Print(x3      [j],cvs,ofile);
    Print(etaphi  [j],cvs,ofile);
    Print(etaData [j],cvs,ofile);
    Print(phiData [j],cvs,ofile);
    Print( x3Data [j],cvs,ofile);
    Print(rnkData [j],cvs,ofile);
    Print(dword   [j],cvs,ofile);
    Print(eword   [j],cvs,ofile);
    Print(deword  [j],cvs,ofile);
    Print(masked  [j],cvs,ofile);
    TString name(dir);  name += "/"; name += SystLabel[j];
    Print2(errors [j],cvs,name);
    Print2(eta    [j],cvs,name);
    Print2(phi    [j],cvs,name);
    Print2(x3     [j],cvs,name);
    Print2(etaphi [j],cvs,name);
    Print2(etaData[j],cvs,name);
    Print2(phiData[j],cvs,name);
    Print2( x3Data[j],cvs,name);
    Print2(rnkData[j],cvs,name);
    Print2(dword  [j],cvs,name);
    Print2(eword  [j],cvs,name);
    Print2(deword [j],cvs,name);
    Print2(masked [j],cvs,name);
  }
  cvs->Print(TString(ofile+"]"));

  TCanvas *cor = new TCanvas("correlations","correlations",0,0,500,450);
  TString cfile(dir); cfile+= "-corr"; cfile += ".ps";
  cor->Print(TString(cfile+"["));
  for(int i=0; i<DEnsys; i++) {
    for(int j=0; j<DEnsys; j++) {
      if(i>j) continue;
      if(i==LTC || j==LTC) continue;
      for(int k=0; k<ncorr; k++) {
	Print(CORR[i][j][k],cor,cfile);
	TString name(dir);  name += "/CORR";
	Print2(CORR[i][j][k],cor,name);
      }
    }
  }
  cor->Print(TString(cfile+"]"));

  return;
}


/// histograming style

bool empty(TH1* h) {return h->GetEntries()==0;}
bool empty(TH2* h) {return h->GetEntries()==0;}

void axis1F(TH1F  *histo,
            TAxis *&xaxis,
	    TAxis *&yaxis,
	    char  *xtitle,
	    char  *ytitle) {

  histo->SetMarkerSize(0.5);
  histo->SetMarkerStyle(kFullCircle);
  xaxis = histo->GetXaxis();
  yaxis = histo->GetYaxis();
  xaxis->SetLabelFont(42);
  yaxis->SetLabelFont(42);
  xaxis->SetLabelOffset(0.015);
  yaxis->SetLabelOffset(0.020);
  xaxis->SetNdivisions(505);
  yaxis->SetNdivisions(505);
  xaxis->SetTitle(xtitle);
  yaxis->SetTitle(ytitle);
  xaxis->SetTitleColor(kBlack);
  yaxis->SetTitleColor(kBlack);  
  xaxis->SetTitleFont(42);
  yaxis->SetTitleFont(42);
  xaxis->SetTitleOffset(2);
  yaxis->SetTitleOffset(1.80);
  xaxis->SetLabelSize(0.055);
  yaxis->SetLabelSize(0.040);
  xaxis->SetTitleSize(0.055);
  yaxis->SetTitleSize(0.045);
  xaxis->SetTitleSize(0.05); //def:0.04
  yaxis->SetTitleSize(0.05); //def:0.04
}

void axis2F(TH2F    *histo,
            TAxis   *&xaxis,
	    TAxis   *&yaxis,
	    TString  xtitle,
	    TString  ytitle)
{
  xaxis = histo->GetXaxis();
  yaxis = histo->GetYaxis();
  xaxis->SetLabelFont(42);
  yaxis->SetLabelFont(42);
  xaxis->SetLabelOffset(0.01);
  yaxis->SetLabelOffset(0.02);
  xaxis->SetNdivisions(505);
  yaxis->SetNdivisions(505);
  xaxis->SetTitleFont(42);
  yaxis->SetTitleFont(42);
  xaxis->SetTitleOffset(1.3);
  yaxis->SetTitleOffset(1.3);
  xaxis->SetTitle(xtitle.Data());
  yaxis->SetTitle(ytitle.Data());
}

void Draw(TH1F *histo,   int    fColor, int fStyle,
          char *drawOpt, int    lColor, int lWidth,
	  int   mColor,  double mSize,  int mStyle)
{
  histo->SetDirectory(0     );
  histo->SetFillColor(fColor);
  histo->SetFillStyle(fStyle);
  histo->SetLineColor  (lColor);
  histo->SetLineWidth  (lWidth);
  histo->SetMarkerColor(mColor);
  histo->SetMarkerSize (mSize );
  histo->SetMarkerStyle(mStyle);
  if (histo != NULL) histo->Draw(drawOpt);
}

void Draw(TH2F *histo,   int    fColor, int fStyle,
          char *drawOpt, int    lColor, int lWidth,
	  int   mColor,  double mSize,  int mStyle)
{
  histo->SetDirectory(0     );
  histo->SetFillColor(fColor);
  histo->SetFillStyle(fStyle);
  histo->SetLineColor  (lColor);
  histo->SetLineWidth  (lWidth);
  histo->SetMarkerColor(mColor);
  histo->SetMarkerSize (mSize );
  histo->SetMarkerStyle(mStyle);
  if (histo != NULL) histo->Draw(drawOpt);
}


void Print(TH1F* h, TCanvas *cvs, TString of) {
  if(empty(h)) return; 
  Draw(h,   19,   1001,   "",    28, 1,   1,  0.8,  20);
  sign();   cvs->Print(of);
}
void Print2(TH1F* h, TCanvas *cvs, TString dir) {
  if(empty(h)) return; 
  Draw(h,   19,   1001,   "",  28, 1,   1,  0.8,  20);
  sign(); 
  TString name(dir); name += "/";  name += h->GetName();
  switch (save_single) {
  case 0: 
    return; break;
  case 1: 
    name+=".eps"; break;
  case 2: 
    name+=".gif"; break;
  case 3: 
    name+=".jpg"; break;
  }
  cvs->Print(name);
  /*
  TString namee(dir);  namee += "/";  namee += h->GetName();  namee += ".eps";
  TString nameg(dir);  nameg += "/";  nameg += h->GetName();  nameg += ".gif";
  TString namej(dir);  namej += "/";  namej += h->GetName();  namej += ".jpg";
  if(save_single==0)
    return;
  else if(save_single==1)
    cvs->Print(namee);
  else if(save_single==2)
    cvs->Print(nameg);
  else if(save_single==2)
    cvs->Print(nameg);
  */
}

void Print(TH2F* h, TCanvas *cvs, TString of) {
  if(empty(h)) return; 
  Draw(h,   38,   1001,   "col",  2, 1,   2,  0.8,  20);
  //surf1cyl,col, box, colz, contz, cont1-2
  sign();   cvs->Print(of);
}
void Print2(TH2F* h, TCanvas *cvs, TString dir) {
  if(empty(h)) return; 
  Draw(h,   38,   1001,   "col",  2, 1,   2,  0.8,  20);
  sign(); 
  TString name(dir); name += "/";  name += h->GetName();
  switch (save_single) {
  case 0: 
    return; break;
  case 1: 
    name+=".eps"; break;
  case 2: 
    name+=".gif"; break;
  case 3: 
    name+=".jpg"; break;
  }
  cvs->Print(name);
  /*
  TString namee(dir);  namee += "/";  namee += h->GetName();  namee += ".eps";
  TString nameg(dir);  nameg += "/";  nameg += h->GetName();  nameg += ".gif";
  if(save_single==0)
    return;
  else if(save_single==1)
    cvs->Print(namee);
  else if(save_single==2)
    cvs->Print(nameg);
  */
}

void sign() {
  TLatex l;
  l.SetTextAlign(23);
  l.SetTextSize(0.03);
  TString ss("L1Comparator");
  ss += "  |  Run:"; ss += run; 
  if(run>20500 && run < 21010) ss += " GRES";
  if(run>29320 && run < 30635) ss += " GREN";
  if(run>37842-1 && run < 38483+1) ss += " GRUMM";
  l.DrawTextNDC(0.25,0.04,ss);
}
