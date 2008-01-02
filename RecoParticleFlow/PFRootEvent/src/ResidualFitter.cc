#include "RecoParticleFlow/PFRootEvent/interface/ResidualFitter.h"

#include <iostream>
#include "TCanvas.h"
#include "TH2.h"
#include "TF1.h"
#include "TDirectory.h"

#include <string>

using namespace std;

// TCanvas* ResidualFitter::canvasFit_ = 0;

//ClassImp(ResidualFitter)

int ResidualFitter::xCanvas_ = 600;
int ResidualFitter::yCanvas_ = 600;


ResidualFitter::ResidualFitter(const char* name, 
                               const char* title, 
                               int nbinsx, double xlow, double xup, 
                               int nbinsy, double ylow, double yup, 
                               int nbinsz, double zlow, double zup) 
  : TH3D( name, title, 
          nbinsx, xlow, xup, 
          nbinsy, ylow, yup, 
          nbinsz, zlow, zup ), 
    fitFunction_( new TF1("gaus", "gaus") ), 
    canvasFit_(0),
    curBin_(0), 
    autoRangeN_(0),
    minN_(5) {

  cout<<"creating residual fitter with name "<<name<<endl;
  
  string meanname = name; meanname += "_mean";
  mean_ = new TH2D(meanname.c_str(), meanname.c_str(),
                   nbinsx, xlow, xup, 
                   nbinsy, ylow, yup);
  mean_->SetStats(0);
  
  string sigmaname = name; sigmaname += "_sigma";
  sigma_ = new TH2D(sigmaname.c_str(), sigmaname.c_str(),
                    nbinsx, xlow, xup, 
                    nbinsy, ylow, yup);

  sigma_->SetStats(0);

  string chi2name = name; chi2name += "_chi2";
  chi2_ = new TH2D(chi2name.c_str(), chi2name.c_str(),
                   nbinsx, xlow, xup, 
                   nbinsy, ylow, yup);
  chi2_->SetStats(0);

  //   string nseenname = name; nseenname += "_nseen";
  //   nseen_ = new TH2D(nseenname.c_str(), nseenname.c_str(),
  //               nbinsx, xlow, xup, 
  //               nbinsy, ylow, yup);
  //   nseen_->SetStats(0);

  gDirectory->ls();

  CreateCanvas();
}

ResidualFitter::~ResidualFitter() {
  delete fitFunction_;
  if(canvas_) delete canvas_;
  if(curBin_) delete curBin_;

  delete mean_;
  delete sigma_;
  delete chi2_;
  //   delete nseen_;
}

void ResidualFitter::CreateCanvas() {
  string cname = "ResidualFitterCanvas_"; cname += GetName();
  canvas_ = new TCanvas(cname.c_str(), cname.c_str(),xCanvas_, yCanvas_);
  
  canvas_ ->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)", 
                    "ResidualFitter",
                    this, "ExecuteEvent(Int_t,Int_t,Int_t,TObject*)");

  
}


void ResidualFitter::Fill() {

  Reset();
  
  for(unsigned it = 0; it<true_.size(); it++) {
    for(unsigned im = 0; im<meas_.size(); im++) {
      TH3D::Fill(true_[it].x_, 
                 true_[it].y_, 
                 meas_[it].z_ - true_[it].z_ );
    }
  }
}

void ResidualFitter::FitSlicesZ(TF1 *f1) {

  mean_->Reset();
  sigma_->Reset();
  chi2_->Reset();
  //   nseen_->Reset();

  cout<<"ResidualFitter::FitSlicesZ"<<endl;
  if(f1) SetFitFunction(f1);
  
  for(int ix=1; ix<=GetNbinsX(); ix++) {
    for(int iy=1; iy<=GetNbinsY(); iy++) {
      Fit(ix, iy, "Q0");
    }
  }

  //  TH3D::FitSlicesZ(f1);
}
  

void ResidualFitter::ExecuteEvent(Int_t event, Int_t px, Int_t py, TObject *sel) {

  if( event != kButton1Down ) return;

  TH2* histo2d = dynamic_cast<TH2*>(sel);
  if(!histo2d) return;

  float x = canvas_->AbsPixeltoX(px);
  float y = canvas_->AbsPixeltoY(py);
  x = canvas_->PadtoX(x);
  y = canvas_->PadtoY(y);

  ShowFit(histo2d, x, y);
}


void ResidualFitter::ShowFit(TH2* histo2d, double x, double y) {
  
  if(!canvasFit_) {
    string cname = "ResidualFitterCanvasFit_"; cname += GetName();
    canvasFit_ = new TCanvas(cname.c_str(), cname.c_str(),300,300);
  }
  canvasFit_ ->cd();
  
  int binx = histo2d->GetXaxis()->FindBin(x); 
  int biny = histo2d->GetYaxis()->FindBin(y); 

  if(binx == oldBinx_ && biny == oldBiny_ ) return;
  oldBinx_ = binx;
  oldBiny_ = biny;

  Fit(binx, biny);
 
  canvasFit_->Modified();
  canvasFit_->Update();

  canvas_->Modified();
  canvas_->Update();
  canvas_->cd();
}


void ResidualFitter::Fit(int binx, int biny, const char* opt) {
  TH1::AddDirectory(0);
  
  if(curBin_) delete curBin_;
  curBin_ = TH3::ProjectionZ("", binx, binx, biny, biny);  

  if(curBin_->GetEntries() < minN_ ) {
    TH1::AddDirectory(1);
    return;
  }

  string sopt = fitOptions_; sopt += opt;

  if( autoRangeN_ ) {
    double maxpos = curBin_->GetBinCenter( curBin_->GetMaximumBin() ); 
    
    double minrange = maxpos-curBin_->GetRMS()* autoRangeN_;
    double maxrange = maxpos+curBin_->GetRMS()* autoRangeN_;

    fitFunction_->SetRange( minrange, maxrange );
    cout<<"range : "<<minrange<<" "<<maxrange<<endl;
  }

  curBin_->Fit(fitFunction_, sopt.c_str() );


  double chi2overndf=0;
  if(fitFunction_->GetNDF() ) {
    chi2overndf = fitFunction_->GetChisquare()/ fitFunction_->GetNDF();
    mean_->SetBinContent(binx,biny, fitFunction_->GetParameter(1) );
    mean_->SetBinError(binx,biny, fitFunction_->GetParError(1) );
    sigma_->SetBinContent(binx,biny, fitFunction_->GetParameter(2) );
    sigma_->SetBinError(binx,biny, fitFunction_->GetParError(2) );
 
    chi2_->SetBinContent(binx,biny,chi2overndf);
  }
  //   nseen_->SetBinContent(binx, biny, 
  //                    fitFunction_->Integral( fitFunction_->GetXmin(), 
  //                                            fitFunction_->GetXmax())
  //                    /curBin_->GetBinWidth(1) );

  TH1::AddDirectory(1);
}


