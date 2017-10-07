#include <cmath>
#include "TH1D.h"
#include "TProfile2D.h"

TH1D* projectProfile2DAlongX(TProfile2D* prof2d) {

  TH1D* res=nullptr;

  if(prof2d) {
    char name[200];
    sprintf(name,"%s_proj",prof2d->GetName());
    res = new TH1D(name,prof2d->GetTitle(),prof2d->GetNbinsY(),prof2d->GetYaxis()->GetXmin(),prof2d->GetYaxis()->GetXmax());
    res->SetDirectory(nullptr);
    res->Sumw2();
    for(int iy=1;iy<prof2d->GetNbinsY()+1;++iy) {
      double sum=0.;
      double sumsq=0.;
      double nevt=0.;
      for(int ix=1;ix<prof2d->GetNbinsX()+1;++ix) {
	const int ibin = prof2d->GetBin(ix,iy);
	sum += prof2d->GetBinContent(ibin)*prof2d->GetBinEntries(ibin);
	sumsq += prof2d->GetBinError(ibin)*prof2d->GetBinError(ibin)*prof2d->GetBinEntries(ibin)*prof2d->GetBinEntries(ibin)+
	  prof2d->GetBinContent(ibin)*prof2d->GetBinContent(ibin)*prof2d->GetBinEntries(ibin);
	nevt += prof2d->GetBinEntries(ibin);
      }
      double mean = nevt==0 ? 0: sum/nevt;
      double meansq = nevt==0 ? 0: sumsq/nevt;
      double err = meansq >= mean*mean ? sqrt(meansq-mean*mean) : 0;
      err = nevt==0 ? 0 : err/sqrt(nevt);
      res->SetBinContent(iy,mean);
      res->SetBinError(iy,err);
    }
  }
  
  return res;
}

TH1D* projectProfile2DAlongY(TProfile2D* prof2d) {

  TH1D* res=nullptr;

  if(prof2d) {
    char name[200];
    sprintf(name,"%s_proj",prof2d->GetName());
    res = new TH1D(name,prof2d->GetTitle(),prof2d->GetNbinsX(),prof2d->GetXaxis()->GetXmin(),prof2d->GetXaxis()->GetXmax());
    res->SetDirectory(nullptr);
    res->Sumw2();
    for(int ix=1;ix<prof2d->GetNbinsX()+1;++ix) {
      double sum=0.;
      double sumsq=0.;
      double nevt=0.;
      for(int iy=1;iy<prof2d->GetNbinsY()+1;++iy) {
	const int ibin = prof2d->GetBin(ix,iy);
	sum += prof2d->GetBinContent(ibin)*prof2d->GetBinEntries(ibin);
	sumsq += prof2d->GetBinError(ibin)*prof2d->GetBinError(ibin)*prof2d->GetBinEntries(ibin)*prof2d->GetBinEntries(ibin)+
	  prof2d->GetBinContent(ibin)*prof2d->GetBinContent(ibin)*prof2d->GetBinEntries(ibin);
	nevt += prof2d->GetBinEntries(ibin);
      }
      double mean = nevt==0 ? 0: sum/nevt;
      double meansq = nevt==0 ? 0: sumsq/nevt;
      double err = meansq >= mean*mean ? sqrt(meansq-mean*mean) : 0;
      err = nevt==0 ? 0 : err/sqrt(nevt);
      res->SetBinContent(ix,mean);
      res->SetBinError(ix,err);
    }
  }
  
  return res;
}

