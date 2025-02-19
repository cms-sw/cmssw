#ifndef OVERLAYER_H
#define OVERLAYER_H

#include "TFile.h"
#include "TH1.h"
#include "TCanvas.h"
#include "TLegend.h"
#include <iostream>
#include <sstream>
#include "round_string.h"

class Overlayer {
public:
 Overlayer(std::string filename, std::string datadir, std::string simdir, Color_t color)
  : file_(new TFile(filename.c_str(),"READ")),
    dataLocation_(datadir),
    simLocation_(simdir),
    color_(color)
{ clear(); }

  Overlayer& clear();
  Overlayer& find(std::string);
  Overlayer& xlabel(std::string s) { xlabel_=s; return *this;}
  Overlayer& ylabel(std::string s) { ylabel_=s; return *this;}
  Overlayer& logy(bool state=true) { logy_=state; return *this;}
  Overlayer& withLegend(bool state=true);
  Overlayer& legendLeft(bool state=true) { left_=state; return *this;}
  void print(std::string);
private:
  std::string getStats(const TH1* const) const;

  TFile* const file_;
  const std::string dataLocation_,simLocation_;
  const Color_t color_;
  std::string histName_,xlabel_,ylabel_,title_;
  TH1* data_;
  TH1* sim_;
  bool logy_,left_;
  TLegend* legend_;
};

Overlayer& Overlayer::
clear() {
  std::cout << "clear" << std::endl;
  data_=sim_=0;
  xlabel_=ylabel_=histName_=title_="";
  logy_=false;
  legend_=0;
  left_=false;
  return *this;
}

Overlayer& Overlayer::
find(std::string name) {
  std::cout << "find " << name << "\t" << std::flush;
  histName_=name; 
  data_ = (TH1*) file_->GetDirectory(dataLocation_.c_str())->FindObjectAny(name.c_str());
  sim_ = (TH1*) file_->GetDirectory(simLocation_.c_str())->FindObjectAny(name.c_str());
  return *this;
}

Overlayer& Overlayer::withLegend(bool state) {
  std::cout << "legend: " << (state?"true\t":"false\t") << std::flush;
  if(!state) {legend_=0; return *this;}
  
  std::string data_stats = getStats(data_);
  std::string sim_stats = getStats(sim_);
  unsigned maxlength = std::max( std::max( dataLocation_.size(), simLocation_.size()), 
				 std::max( data_stats.size(), sim_stats.size()) );

  if(left_) legend_ = new TLegend(0,0.75,maxlength*0.015,1.0);
  else legend_ = new TLegend(1.0-maxlength*0.015,0.75,1.0,1.0); 

  legend_->SetFillColor(kWhite);
  legend_->AddEntry(data_,("#splitline{"+dataLocation_+"}{"+ data_stats + "}").c_str() );
  legend_->AddEntry(sim_,("#splitline{"+simLocation_+"}{"+ sim_stats + "}").c_str() );
  return *this;
}

std::string Overlayer::
getStats(const TH1* const hist) const {
  stringstream ss;
  ss << "N: " << hist->GetEntries() << ", "
     << "#mu: " << round_string()(std::make_pair(hist->GetMean(),hist->GetMeanError())) << ", "
     << "#sigma: " << round_string()(std::make_pair(hist->GetRMS(),hist->GetRMSError()));
  return ss.str().c_str();
}

void Overlayer::
print(std::string suffix) {
  std::cout << "print\t" << std::flush;

  sim_->GetXaxis()->SetLabelSize(0.05);  sim_->GetXaxis()->SetTitleSize(0.05);
  sim_->GetYaxis()->SetLabelSize(0.05);  sim_->GetYaxis()->SetTitleSize(0.05);
  sim_->SetTitle((title_+";"+xlabel_+";"+ylabel_).c_str());
  sim_->Scale(data_->Integral()/sim_->Integral());  
  sim_->SetMinimum( (logy_?0.1:0) );
  sim_->SetMaximum( (logy_?2:1.1) * std::max( sim_->GetBinContent(sim_->GetMaximumBin()) + sim_->GetBinError(sim_->GetMaximumBin()) ,
					      data_->GetBinContent(data_->GetMaximumBin()) + data_->GetBinError(data_->GetMaximumBin()) ) );

  sim_->SetFillColor(color_);
  TH1* sim_errors = (TH1*) sim_->Clone((histName_+"_clone").c_str());
  sim_->SetLineColor(kRed);
  sim_errors->SetFillColor(kRed);
  sim_errors->SetFillStyle(3244);
  data_->SetMarkerStyle(20);
  
  TCanvas c("c","",800,600);
  if(logy_) c.SetLogy();

  sim_->Draw("hist");
  sim_errors->Draw("e2same");
  data_->Draw("same");
  if(legend_) legend_->Draw();

  c.Print((histName_+suffix).c_str());  
}

#endif
