#ifndef HIST_MANAGER_BASE_H
#define HIST_MANAGER_BASE_H

////// Saswati Nandan, Inida/INFN,Pisa /////
#include "hist_auxiliary.h"
#include "plot_auxiliary.h"
#include "TCanvas.h"

using namespace std;

const int numBins = 7;
Double_t customBins[numBins + 1] = {0.1, 0.5, 1.0, 2.0, 6.0, 9.0, 30.0, 100.0};
Double_t customBins_lowpt[numBins + 1] = {0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.85, 1.};
Double_t customBins_highpt[numBins + 1] = {1.0, 2.0, 4.0, 6.0, 9.0, 30.0, 50.0, 100.0};

const int numBins_jets = 8;
Double_t customBins_jets[numBins_jets + 1] = {20, 25, 30, 35, 40, 50, 60, 100.0, 200.0};

class histManagerBase
{
public:
 histManagerBase(const string& obj):
 base_name(obj)
 {};

 ~histManagerBase() {
  for(auto & [k, v]: hists)    delete v;
  for(auto & [k, v]: hists_2d) delete v;
 }

 const string get_base_name() const {
  return base_name;
 }

 void fill(const string& histname, const float& val) {
       fillWithOverFlow(hists[histname], val);
 };

 void fill(const string& histname, const float& valx, const float& valy) {
       fillWithOverFlow(hists_2d[histname], valx, valy);
 };

 void write()
 { 
   for(auto & [k, v]: hists)    v->Write();
   for(auto & [k, v]: hists_2d) v->Write();
 };

 void Plot_single(const vector<string>& histnames)
 {

   for (const auto histname: histnames)
   {
     auto hist = hists[histname.c_str()];
     hist->Scale(1/hist->Integral());

     TCanvas* canv = create_canvas(1);
     canv->SetMargin(0.18, 0.20, 0.12, 0.07);
     canv->SetLogy(true);

     hist->Draw("e");
   
     canv->SaveAs(Form("%s_%s.png", base_name.c_str(), histname.c_str()));
     delete canv;
   }

 };

 template<class T>

 void compareDist(T& hists_1, const string& obj_name_1,
           T& hists_2, const string& obj_name_2,
	   const string& obj_name="")
 {
   for (auto & [key_name, hist] : hists_1)
    {

      string type = typeid(hist).name();

      auto other_hist = hists_2[key_name];
      hist->Scale(1/hist->Integral());
      other_hist->Scale(1/other_hist->Integral());

      PlotStyle(hist); hist->SetLineColor(46);
      PlotStyle(other_hist); other_hist->SetLineWidth(0); other_hist->SetFillColorAlpha(31, 0.4); other_hist->SetLineColorAlpha(31, 0.4);

      TCanvas* canv = create_canvas();
      
      bool do_ratio = (type.find("TH1") != string::npos);
	         
      if (do_ratio)
      {
         TPad* topPad = create_Pad(0.25, 1.0, 0.055, 0.12, 0.03, 0.12, true);
         TPad* bottomPad = create_Pad(0.05, 0.35, 0.02, 0.12, 0.31, 0.12, false);
         canv->cd();
         topPad->Draw();
         topPad->cd();
         topPad->SetMargin (0.18, 0.20, 0.12, 0.07);
	 canv->SetLogy(true);

	 TLegend* leg = new TLegend(.62, .6, .87, .8);
         leg->AddEntry(hist, obj_name_1.c_str(), "f");
         leg->AddEntry(other_hist, obj_name_2.c_str(), "f");
         formatLegend(leg);

	 hist->Draw("hist same");
         other_hist->Draw("hist same");
         leg->Draw("same");

	 canv->cd();
         bottomPad->Draw();
         bottomPad->cd();
         bottomPad->SetLogy(0);
         bottomPad->SetMargin (0.18, 0.20, 0.30, 0.07);

         TH1F* h_ratio = (TH1F*) hist->Clone("ratio");
         h_ratio->GetYaxis()->SetTitle("Ratio");
         h_ratio->GetYaxis()->SetLabelSize(20);
         h_ratio->GetXaxis()->SetLabelOffset(0.01);
         h_ratio->GetXaxis()->SetLabelSize(20);
         h_ratio->GetXaxis()->SetTitleOffset(0.6);
         h_ratio->GetXaxis()->SetTitleSize(20);
         h_ratio->Divide(other_hist);
         h_ratio->GetYaxis()->SetRangeUser(0.90*h_ratio->GetMinimum(),1.10*h_ratio->GetMaximum());
         h_ratio->Draw("e");
      }
      else
      {
	canv->SetMargin(0.18, 0.20, 0.12, 0.07);
        canv->SetLogy(true);
	hist->Divide(other_hist);
	hist->Draw("colz");
      }
     
     canv->SaveAs(Form("%s%s.png", obj_name.c_str(), key_name.c_str()));
     delete canv;
    }
};

 void compareDist(histManagerBase& other)
 {

     compareDist(this->hists, this->base_name,
          other.hists, other.base_name);

     compareDist(this->hists_2d, this->base_name,
          other.hists_2d, other.base_name);
 };

protected:
 std::string base_name;
 map<string, TH1F*> hists;
 map<string, TH2F*> hists_2d;

};


#endif //HISTMANAGERBASE
