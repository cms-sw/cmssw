#ifndef HIST_AUXILIARY_H
#define HIST_AUXILIARY_H

#include "TH1F.h"
#include "TH2.h"

template<class T>
void set_properties(T* h){

	h->Sumw2();
        h->SetDirectory(0);
}

auto createhist(const std::string& name, const std::string& title, const int& nbins, const float xlow=0.f, const float& xhigh=0.f){

	TH1F* h = new TH1F(name.c_str(), title.c_str(), nbins, xlow, xhigh);
        set_properties(h);
        return h;
}

auto createhist(const std::string& name, const std::string& title, const int& nxbins, const float xlow, const float& xhigh, const int& nybins, const float ylow, const float& yhigh){

         TH2F* h = new TH2F(name.c_str(), title.c_str(), nxbins, xlow, xhigh, nybins, ylow, yhigh);
         set_properties(h);
         return h;
}

template<typename std::size_t S>
auto createhist(const std::string& name, const std::string& title, const int& nbins, const double (&arry)[S]){

         TH1F* h = new TH1F(name.c_str(), title.c_str(), nbins, arry);
	 set_properties(h);
         return h;
}

template<typename std::size_t S>
auto createhist(const std::string& name, const std::string& title, const double (&arry)[S], const int& nybins,const float ylow, const float& yhigh) {
         TH2F* h = new TH2F(name.c_str(), title.c_str(), sizeof(arry)/sizeof(arry[0])-1, arry, nybins, ylow, yhigh);
         set_properties(h);
         return h;
}
void Divide_w_sameDsets(TH1F* num, TH1F* denom, TH1F* ratio)
{
        for (int _x = 1; _x < num->GetNbinsX()+1; ++_x)
        {
                float _r        = denom->GetBinContent(_x) ? num->GetBinContent(_x)/denom->GetBinContent(_x) : 0.;
                float _n_relErr = TMath::Abs(   num->GetBinError(_x)/  num->GetBinContent(_x) );
                float _d_relErr = TMath::Abs( denom->GetBinError(_x)/denom->GetBinContent(_x) );
                float _r_err    = TMath::Abs(_r) * ( (_n_relErr > _d_relErr) ? _n_relErr : _d_relErr  );
                ratio->SetBinContent(_x, _r);
                ratio->SetBinError(_x, _r_err);
        }
}

template <typename T,
            typename = typename std::enable_if<std::is_arithmetic<T>::value>>
T
constrainValue(T value,
               T lowerBound,              
               T upperBound)
  {
    assert(lowerBound <= upperBound);
    value = std::max(value, lowerBound);    
    value = std::min(value, upperBound);
    return value;
}

int fillWithOverFlow(TH1 * histogram,
                 double x,
                 double evtWeight=1.,
                 double evtWeightErr=0.)
{
  if(!histogram) assert(0);
  const TAxis * const xAxis = histogram->GetXaxis();
  const int bin = constrainValue(xAxis->FindBin(x), 1, xAxis->GetNbins());
  const double binContent = histogram->GetBinContent(bin);
  const double binError   = histogram->GetBinError(bin);
  histogram->SetBinContent(bin, binContent + evtWeight);
  histogram->SetBinError(bin, std::sqrt(pow(binError,2) + 1));
  return ((bin == xAxis->GetNbins()) || (bin == 1)) ? 1 : 0;
}

int fillWithOverFlow(TH2 * histogram,
                 double x,
		 double y,
                 double evtWeight=1.,
                 double evtWeightErr=0.)
{
  if(!histogram) assert(0);
  const TAxis * const xAxis = histogram->GetXaxis();
  const TAxis * const yAxis = histogram->GetYaxis();
  const int binx = constrainValue(xAxis->FindBin(x), 1, xAxis->GetNbins());
  const int biny = constrainValue(yAxis->FindBin(y), 1, yAxis->GetNbins());
  const double binContent = histogram->GetBinContent(binx, biny);
  const double binError   = histogram->GetBinError(binx, biny);
  histogram->SetBinContent(binx, biny, binContent + evtWeight);
  histogram->SetBinError(binx, biny, std::sqrt(pow(binError,2) + 1));
  return 0;//((bin == xAxis->GetNbins()) || (binx == 1)) ? 1 : 0;
}

template<class T>
void PlotStyle(T* h)
{
        //fonts
    int defaultFont       = 43;
    float x_title_size    = 28;
    float y_title_size    = 28;

    float x_title_offset  = 1.5;
    float y_title_offset  = 2.2;

    float label_size      = 28;
    float label_offset    = 0.013;

        h->GetXaxis()->SetLabelFont(defaultFont);
        h->GetXaxis()->SetTitleFont(defaultFont);
        h->GetYaxis()->SetLabelFont(defaultFont);
        h->GetYaxis()->SetTitleFont(defaultFont);
        h->GetZaxis()->SetLabelFont(defaultFont);
        h->GetZaxis()->SetTitleFont(defaultFont);

        // gStyle->SetTitleFontSize(16);
        h->SetTitleOffset  (0);


        h->GetYaxis()->SetTitleOffset  (y_title_offset);
        h->GetYaxis()->CenterTitle();
        h->GetYaxis()->SetTitleSize    (x_title_size);
        h->GetYaxis()->SetLabelOffset  (label_offset);
        h->GetYaxis()->SetLabelSize    (label_size);
        h->GetYaxis()->SetNdivisions(508);

        h->GetXaxis()->SetTitleOffset  (1.2);
        h->GetXaxis()->CenterTitle();
        h->GetXaxis()->SetTitleSize    (x_title_size);
        h->GetXaxis()->SetLabelOffset  (label_offset);
        h->GetXaxis()->SetLabelSize    (label_size);
        h->GetXaxis()->SetNdivisions(508);

        h->GetZaxis()->SetTitleOffset  (1.8);
        h->GetZaxis()->CenterTitle();
        h->GetZaxis()->SetTitleSize    (x_title_size);
        h->GetZaxis()->SetLabelOffset  (label_offset);
        h->GetZaxis()->SetLabelSize    (label_size);
        h->GetZaxis()->SetNdivisions(508);

        h->SetLineWidth(2);
        // h->SetMinimum(-0.001);

}

#endif //HIST_AUXILIAY_H
