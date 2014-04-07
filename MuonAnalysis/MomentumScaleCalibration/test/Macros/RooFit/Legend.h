#ifndef Legend_h
#define Legend_h

#include "TString.h"
#include "TPaveText.h"
#include "TF1.h"

#include <iomanip>
#include <sstream>

/**
 * Small function to simplify the creation of text in the TPaveText. <br>
 * It automatically writes the corret number of figures.
 */
TString setText(const char * text, const double & num1, const char * divider = "", const double & num2 = 0)
{

  // Counter gives the precision
  int precision = 1;
  int k=1;
  while( int(num2*k) == 0 ) {
    k*=10;
    ++precision;
  }

  std::stringstream numString;
  TString textString(text);
  numString << std::setprecision(precision) << std::fixed << num1;
  textString += numString.str();
  if( num2 != 0 ) {
    textString += divider;
    numString.str("");
    if( std::string(text).find("ndf") != std::string::npos ) precision = 0;
    numString << std::setprecision(precision) << std::fixed << num2;
    textString += numString.str();
  }
  return textString;
}

/// This function sets up the TPaveText with chi2/ndf and parameters values +- errors
void setTPaveText(const TF1 * fit, TPaveText * paveText) {
  Double_t chi2 = fit->GetChisquare();
  Int_t ndf = fit->GetNDF();
  paveText->AddText(setText("#chi^2 / ndf = ", chi2,  " / ", ndf));

  for( int iPar=0; iPar<fit->GetNpar(); ++iPar ) {
    TString parName(fit->GetParName(iPar));
    Double_t parValue = fit->GetParameter(iPar); // value of Nth parameter
    Double_t parError = fit->GetParError(iPar);  // error on Nth parameter
    paveText->AddText(setText(parName + " = ", parValue, " #pm ", parError));
  }
}

/**
 * This class can be used to create two legends which are then filled with the results of the fit
 * for two functions.
 */
class TwinLegend
{
 public:
  TwinLegend() : drawSecondLegend_(true)
  {
    paveText1_ = new TPaveText(0.20,0.15,0.49,0.28,"NDC");
    paveText1_->SetFillColor(0);
    paveText1_->SetTextColor(1);
    paveText1_->SetTextSize(0.04);
    paveText1_->SetBorderSize(1);

    paveText2_ = new TPaveText(0.59,0.15,0.88,0.28,"NDC");
    paveText2_->SetFillColor(0);
    paveText2_->SetTextColor(2);
    paveText2_->SetTextSize(0.04);
    paveText2_->SetBorderSize(1);
  }

  void setText(TF1 * fit1, TF1 * fit2)
  {
    setTPaveText(fit1, paveText1_);
    if( fit2 != 0 ) {
      setTPaveText(fit2, paveText2_);
    }
    else {
      drawSecondLegend_ = false;
    }
  }

  void Draw(const TString & option)
  {
    paveText1_->Draw(option);
    if( drawSecondLegend_ ) {
      paveText2_->Draw(option);
    }
  }

  TPaveText * paveText1_;
  TPaveText * paveText2_;
  bool drawSecondLegend_;
};

#endif
