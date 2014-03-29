/**
 * This class takes two histograms and a range. It extracts the specified range from
 * each histogram and scales them so that they have a consistent normalization.
 */

#include "TH1F.h"
#include "TProfile.h"
#include <utility>
#include <iostream>


class ScaleFraction
{
public:
  ScaleFraction() {};
  // pair<TH1*, TProfile*> scale( TH1F * histo1, TProfile * histo2, const double & min, const double & max, const TString & index );
  pair<TH1*, TH1*> scale( TH1F * histo1, TProfile * histo2, const double & min, const double & max, const TString & index );
protected:
  template <class T>
  TH1 * scaleOne( T * histo, const double & min, const double & max, const TString & index );
  // Two overloaded inline methods because SetBinContent does not work with TProfile
  inline void fill(TH1F * histo, const int i, const double & value)
  {
    histo->SetBinContent( i, value );
    // histo->Fill( i, value );
  }
  inline void fill(TProfile * histo, const double & x, const double & y)
  {
    std::cout << "inside fill: x = " << x << ", y = " << y << std::endl;
    histo->Fill( x, y );
  }
};

template <class T>
TH1 * ScaleFraction::scaleOne( T * histo, const double & min, const double & max, const TString & index )
{
  int minBin = histo->FindBin(min);
  int maxBin = histo->FindBin(max);

  std::cout << "For " << histo->GetName() << std::endl;
  std::cout << "minBin = " << minBin << std::endl;
  std::cout << "maxBin = " << maxBin << std::endl;

  // T * newHisto = (T*)histo->Clone();
  // newHisto->Reset();
  // newHisto->SetName(TString(histo->GetName())+"_"+index);
  // newHisto->SetTitle(TString(histo->GetTitle())+"_"+index);

  TH1F * newHisto = new TH1F(TString(histo->GetName())+"_"+index, TString(histo->GetTitle())+"_"+index,
			     histo->GetNbinsX(), histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());

  for( int i=minBin; i<=maxBin; ++i ) {
    if( histo->GetBinContent(i) != 0 ) {
      // std::cout << "first("<<i<<") = " << histo->GetBinContent(i) << std::endl;
      std::cout << "first bin center("<<i<<") = " << histo->GetBinCenter(i) << " value = " << histo->GetBinContent(i) << std::endl;
    }
    // fill(newHisto, i, histo->GetBinContent(i));
    newHisto->SetBinContent( i, histo->GetBinContent(i) );
    newHisto->SetBinError( i, histo->GetBinError(i) );
  }

//   newHisto->Multiply(maskHisto);

  // newHisto->Scale(1/newHisto->Integral("width"));
  // newHisto->Scale(1/newHisto->GetEntries());

  for( int i=minBin; i<=maxBin; ++i ) {
    if( newHisto->GetBinContent(i) != 0 ) {
      std::cout << "first("<<i<<") = " << newHisto->GetBinContent(i) << std::endl;
    }
  }

  return newHisto;
}

// pair<TH1*, TProfile*> ScaleFraction::scale( TH1F * histo1, TProfile * histo2, const double & min, const double & max, const TString & index )
pair<TH1*, TH1*> ScaleFraction::scale( TH1F * histo1, TProfile * histo2, const double & min, const double & max, const TString & index )
{
  return make_pair(scaleOne(histo1, min, max, index), scaleOne(histo2, min, max, index));
  // TH1F * fakeHisto1 = 0;
  // return make_pair(fakeHisto1, scaleOne(histo2, min, max, index));
}
