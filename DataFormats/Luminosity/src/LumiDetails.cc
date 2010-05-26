
// $Id: LumiDetails.cc,v 1.4 2008/10/23 21:38:25 wdd Exp $

#include "DataFormats/Luminosity/interface/LumiDetails.h"

#include <iomanip>

using namespace std;

bool
LumiDetails::isProductEqual(LumiDetails const& next) const {
  return (lumietsum_ == next.lumietsum_ &&
          lumietsumerr_ == next.lumietsumerr_ &&
          lumietsumqual_ == next.lumietsumqual_ &&
          lumiocc_ == next.lumiocc_ &&
          lumioccerr_ == next.lumioccerr_ &&
          lumioccqual_ == next.lumioccqual_);
}

std::ostream& operator<<(std::ostream& s, const LumiDetails& lumiDetails) {

  const std::vector<float>& lumietsum     = lumiDetails.lumiEtSum();
  const std::vector<float>& lumietsumerr  = lumiDetails.lumiEtSumErr();
  const std::vector<int>& lumietsumqual = lumiDetails.lumiEtSumQual();
  const std::vector<float>& lumiocc       = lumiDetails.lumiOcc();
  const std::vector<float>& lumioccerr    = lumiDetails.lumiOccErr();
  const std::vector<int>& lumioccqual   = lumiDetails.lumiOccQual();

  unsigned int maxSize = lumietsum.size();
  if (lumietsumerr.size() > maxSize) maxSize = lumietsumerr.size();
  if (lumietsumqual.size() > maxSize) maxSize = lumietsumqual.size();
  if (lumiocc.size() > maxSize) maxSize = lumiocc.size();
  if (lumioccerr.size() > maxSize) maxSize = lumioccerr.size();
  if (lumioccqual.size() > maxSize) maxSize = lumioccqual.size();

  s << "\nDumping LumiDetails\n";
  s << setw(12) << "etsum";
  s << setw(12) << "etsumerr";
  s << setw(12) << "etsumqual";
  s << setw(12) << "occ";
  s << setw(12) << "occerr";
  s << setw(12) << "occqual";
  s << "\n";

  for (unsigned int i = 0; i < maxSize; ++i) {

    s << setw(12);
    i < lumietsum.size() ? s << lumietsum[i] : s << " ";

    s << setw(12);
    i < lumietsumerr.size() ? s << lumietsumerr[i] : s << " ";

    s << setw(12);
    i < lumietsumqual.size() ? s << lumietsumqual[i] : s << " ";

    s << setw(12);
    i < lumiocc.size() ? s << lumiocc[i] : s << " ";

    s << setw(12);
    i < lumioccerr.size() ? s << lumioccerr[i] : s << " ";

    s << setw(12);
    i < lumioccqual.size() ? s << lumioccqual[i] : s << " ";

    s << "\n";
  }
  return s << "\n";
}
