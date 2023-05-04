//-*-c++-*-
//-*-Weighting.cpp-*-
//   Written by James Monk and Andrew Pilkington
/////////////////////////////////////////////////////////////////////////////
#include "GeneratorInterface/ExhumeInterface/interface/Weight.h"

/////////////////////////////////////////////////////////////////////////////
void Exhume::Weight::WeightInit(const double &min, const double &max) {
  Max_ = max;
  FuncMap.insert(std::pair<double, double>(min, WeightFunc(min)));

  std::map<double, double> UNLineShape;

  UNLineShape.insert(std::pair<double, double>(0.0, min));
  UNLineShape.insert(std::pair<double, double>(1.0, max));

  unsigned int jj10 = 1;
  double incr, y, x;
  double NewF, OldF, Oldx;

  while (jj10 < NPoints) {
    jj10 = jj10 * 10;
    LineShape = UNLineShape;
    UNLineShape.clear();
    UNLineShape.insert(std::pair<double, double>(0.0, min));
    incr = (LineShape.rbegin()->first) / double(jj10 - 1.0);
    y = 0.0;
    x = GetValue(y);
    NewF = WeightFunc(x);

    for (unsigned int kk = 1; kk < jj10; kk++) {
      y = y + incr;
      Oldx = x;
      x = GetValue(y);
      OldF = NewF;
      NewF = WeightFunc(x);
      FuncMap.insert(std::pair<double, double>(x, NewF));
      UNLineShape.insert(std::pair<double, double>(UNLineShape.rbegin()->first + 0.5 * (NewF + OldF) * (x - Oldx), x));
    }
  }

  LineShape.clear();
  UNLineShape.clear();
  UNLineShape.insert(std::pair<double, double>(0.0, min));

  std::map<double, double>::iterator jj = FuncMap.begin();
  jj++;
  for (std::map<double, double>::iterator ii = FuncMap.begin(); jj != FuncMap.end(); ii++) {
    UNLineShape.insert(std::pair<double, double>(
        UNLineShape.rbegin()->first + 0.5 * (ii->second + jj->second) * (jj->first - ii->first), jj->first));
    jj++;
  }

  TotalIntegral = UNLineShape.rbegin()->first;
  double C = 1.0 / TotalIntegral;

  for (jj = UNLineShape.begin(); jj != UNLineShape.end(); jj++) {
    LineShape.insert(std::pair<double, double>(C * (jj->first), jj->second));
  }

  UNLineShape.clear();

  std::cout << std::endl
            << "  " << LineShape.size() << " points calculated mapping {" << LineShape.begin()->first << ","
            << LineShape.rbegin()->first << "} to {" << (LineShape.begin())->second << ","
            << (LineShape.rbegin())->second << "}" << std::endl
            << std::endl;

  std::cout << "  Event weighting initialised" << std::endl;

  std::cout << std::endl << " ........................................................................." << std::endl;

  return;
}
/////////////////////////////////////////////////////////////////////////////
void Exhume::Weight::AddPoint(const double &xx_, const double &f_) {
  std::map<double, double>::iterator high_, low_;
  high_ = FuncMap.upper_bound(xx_);
  low_ = high_;
  low_--;

  double OldSegment = 0.5 * (high_->second + low_->second) * (high_->first - low_->first);

  double NewSegment = 0.5 * ((low_->second + f_) * (xx_ - low_->first) + (high_->second + f_) * (high_->first - xx_));

  std::cout << "   Adding point to weight function map" << std::endl;
  std::cout << "   Updating TotalIntegral from " << TotalIntegral;

  TotalIntegral = TotalIntegral - OldSegment + NewSegment;

  std::cout << " to " << TotalIntegral << std::endl;

  FuncMap.insert(std::pair<double, double>(xx_, f_));

  return;
}
/////////////////////////////////////////////////////////////////////////////
