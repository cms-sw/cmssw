#include "AnalysisDataFormats/TopObjects/interface/TtSemiMassSolution.h"

TtSemiMassSolution::TtSemiMassSolution()
{
  dmtop_ = -999.;
}

TtSemiMassSolution::TtSemiMassSolution(TtSemiEvtSolution asol): 
  TtSemiEvtSolution(asol) 
{
  dmtop_ = -999.;
}

TtSemiMassSolution::~TtSemiMassSolution()
{
}

void TtSemiMassSolution::setScanValues(std::vector<std::pair<double,double> > v)
{
  for(unsigned int i=0; i<v.size(); i++) scanValues.push_back(v[i]);
}
