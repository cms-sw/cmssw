// -*- C++ -*-
//
// Package:     TtSemiMassSolution
// Class  :     TtSemiMassSolution
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Wed May 10 11:48:25 CEST 2006
// $Id: TtSemiMassSolution.cc,v 1.4 2007/05/15 16:06:19 heyninck Exp $
//

// system include files

// user include files
#include "AnalysisDataFormats/TopObjects/interface/TtSemiMassSolution.h"



TtSemiMassSolution::TtSemiMassSolution()
{
  dmtop = -999.;
}
TtSemiMassSolution::TtSemiMassSolution(TtSemiEvtSolution asol): TtSemiEvtSolution(asol) {
  dmtop = -999.;
}

TtSemiMassSolution::~TtSemiMassSolution()
{
}


void TtSemiMassSolution::setMtopUncertainty(double dm)		{ dmtop = dm; }
void TtSemiMassSolution::setScanValues(std::vector<std::pair<double,double> > v)    {
  for(unsigned int i=0; i<v.size(); i++) scanValues.push_back(v[i]);
}
