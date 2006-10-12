#include "DQM/SiStripCommissioningSummary/interface/PedestalsSummary.h"

//------------------------------------------------------------------------------

PedestalsSummary::PedestalsSummary(sistrip::View view) :
  CommissioningSummary(view)
{;}

//------------------------------------------------------------------------------

PedestalsSummary::~PedestalsSummary() {;}

//------------------------------------------------------------------------------

void PedestalsSummary::format() {

  TH1F* const summ = getSummary();
  TH1F* const hist = getHistogram();

  //@@ implement any extra histogram formatting here...

  summ->SetLineColor(kBlue);
  hist->SetLineColor(kBlue);
}

//------------------------------------------------------------------------------
