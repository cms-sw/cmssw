
#include "DQM/SiStripCommissioningSummary/interface/SiStripPedestalsSummary.h"

//------------------------------------------------------------------------------

SiStripPedestalsSummary::SiStripPedestalsSummary(sistrip::View view) :
  SiStripSummary(view)

{;}

//------------------------------------------------------------------------------

SiStripPedestalsSummary::~SiStripPedestalsSummary() {;}

//------------------------------------------------------------------------------

void SiStripPedestalsSummary::format() {

  TH1F* const summ = getSummary();
  TH1F* const hist = getHistogram();

  //@@ implement any extra histogram formatting here...

  summ->SetLineColor(kBlue);
  hist->SetLineColor(kBlue);
}

//------------------------------------------------------------------------------
