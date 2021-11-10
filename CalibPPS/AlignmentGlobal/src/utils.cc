/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*  Mateusz Kocot (mateuszkocot99@gmail.com)
****************************************************************************/

#include "CalibPPS/AlignmentGlobal/interface/utils.h"

#include "TF1.h"

#include <memory>

// Fits a linear function to a TProfile.
int alig_utils::fitProfile(TProfile* p,
                           double x_mean,
                           double x_rms,
                           unsigned int minBinEntries,
                           unsigned int minNBinsReasonable,
                           double& sl,
                           double& sl_unc) {
  unsigned int n_reasonable = 0;
  for (int bi = 1; bi <= p->GetNbinsX(); bi++) {
    if (p->GetBinEntries(bi) < minBinEntries) {
      p->SetBinContent(bi, 0.);
      p->SetBinError(bi, 0.);
    } else {
      n_reasonable++;
    }
  }

  if (n_reasonable < minNBinsReasonable)
    return 1;

  double x_min = x_mean - x_rms, x_max = x_mean + x_rms;

  auto ff_pol1 = std::make_unique<TF1>("ff_pol1", "[0] + [1]*x");

  ff_pol1->SetParameter(0., 0.);
  p->Fit(ff_pol1.get(), "Q", "", x_min, x_max);

  sl = ff_pol1->GetParameter(1);
  sl_unc = ff_pol1->GetParError(1);

  return 0;
}
