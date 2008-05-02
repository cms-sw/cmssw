#include "RecoEgamma/EgammaTools/interface/ConversionLikelihoodCalculator.h"


ConversionLikelihoodCalculator::ConversionLikelihoodCalculator()
{
   reader_ = new TMVA::Reader();
   
   reader_->AddVariable("log(e_over_p)",	&log_e_over_p_);
   reader_->AddVariable("log(abs(cot_theta))",	&log_abs_cot_theta_);
   reader_->AddVariable("log(abs(delta_phi))",	&log_abs_delta_phi_);
   reader_->AddVariable("log(chi2_max_pt)",	&log_chi2_max_pt_);
   reader_->AddVariable("log(chi2_min_pt)",	&log_chi2_min_pt_);

}

void ConversionLikelihoodCalculator::setWeightsFile(const char * weightsFile)
{
   reader_->BookMVA("Likelihood", weightsFile);
}

double ConversionLikelihoodCalculator::calculateLikelihood(reco::ConvertedPhoton * conversion)
{
   if (conversion->nTracks() != 2) return -1.;
   
   log_e_over_p_ = log(conversion->EoverP());

   log_abs_cot_theta_ = log(fabs(conversion->pairCotThetaSeparation()));

   double delta_phi = conversion->tracks()[0]->innerMomentum().phi()-conversion->tracks()[1]->innerMomentum().phi();
   double pi = 3.14159265;
   // phi normalization
   while (delta_phi > pi) delta_phi -= 2*pi;
   while (delta_phi < -pi) delta_phi += 2*pi;
   log_abs_delta_phi_ = log(fabs(delta_phi));

   double chi2_1 = conversion->tracks()[0]->normalizedChi2();
   double pt_1 = conversion->tracks()[0]->pt();

   double chi2_2 = conversion->tracks()[1]->normalizedChi2();
   double pt_2 = conversion->tracks()[1]->pt();

   double chi2_max_pt=chi2_1;
   double chi2_min_pt=chi2_2;

   if (pt_2 > pt_1) {
      chi2_max_pt=chi2_2;
      chi2_min_pt=chi2_1;
   }

   log_chi2_max_pt_ = log(chi2_max_pt);
   log_chi2_min_pt_ = log(chi2_min_pt);

   return reader_->EvaluateMVA("Likelihood");
}
