#ifndef ConversionLikelihoodCalculator_h
#define ConversionLikelihoodCalculator_h

#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"

#include "TMVA/Reader.h"

class ConversionLikelihoodCalculator
{
    public:
        ConversionLikelihoodCalculator();
        void setWeightsFile(const char * weightsFile);

        double calculateLikelihood(reco::ConversionRef conversion);
        double calculateLikelihood(reco::Conversion & conversion);

    private:
        TMVA::Reader * reader_;
        float log_e_over_p_;
        float log_abs_cot_theta_;
        float log_abs_delta_phi_;
        float log_chi2_max_pt_;
        float log_chi2_min_pt_;

};

#endif

