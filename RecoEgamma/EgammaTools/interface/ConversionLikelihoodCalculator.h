#ifndef ConversionLikelihoodCalculator_h
#define ConversionLikelihoodCalculator_h

#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"

//#include "TMVA/Reader.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"
#include "CondFormats/DataRecord/interface/PhotonConversionMVAComputerRcd.h"

class ConversionLikelihoodCalculator
{
   public:
      ConversionLikelihoodCalculator();
//      void setWeightsFile(const char * weightsFile);

      double calculateLikelihood(reco::ConversionRef conversion, const edm::EventSetup& iSetup);
      double calculateLikelihood(reco::Conversion& conversion,   const edm::EventSetup& iSetup);

   private:
//      TMVA::Reader * reader_;
      float log_e_over_p_;
      float log_abs_cot_theta_;
      float log_abs_delta_phi_;
      float log_chi2_max_pt_;
      float log_chi2_min_pt_;
    PhysicsTools::MVAComputerCache mva_;

};

#endif
