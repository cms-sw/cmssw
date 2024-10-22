#ifndef RecoJets_JetProducers_CSJetProducer_h
#define RecoJets_JetProducers_CSJetProducer_h

/* *********************************************************
  \class CSJetProducer

  \brief Jet producer to produce CMS-style constituent subtracted jets

  \author   Marta Verweij
  \modified for granular eta map Chris McGinn
  \version

         Notes on implementation:

         Constituent subtraction using fastjet contrib package
         The background densities change within the jet as function of eta.

 ************************************************************/

#include <vector>
#include "RecoJets/JetProducers/plugins/VirtualJetProducer.h"

namespace cms {
  class CSJetProducer : public VirtualJetProducer {
  public:
    CSJetProducer(const edm::ParameterSet& ps);

    ~CSJetProducer() override {}
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    static void fillDescriptionsFromCSJetProducer(edm::ParameterSetDescription& desc);

    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  protected:
    void runAlgorithm(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    double getModulatedRhoFactor(const double phi, const edm::Handle<std::vector<double>>& flowParameters);

    double csRParam_;  /// for constituent subtraction : R parameter
    double csAlpha_;   /// for HI constituent subtraction : alpha (power of pt in metric)

    bool useModulatedRho_;    /// flag to turn on/off flow-modulated rho and rhom
    double minFlowChi2Prob_;  /// flowFit chi2/ndof minimum compatability requirement
    double maxFlowChi2Prob_;  /// flowFit chi2/ndof minimum compatability requirement
    //input rho and rho_m + eta map
    edm::EDGetTokenT<std::vector<double>> etaToken_;
    edm::EDGetTokenT<std::vector<double>> rhoToken_;
    edm::EDGetTokenT<std::vector<double>> rhomToken_;
    edm::EDGetTokenT<std::vector<double>> rhoFlowFitParamsToken_;
  };
}  // namespace cms
#endif
