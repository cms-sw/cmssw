#ifndef RecoJets_JetProducers_CSJetProducer_h
#define RecoJets_JetProducers_CSJetProducer_h

/* *********************************************************
  \class CSJetProducer

  \brief Jet producer to produce CMS-style constituent subtracted jets

  \author   Marta Verweij
  \version  

         Notes on implementation:

         Constituent subtraction using fastjet contrib package
         The background densities change within the jet as function of eta.

 ************************************************************/


#include "RecoJets/JetProducers/plugins/VirtualJetProducer.h"

namespace cms
{
  class CSJetProducer : public VirtualJetProducer
  {
  public:

    CSJetProducer(const edm::ParameterSet& ps);

    virtual ~CSJetProducer() {}
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    static void fillDescriptionsFromCSJetProducer(edm::ParameterSetDescription& desc);

    virtual void produce( edm::Event & iEvent, const edm::EventSetup & iSetup ) override;
    
  protected:

    virtual void runAlgorithm( edm::Event& iEvent, const edm::EventSetup& iSetup );

    double csRParam_;           /// for constituent subtraction : R parameter
    double csAlpha_;            /// for HI constituent subtraction : alpha (power of pt in metric)

    //input rho and rho_m + eta map
    edm::EDGetTokenT<std::vector<double>>                       etaToken_;
    edm::EDGetTokenT<std::vector<double>>                       rhoToken_;
    edm::EDGetTokenT<std::vector<double>>                       rhomToken_;
  };
}
#endif
