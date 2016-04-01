#ifndef RecoJets_JetProducers_CSJetProducer_h
#define RecoJets_JetProducers_CSJetProducer_h

/* *********************************************************
  \class CSJetProducer

  \brief Jet producer to produce CMS-style constituent subtracted jets

  \author   Marta Verweij
  \version  

         Notes on implementation:

         Reimplementation of constituent subtraction from fastjet contrib package
         to allow the use of eta-dependent rho and rho_m for the constituents 
         inside a jet

 ************************************************************/


#include "RecoJets/JetProducers/plugins/VirtualJetProducer.h"

namespace cms
{
  class CSJetProducer : public VirtualJetProducer
  {
  public:

    CSJetProducer(const edm::ParameterSet& ps);

    virtual ~CSJetProducer() {}

    virtual void produce( edm::Event & iEvent, const edm::EventSetup & iSetup );
    
  protected:

    virtual void runAlgorithm( edm::Event& iEvent, const edm::EventSetup& iSetup );

    static bool function_used_for_sorting(std::pair<double,int> i,std::pair<double, int> j);
    
     // calls VirtualJetProducer::inputTowers
    //virtual void inputTowers();

    double csRho_EtaMax_;       /// for constituent subtraction : maximum rapidity for ghosts
    double csRParam_;           /// for constituent subtraction : R parameter for KT alg in jet median background estimator
    double csAlpha_;            /// for HI constituent subtraction : alpha (power of pt in metric)

    //input rho and rho_m + eta map
    edm::EDGetTokenT<std::vector<double>>                       etaToken_;
    edm::EDGetTokenT<std::vector<double>>                       rhoToken_;
    edm::EDGetTokenT<std::vector<double>>                       rhomToken_;
  };
}
#endif
