#ifndef RecoJets_JetProducers_ECFAdder_h
#define RecoJets_JetProducers_ECFAdder_h

#include <memory>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "fastjet/contrib/EnergyCorrelator.hh"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"


class ECFAdder : public edm::stream::EDProducer<> { 
 public:
    explicit ECFAdder(const edm::ParameterSet& iConfig);
    
    void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) override;
    float getECF(unsigned index, const edm::Ptr<reco::Jet> & object) const;

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    
 private:	
    edm::InputTag                          src_;
    edm::EDGetTokenT<edm::View<reco::Jet>> src_token_;
    std::vector<unsigned>                  Njets_;
    std::vector<std::string>               cuts_;
    std::string                            ecftype_;     // Options: ECF (or empty); C; D; N; M; U;
    std::vector<std::string>               variables_;
    double                                 alpha_; 
    double                                 beta_ ;

    std::vector< std::shared_ptr<fastjet::FunctionOfPseudoJet<double> > > routine_;    
    std::vector< StringCutObjectSelector<reco::Jet> >   selectors_;
};

#endif
