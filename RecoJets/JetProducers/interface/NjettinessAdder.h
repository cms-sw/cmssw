#ifndef NjettinessAdder_h
#define NjettinessAdder_h

#include <memory>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "fastjet/contrib/Njettiness.hh"


class NjettinessAdder : public edm::stream::EDProducer<> { 
 public:

    enum MeasureDefinition_t {
        NormalizedMeasure=0,       // (beta,R0) 
        UnnormalizedMeasure,       // (beta) 
        GeometricMeasure,          // (beta) 
        NormalizedCutoffMeasure,   // (beta,R0,Rcutoff) 
        UnnormalizedCutoffMeasure, // (beta,Rcutoff) 
        GeometricCutoffMeasure,    // (beta,Rcutoff) 
	N_MEASURE_DEFINITIONS
    };
    enum AxesDefinition_t {
      KT_Axes=0,
      CA_Axes,
      AntiKT_Axes,   // (axAxesR0)
      WTA_KT_Axes,
      WTA_CA_Axes,
      Manual_Axes,
      OnePass_KT_Axes,
      OnePass_CA_Axes,
      OnePass_AntiKT_Axes,   // (axAxesR0)
      OnePass_WTA_KT_Axes,
      OnePass_WTA_CA_Axes,
      OnePass_Manual_Axes,
      MultiPass_Axes,
      N_AXES_DEFINITIONS
    };



    explicit NjettinessAdder(const edm::ParameterSet& iConfig);
    
    virtual ~NjettinessAdder() {}
    
    void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) ;
    float getTau(unsigned num, const edm::Ptr<reco::Jet> & object) const;
    
 private:	
    edm::InputTag                          src_;
    edm::EDGetTokenT<edm::View<reco::Jet>> src_token_;
    std::vector<unsigned>                  Njets_;

    // Measure definition : 
    unsigned                               measureDefinition_;
    double                                 beta_ ;
    double                                 R0_;
    double                                 Rcutoff_;

    // Axes definition : 
    unsigned                               axesDefinition_;
    int                                    nPass_;
    double                                 akAxesR0_;



    std::auto_ptr<fastjet::contrib::Njettiness>   routine_; 

};

#endif
