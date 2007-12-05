#ifndef DataFormats_Scalers_ScalersProducer
#define DataFormats_Scalers_ScalersProducer

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ScalersProducer : public edm::EDProducer
{
public:
   
    explicit ScalersProducer(const edm::ParameterSet  &);
    ~ScalersProducer();
    
    virtual void produce(edm::Event &, const edm::EventSetup &);
    
    private:
    
    bool verbose_;
    unsigned char buffer [1024];
    const L1TriggerScalers *previousTrig;
};

#endif
