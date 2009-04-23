/*
 *  File: DataFormats/Scalers/interface/ScalersProducer.h
 */

#ifndef DataFormats_Scalers_ScalersProducer
#define DataFormats_Scalers_ScalersProducer

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Scalers/interface/L1AcceptBunchCrossing.h"
#include "DataFormats/Scalers/interface/L1TriggerScalers.h"
#include "DataFormats/Scalers/interface/L1TriggerRates.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"

class ScalersProducer : public edm::EDProducer
{
public:
   
    explicit ScalersProducer(const edm::ParameterSet  &);
    ~ScalersProducer();
    
    virtual void produce(edm::Event &, const edm::EventSetup &);
    
  // BeginJob
    virtual void beginJob(const edm::EventSetup & c);

  // EndJob
    virtual void endJob();
    
    private:
    
    bool verbose_;
    unsigned char buffer [sizeof(struct ScalersEventRecordRaw_v1)];
    char * fileName;
    int bytes;
    int fd;
    int ev;
    const L1TriggerScalers *previousTrig;
};

#endif
