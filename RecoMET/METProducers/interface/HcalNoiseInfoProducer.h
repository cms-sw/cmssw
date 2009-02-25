#ifndef _RECOMET_METPRODUCER_HCALNOISEINFOPRODUCER_H_
#define _RECOMET_METPRODUCER_HCALNOISEINFOPRODUCER_H_

//
// HcalNoiseInfoProducer.h
//
//   description: Definition of the producer for the HCAL noise information.
//                Uses various algorithms to process digis, rechits, and calotowers
//                and produce a vector of HcalNoiseRBXs.
//                To minimize the time used to sort the data into HPD/RBX space, we fill
//                an rbxarray of size 72, and then pick which rbxs are interesting at
//                the end.
//
//   author: J.P. Chou, Brown
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/METReco/interface/HcalNoiseRBXArray.h"

namespace reco {

  //
  // class declaration
  //
  
  class HcalNoiseInfoProducer : public edm::EDProducer {
  public:
    explicit HcalNoiseInfoProducer(const edm::ParameterSet&);
    ~HcalNoiseInfoProducer();
    
  private:
    
    //
    // methods inherited from EDProducer
    // produce(...) fills an HcalNoiseRBXArray with information from various places, and then
    // picks which rbxs are interesting, storing them to the EDM.
    // Pedestals are uploaded and stored each run
    //
    
    virtual void beginJob(const edm::EventSetup&) ;
    virtual void endJob() ;
    virtual void produce(edm::Event&, const edm::EventSetup&);
    virtual void beginRun(edm::Run&, const edm::EventSetup&);
    virtual void endRun(edm::Run&, const edm::EventSetup&);
    
    //
    // more internal methods
    // fills an HcalNoiseRBXArray with various data
    // filldigis() depends on fillrechits() being called first
    //
    
    void fillrechits(edm::Event&, const edm::EventSetup&, HcalNoiseRBXArray&) const;
    void filldigis(edm::Event&, const edm::EventSetup&, HcalNoiseRBXArray&) const;
    void fillcalotwrs(edm::Event&, const edm::EventSetup&, HcalNoiseRBXArray&) const;
    
    //
    // parameters
    //      
    
    bool fillDigis_;        // fill digi information into HcalNoiseRBXs
    bool fillRecHits_;      // fill rechit information into HcalNoiseRBXs
    bool fillCaloTowers_;   // fill calotower information into HcalNoiseRBXs
    
    double HPDEnergyThreshold_; // HPD Energy threshold
    double RBXEnergyThreshold_; // RBX Energy threshold
    
    double recHitEnergyThreshold_;     // rechit energy threshold for hit counting purposes
    double recHitTimeEnergyThreshold_; // rechit energy threshold for timing purposes
    
    std::string digiCollName_;      // name of the digi collection
    std::string recHitCollName_;    // name of the rechit collection
    std::string caloTowerCollName_; // name of the caloTower collection
    
    bool requirePedestals_;  // require that pedestals are found, or throw an exception
    double nominalPedestal_; // nominal pedestal used if pedestals aren't found
    
    
    //
    // pedestals stored for each run
    //
    
    typedef std::map<HcalDetId, double> pedestalmap_t;
    pedestalmap_t pedestalmap_;
  };
  
} // end of namespace

#endif
