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

#include "RecoMET/METAlgorithms/interface/HcalNoiseAlgo.h"
#include "RecoMET/METAlgorithms/interface/HcalNoiseRBXArray.h"
#include "DataFormats/METReco/interface/HcalNoiseSummary.h"


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
    //
    
    virtual void produce(edm::Event&, const edm::EventSetup&) override;
    
    //
    // more internal methods
    // fills an HcalNoiseRBXArray with various data
    // filldigis() depends on fillrechits() being called first
    //
    void fillrechits(edm::Event&, const edm::EventSetup&, HcalNoiseRBXArray&, HcalNoiseSummary&) const;
    void filldigis(edm::Event&, const edm::EventSetup&, HcalNoiseRBXArray&, HcalNoiseSummary&);
    void fillcalotwrs(edm::Event&, const edm::EventSetup&, HcalNoiseRBXArray&, HcalNoiseSummary&) const;
    void filltracks(edm::Event&, const edm::EventSetup&, HcalNoiseSummary&) const;

    // other helper functions
    void fillOtherSummaryVariables(HcalNoiseSummary& summary, const CommonHcalNoiseRBXData& data) const;

    
    //
    // parameters
    //
    
    bool fillDigis_;        // fill digi information into HcalNoiseRBXs
    bool fillRecHits_;      // fill rechit information into HcalNoiseRBXs and HcalNoiseSummary
    bool fillCaloTowers_;   // fill calotower information into HcalNoiseRBXs and HcalNoiseSummary
    bool fillTracks_;       // fill track information into HcalNoiseSummary

    // These provide the requirements for writing an RBX to the event
    int maxProblemRBXs_;   // maximum number of problematic RBXs to be written to the event record

    // parameters for calculating summary variables
    int maxCaloTowerIEta_;      // maximum caloTower ieta
    double maxTrackEta_;        // maximum eta of the track
    double minTrackPt_;         // minimum track Pt
    
    std::string digiCollName_;         // name of the digi collection
    std::string recHitCollName_;       // name of the rechit collection
    std::string caloTowerCollName_;    // name of the caloTower collection
    std::string trackCollName_;        // name of the track collection

    double TotalCalibCharge;    // placeholder to calculate total charge in calibration channels

    double minRecHitE_, minLowHitE_, minHighHitE_; // parameters used to determine noise status
    HcalNoiseAlgo algo_; // algorithms to determine if an RBX is noisy

    bool useCalibDigi_;

    // Variables to store info regarding HBHE calibration digis
    double calibdigiHBHEthreshold_;  // minimum charge calib digi in order to be counted by noise algorithm
    std::vector<int> calibdigiHBHEtimeslices_; // time slices to use when computing calibration charge
    // Variables to store info regarding HF calibration digis
    double calibdigiHFthreshold_;
    std::vector<int> calibdigiHFtimeslices_;


    double TS4TS5EnergyThreshold_;
    std::vector<std::pair<double, double> > TS4TS5UpperCut_;
    std::vector<std::pair<double, double> > TS4TS5LowerCut_;

    uint32_t HcalAcceptSeverityLevel_;
    std::vector<int> HcalRecHitFlagsToBeExcluded_;
    
    float adc2fC[128];
  };
  
} // end of namespace

#endif
