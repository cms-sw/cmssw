#ifndef HCALSIMPLERECONSTRUCTOR_H
#define HCALSIMPLERECONSTRUCTOR_H 1

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParams.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParam.h" 
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSimpleRecAlgo.h"


    /** \class HcalSimpleReconstructor
	
    $Date: 2013/05/08 23:22:39 $
    $Revision: 1.12 $
    \author J. Mans - Minnesota
    */
class HcalTopology;

    class HcalSimpleReconstructor : public edm::EDProducer {
    public:
      explicit HcalSimpleReconstructor(const edm::ParameterSet& ps);
      virtual ~HcalSimpleReconstructor();
      virtual void produce(edm::Event& e, const edm::EventSetup& c) override final;
      virtual void beginRun(edm::Run const&r, edm::EventSetup const & es) override final;
      virtual void endRun(edm::Run const&r, edm::EventSetup const & es) override final;
    private:      
      template<class DIGICOLL, class RECHITCOLL> void process(edm::Event& e, const edm::EventSetup& c);
      void processUpgrade(edm::Event& e, const edm::EventSetup& c);
      HcalSimpleRecAlgo reco_;
      DetId::Detector det_;
      int subdet_;
      HcalOtherSubdetector subdetOther_;
      edm::InputTag inputLabel_;

      bool dropZSmarkedPassed_; // turn on/off dropping of zero suppression marked and passed digis

      // legacy parameters for config-set values compatibility 
      // to be removed after 4_2_0...
      int firstSample_;
      int samplesToAdd_;
      bool tsFromDB_;
      double firstDepthWeight_;
      bool upgradeHBHE_;
      bool upgradeHF_;

      HcalRecoParams* paramTS;  // firstSample & sampleToAdd from DB  
      HcalTopology *theTopology;
    };

#endif
