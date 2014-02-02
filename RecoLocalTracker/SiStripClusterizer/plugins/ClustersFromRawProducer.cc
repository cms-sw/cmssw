
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithmFactory.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

#define VIDEBUG
#ifdef VIDEBUG
#include<iostream>
#define COUT std::cout << "VI "
#else
#define COUT LogDebug("")
#endif


class SiStripClusterizerFromRaw final : public edm::EDProducer  {

public:

  explicit SiStripClusterizerFromRaw(const edm::ParameterSet& conf) :
    cabling_(nullptr),
    clusterizer_(StripClusterizerAlgorithmFactory::create(conf.getParameter<edm::ParameterSet>("Clusterizer"))),
    rawAlgos_(SiStripRawProcessingFactory::create(conf.getParameter<edm::ParameterSet>("Algorithms"))),
    doAPVEmulatorCheck_(conf.existsAs<bool>("DoAPVEmulatorCheck") ? conf.getParameter<bool>("DoAPVEmulatorCheck") : true)
      {
	produces< edmNew::DetSetVector<SiStripCluster> > ();
      }

  void produce(edm::Event& ev, const edm::EventSetup& es) {

    std::auto_ptr< edmNew::DetSetVector<SiStripCluster> > output(new edmNew::DetSetVector<SiStripCluster>());
    output->reserve(10000,4*10000);

    initialize(es);

    run();

    COUT << output->dataSize() << " clusters from " 
	 << output->size()     << " modules" 
	 << std::endl;


    ev.put(output);

  }

private:

  void initialize(const edm::EventSetup& es);

  void run();


 private:

    SiStripDetCabling const * cabling_;

    std::auto_ptr<StripClusterizerAlgorithm> clusterizer_;
    std::auto_ptr<SiStripRawProcessingAlgorithms> rawAlgos_;


    // March 2012: add flag for disabling APVe check in configuration
    bool doAPVEmulatorCheck_; 

};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripClusterizerFromRaw);




void SiStripClusterizerFromRaw::initialize(const edm::EventSetup& es) {


}

void SiStripClusterizerFromRaw::run() {



}
