#ifndef AnalysisExamples_SiStripDetectorPerformance_MTCCAmplifyDigis_h
#define AnalysisExamples_SiStripDetectorPerformance_MTCCAmplifyDigis_h

#include <string>

#include "FWCore/Framework/interface/EDProducer.h"

// Save Compile time by forwarding declarations
#include "FWCore/Framework/interface/Frameworkfwd.h"

class MTCCAmplifyDigis: public edm::EDProducer {
  public:
    explicit MTCCAmplifyDigis( const edm::ParameterSet &roPARAMETER_SET);
    ~MTCCAmplifyDigis();

  private:
    virtual void produce( edm::Event &roEVENT, 
                          const edm::EventSetup &roEVENT_SETUP);

    // Original Digis Labels
    std::string oSiStripDigisLabel_;
    std::string oSiStripDigisProdInstName_;

    // New Digis Label
    std::string oNewSiStripDigisLabel_;

    // Sigma of Gauss distribution used for Digi amplification
    struct {
      double dTIB;
      double dTOB;
    } oDigiAmplifySigma_;

    struct {
      struct {
	double dL1;
	double dL2;
      } oTIB;

      struct {
	double dL1;
	double dL2;
      } oTOB;
    } oDigiScaleFactor_;
};

#endif // AnalysisExamples_SiStripDetectorPerformance_MTCCAmplifyDigis_h
