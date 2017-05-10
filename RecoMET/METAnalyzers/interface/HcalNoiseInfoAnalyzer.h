#ifndef _RECOMET_METANALYZER_HCALNOISEINFOANALYZER_H_
#define _RECOMET_METANALYZER_HCALNOISEINFOANALYZER_H_

//
// HcalNoiseInfoAnalyzer.h
//
//    description: Skeleton analyzer for the HCAL noise information
//
//    author: J.P. Chou, Brown
//
//

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


// forward declarations
class TH1D;
class TH2D;
class TFile;

namespace reco {

  //
  // class declaration
  //

  class HcalNoiseInfoAnalyzer : public edm::EDAnalyzer {
  public:
    explicit HcalNoiseInfoAnalyzer(const edm::ParameterSet&);
    ~HcalNoiseInfoAnalyzer();
  
  
  private:
    virtual void beginJob(const edm::EventSetup&);
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    virtual void endJob();

    // root file/histograms
    TFile* rootfile_;

    TH1D* hMaxZeros_;         // maximum # of zeros in an RBX channel
    TH1D* hTotalZeros_;       // total # of zeros in an RBX
    TH1D* hE2ts_;             // E(2ts) for the highest energy digi in an HPD
    TH1D* hE10ts_;            // E(10ts) for the highest energy digi in an HPD
    TH1D* hE2tsOverE10ts_;    // E(t2s)/E(10ts) for the highest energy digi in an HPD
    TH1D* hRBXE2ts_;          // Sum RBX E(2ts)
    TH1D* hRBXE10ts_;         // Sum RBX E(10ts)
    TH1D* hRBXE2tsOverE10ts_; // Sum RBXE(t2s)/E(10ts)
    TH1D* hHPDNHits_;         // Number of Hits with E>1.5 GeV in an HPD

    TH1D* hFailures_;         // code designating which cut the event failed (if any)
    TH1D* hBeforeRBXEnergy_;  // Total RecHit Energy in RBX before cuts
    TH1D* hAfterRBXEnergy_;   // Total RecHit Energy in RBX after cuts

    // parameters
    std::string rbxCollName_;      // label for the HcalNoiseRBXCollection
    std::string rootHistFilename_; // name of the histogram file
    int noisetype_;                // fill histograms for a specfic noise type
                                   // 0=ionfeedback, 1=HPD discharge, 2=RBX noise, 3=all

  };

} // end of namespace

#endif
