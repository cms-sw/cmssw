#ifndef DQMOFFLINE_TRIGGER_TAU3MUMONITOR_H
#define DQMOFFLINE_TRIGGER_TAU3MUMONITOR_H

#include <string>

// Framework
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// TriggerUtils
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

// DataFormats
#include "DataFormats/Candidate/interface/CompositeCandidate.h"

struct MEbinning {
  unsigned int nbins;
  double xmin;
  double xmax;
};

class Tau3MuMonitor : public DQMEDAnalyzer 
{
  public:
    Tau3MuMonitor( const edm::ParameterSet& );
    ~Tau3MuMonitor() override;
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  
  protected:  
    void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;  
    void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  
  private:
    static MEbinning getHistoPSet   (edm::ParameterSet pset);
  
    std::string folderName_;
    
    bool validProduct_ = true; // internally store a flag to remember whether the needed tau3mu collection is present and valid
  
    edm::EDGetTokenT<reco::CompositeCandidateCollection> tauToken_; // tau 3 mu collection
  
    MonitorElement* tau1DPt_    ; // 1D tau pt histogram
    MonitorElement* tau1DEta_   ; // 1D tau eta histogram
    MonitorElement* tau1DPhi_   ; // 1D tau phi histogram
    MonitorElement* tau1DMass_  ; // 1D tau mass histogram
    MonitorElement* tau2DEtaPhi_; // 2D tau eta vs phi histogram

    MEbinning pt_binning_  ; // for the 1D tau pt histogram
    MEbinning eta_binning_ ; // for the 1D tau eta histogram and 2D tau eta vs phi histogram
    MEbinning phi_binning_ ; // for the 1D tau phi histogram and 2D tau eta vs phi histogram
    MEbinning mass_binning_; // for the 1D tau mass histogram
      
    std::unique_ptr<GenericTriggerEventFlag> genTriggerEventFlag_; // "is trigger fired?" flag
};

#endif // DQMOFFLINE_TRIGGER_TAU3MUMONITOR_H
