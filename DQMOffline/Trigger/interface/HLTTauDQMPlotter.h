// -*- c++ -*-
#ifndef HLTTauDQMPlotter_h
#define HLTTauDQMPlotter_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/Math/interface/LorentzVector.h"

//Include DQM core
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

using LV = math::XYZTLorentzVectorD;
using LVColl = std::vector<LV>;

struct HLTTauDQMOfflineObjects {
  void clear() {
    electrons.clear();
    muons.clear();
    taus.clear();
    met.clear();
  };
  std::vector<LV> electrons;
  std::vector<LV> muons;
  std::vector<LV> taus;
  std::vector<LV> met;
};

//Virtual base class for HLT-Tau-DQM Plotters
class HLTTauDQMPlotter {
public:
    HLTTauDQMPlotter(const edm::ParameterSet& pset, std::string  dqmBaseFolder);
    HLTTauDQMPlotter(const std::string& dqmFolder, const std::string& dqmBaseFolder);
    ~HLTTauDQMPlotter();
    bool isValid() const { return configValid_; }

protected:
    //Helper functions
    std::pair<bool,LV> match( const LV&, const LVColl&, double );    
    const std::string& triggerTag() const { return dqmFullFolder_; }
    
    //DQM folders
    std::string dqmFullFolder_;
    std::string dqmFolder_;
    
    //Validity check
    bool configValid_;
};
#endif
