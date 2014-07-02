// -*- C++ -*-
//
// 
/**\class dEdxAnalyzer dEdxAnalyzer.cc 
Monitoring source for general quantities related to track dEdx.
*/
// Original Author: Loic Quertenmont 2012/07/25 

#include <memory>
#include <fstream>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

class DQMStore;
class GenericTriggerEventFlag;

class dEdxAnalyzer : public DQMEDAnalyzer {
 public:
  explicit dEdxAnalyzer(const edm::ParameterSet&);
  ~dEdxAnalyzer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  virtual void beginJob();
  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  virtual void endJob() ;

  double mass(double P, double I);
  
  //  virtual void beginRun(const edm::Run&, const edm::EventSetup&); 
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  
 private:
  // ----------member data ---------------------------
  DQMStore * dqmStore_;
  edm::ParameterSet fullconf_;
  edm::ParameterSet conf_;
  
  bool doAllPlots_;
  bool doDeDxPlots_;
  
  struct dEdxMEs 
  {
    MonitorElement* ME_MipDeDx;
    MonitorElement* ME_MipDeDxNHits;
    MonitorElement* ME_MipDeDxNSatHits;
    MonitorElement* ME_MipDeDxMass;
    MonitorElement* ME_HipDeDxMass;
    
    dEdxMEs()
      :ME_MipDeDx(NULL)
      ,ME_MipDeDxNHits(NULL)
      ,ME_MipDeDxNSatHits(NULL)
      ,ME_MipDeDxMass(NULL)
      ,ME_HipDeDxMass(NULL)
    {}
  };
  
  double TrackHitMin, HIPdEdxMin;
  double dEdxK, dEdxC;
  
  edm::InputTag trackInputTag_;
  edm::EDGetTokenT<reco::TrackCollection> trackToken_;

  std::vector<std::string> dEdxInputList_;
  std::vector<edm::EDGetTokenT<reco::DeDxDataValueMap> > dEdxTokenList_;

  std::string TrackName ;
  std::vector< std::string  > AlgoNames;
  std::vector< dEdxMEs > dEdxMEsVector;
  std::string histname;  //for naming the histograms according to algorithm used
  
  GenericTriggerEventFlag* genTriggerEventFlag_;
  
};
