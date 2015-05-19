// -*- C++ -*-
//
// 
/**\class dEdxHitAnalyzer dEdxHitAnalyzer.cc 
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
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/DeDxHitInfo.h"





class DQMStore;
class GenericTriggerEventFlag;

class dEdxHitAnalyzer : public DQMEDAnalyzer {
 public:
  explicit dEdxHitAnalyzer(const edm::ParameterSet&);
  ~dEdxHitAnalyzer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  virtual void beginJob();
  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  virtual void endJob() ;

  double harmonic2(const reco::DeDxHitInfo* dedxHits);
  
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
    MonitorElement* ME_StripHitDeDx;
    MonitorElement* ME_PixelHitDeDx;
    MonitorElement* ME_NHitDeDx;
    MonitorElement* ME_Harm2DeDx;
  
    dEdxMEs()
      :ME_StripHitDeDx(NULL)
      ,ME_PixelHitDeDx(NULL)
      ,ME_NHitDeDx(NULL)
      ,ME_Harm2DeDx(NULL)
    {}
  };
  
  edm::InputTag trackInputTag_;
  edm::EDGetTokenT<reco::TrackCollection> trackToken_;

  std::vector<std::string> dEdxInputList_;
  std::vector<edm::EDGetTokenT<reco::DeDxHitInfoAss> > dEdxTokenList_;

  std::string TrackName ;
  std::vector< std::string  > AlgoNames;
  std::vector< dEdxMEs > dEdxMEsVector;
  std::string histname;  //for naming the histograms according to algorithm used
  
  GenericTriggerEventFlag* genTriggerEventFlag_;
  
};
