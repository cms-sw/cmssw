#include "DataFormats/Math/interface/LorentzVector.h"
#include "DQM/PhysicsHWW/interface/HWWAnalyzer.h"
#include "DQM/PhysicsHWW/interface/HWW.h"


HWWAnalyzer::HWWAnalyzer(const edm::ParameterSet& iConfig)
            :vertexMaker        (iConfig, consumesCollector()),
             trackMaker         (iConfig, consumesCollector()),
             electronMaker      (iConfig, consumesCollector()),
             muonMaker          (iConfig, consumesCollector()),
             pfJetMaker         (iConfig, consumesCollector()),
             pfCandidateMaker   (iConfig, consumesCollector()),
             pfElectronMaker    (iConfig, consumesCollector()),
             gsfTrackMaker      (iConfig, consumesCollector()),
             recoConversionMaker(iConfig, consumesCollector()),
             rhoMaker           (iConfig, consumesCollector()),
             pfMETMaker         (iConfig, consumesCollector()),
             mvaJetIdMaker      (iConfig, consumesCollector())
{

  doTest = iConfig.getParameter<bool>("doTest");
  if(doTest) edm::LogInfo("OutputInfo") << "running with doTest==True";

  egammaMvaEleEstimator = 0;
  muonMVAEstimator = 0;


  // --------------- EGamma Id MVA  --------------------------
  std::string egammaweights1 = iConfig.getParameter<edm::FileInPath> ("InputEGammaWeights1").fullPath();
  std::string egammaweights2 = iConfig.getParameter<edm::FileInPath> ("InputEGammaWeights2").fullPath();
  std::string egammaweights3 = iConfig.getParameter<edm::FileInPath> ("InputEGammaWeights3").fullPath();
  std::string egammaweights4 = iConfig.getParameter<edm::FileInPath> ("InputEGammaWeights4").fullPath();
  std::string egammaweights5 = iConfig.getParameter<edm::FileInPath> ("InputEGammaWeights5").fullPath();
  std::string egammaweights6 = iConfig.getParameter<edm::FileInPath> ("InputEGammaWeights6").fullPath();
  std::vector<std::string> egammaweights = {
    egammaweights1,
    egammaweights2,
    egammaweights3,
    egammaweights4,
    egammaweights5,
    egammaweights6
  };
  egammaMvaEleEstimator = new EGammaMvaEleEstimator();
  egammaMvaEleEstimator->initialize("BDT", EGammaMvaEleEstimator::kTrig, true, egammaweights );

  // --------------- Muon RingIso MVA  --------------------------
  std::string muonisoweights1 = iConfig.getParameter<edm::FileInPath> ("InputMuonIsoWeights1").fullPath();
  std::string muonisoweights2 = iConfig.getParameter<edm::FileInPath> ("InputMuonIsoWeights2").fullPath();
  std::string muonisoweights3 = iConfig.getParameter<edm::FileInPath> ("InputMuonIsoWeights3").fullPath();
  std::string muonisoweights4 = iConfig.getParameter<edm::FileInPath> ("InputMuonIsoWeights4").fullPath();
  std::string muonisoweights5 = iConfig.getParameter<edm::FileInPath> ("InputMuonIsoWeights5").fullPath();
  std::string muonisoweights6 = iConfig.getParameter<edm::FileInPath> ("InputMuonIsoWeights6").fullPath();
  std::vector<std::string> muonisoweights = {
    muonisoweights1,
    muonisoweights2,
    muonisoweights3,
    muonisoweights4,
    muonisoweights5,
    muonisoweights6
  };
  muonMVAEstimator = new MuonMVAEstimator();
  muonMVAEstimator->initialize( "MuonIso_BDTG_IsoRings", MuonMVAEstimator::kIsoRings, true, muonisoweights );

}


HWWAnalyzer::~HWWAnalyzer(){ 
}


void HWWAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){

  using namespace HWWFunctions;

  HWW hww;

  //count total events
  eventMonitor->count(MM, "total events", 1.0);
  eventMonitor->count(EE, "total events", 1.0);
  eventMonitor->count(EM, "total events", 1.0);
  eventMonitor->count(ME, "total events", 1.0);

  //if doTest flag is true, all we do is access all the collections 
  //without having to make it through the cutflow.
  if(doTest){

    eventMaker          .SetVars(hww, iEvent, iSetup);
    vertexMaker         .SetVars(hww, iEvent, iSetup);
    trackMaker          .SetVars(hww, iEvent, iSetup);
    electronMaker       .SetVars(hww, iEvent, iSetup);
    muonMaker           .SetVars(hww, iEvent, iSetup);
    pfJetMaker          .SetVars(hww, iEvent, iSetup);
    hypDilepMaker       .SetVars(hww, iEvent, iSetup);
    pfCandidateMaker    .SetVars(hww, iEvent, iSetup);
    pfElectronMaker     .SetVars(hww, iEvent, iSetup);
    pfElToElAssMaker    .SetVars(hww, iEvent, iSetup);
    gsfTrackMaker       .SetVars(hww, iEvent, iSetup);
    recoConversionMaker .SetVars(hww, iEvent, iSetup);
    rhoMaker            .SetVars(hww, iEvent, iSetup);
    pfMETMaker          .SetVars(hww, iEvent, iSetup);
    trkMETMaker         .SetVars(hww, iEvent, iSetup);
    mvaJetIdMaker       .SetVars(hww, iEvent, iSetup);

    return;

  }

  //get variables
  eventMaker    .SetVars(hww, iEvent, iSetup);
  vertexMaker   .SetVars(hww, iEvent, iSetup);
  trackMaker    .SetVars(hww, iEvent, iSetup);

  if(hww.trks_trk_p4().size() < 2) return;

  electronMaker .SetVars(hww, iEvent, iSetup);
  muonMaker     .SetVars(hww, iEvent, iSetup);
  pfJetMaker    .SetVars(hww, iEvent, iSetup);
  hypDilepMaker .SetVars(hww, iEvent, iSetup);

  //check some basic event requirements
  std::vector<int> goodHyps;
  for(unsigned int i=0; i < hww.hyp_p4().size(); i++){
    if(!passFirstCuts(hww, i)) continue;
    goodHyps.push_back(i);
  }
  
  //no need to continue if event didn't pass basic requirements
  if(goodHyps.size() > 0){

    //get variables
    pfCandidateMaker    .SetVars(hww, iEvent, iSetup);
    pfElectronMaker     .SetVars(hww, iEvent, iSetup);
    pfElToElAssMaker    .SetVars(hww, iEvent, iSetup);
    gsfTrackMaker       .SetVars(hww, iEvent, iSetup);
    recoConversionMaker .SetVars(hww, iEvent, iSetup);
    rhoMaker            .SetVars(hww, iEvent, iSetup);

    //to hold indices of candidate lepton pairs
    std::vector<int> candidates;

    //get lepton pairs that pass baseline selection
    for(unsigned int i=0; i < goodHyps.size(); i++){
      if(!passBaseline(hww, goodHyps.at(i), egammaMvaEleEstimator, muonMVAEstimator)) continue;
      candidates.push_back(i);      
    }

    if(candidates.size()>0){

      //get variables
      pfMETMaker            .SetVars(hww, iEvent, iSetup);
      trkMETMaker           .SetVars(hww, iEvent, iSetup);
      mvaJetIdMaker         .SetVars(hww, iEvent, iSetup);

      //find best lepton pair
      int bestHyp = bestHypothesis(hww, candidates);

      //perform remaining selections
      doCutFlow(hww, bestHyp, *eventMonitor, egammaMvaEleEstimator, muonMVAEstimator);

    }
  }  
}//end analyze


void HWWAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,edm::Run const &, edm::EventSetup const &) {

  eventMonitor.reset(new EventMonitor(ibooker));
}

DEFINE_FWK_MODULE(HWWAnalyzer);
