#ifndef HWWANALYZER_H
#define HWWANALYZER_H  

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "RecoJets/JetProducers/interface/PileupJetIdAlgo.h"

#include "DQM/PhysicsHWW/interface/analysisSelections.h"
#include "DQM/PhysicsHWW/interface/EGammaMvaEleEstimator.h"
#include "DQM/PhysicsHWW/interface/MuonMVAEstimator.h"
#include "DQM/PhysicsHWW/interface/monitor.h"

#include "DQM/PhysicsHWW/interface/EventMaker.h"
#include "DQM/PhysicsHWW/interface/VertexMaker.h"
#include "DQM/PhysicsHWW/interface/TrackMaker.h"
#include "DQM/PhysicsHWW/interface/ElectronMaker.h"
#include "DQM/PhysicsHWW/interface/MuonMaker.h"
#include "DQM/PhysicsHWW/interface/PFJetMaker.h"
#include "DQM/PhysicsHWW/interface/HypDilepMaker.h"
#include "DQM/PhysicsHWW/interface/PFCandidateMaker.h"
#include "DQM/PhysicsHWW/interface/PFElectronMaker.h"
#include "DQM/PhysicsHWW/interface/PFElToElAssMaker.h"
#include "DQM/PhysicsHWW/interface/GSFTrackMaker.h"
#include "DQM/PhysicsHWW/interface/RecoConversionMaker.h"
#include "DQM/PhysicsHWW/interface/RhoMaker.h"
#include "DQM/PhysicsHWW/interface/PFMETMaker.h"
#include "DQM/PhysicsHWW/interface/TrkMETMaker.h"
#include "DQM/PhysicsHWW/interface/MVAJetIdMaker.h"

#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>


class HWWAnalyzer : public DQMEDAnalyzer {

   public:

      HWWAnalyzer(const edm::ParameterSet&);
      virtual ~HWWAnalyzer();

   protected:

      void analyze(const edm::Event&, const edm::EventSetup&);
      void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

   private:
    
      bool doTest;

      EGammaMvaEleEstimator* egammaMvaEleEstimator;
      MuonMVAEstimator* muonMVAEstimator;
  
      VertexMaker         vertexMaker;
      EventMaker          eventMaker;
      TrackMaker          trackMaker;
      ElectronMaker       electronMaker;
      MuonMaker           muonMaker;
      PFJetMaker          pfJetMaker;
      HypDilepMaker       hypDilepMaker;
      PFCandidateMaker    pfCandidateMaker;
      PFElectronMaker     pfElectronMaker;
      PFElToElAssMaker    pfElToElAssMaker;
      GSFTrackMaker       gsfTrackMaker;
      RecoConversionMaker recoConversionMaker;
      RhoMaker            rhoMaker;
      PFMETMaker          pfMETMaker;
      TrkMETMaker         trkMETMaker;
      MVAJetIdMaker       mvaJetIdMaker;

      std::unique_ptr<EventMonitor> eventMonitor;
};

#endif
