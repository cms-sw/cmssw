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

#include "DQM/Physics_HWW/interface/analysisSelections.h"
#include "DQM/Physics_HWW/interface/EGammaMvaEleEstimator.h"
#include "DQM/Physics_HWW/interface/MuonMVAEstimator.h"
#include "DQM/Physics_HWW/interface/monitor.h"

#include "DQM/Physics_HWW/interface/EventMaker.h"
#include "DQM/Physics_HWW/interface/VertexMaker.h"
#include "DQM/Physics_HWW/interface/TrackMaker.h"
#include "DQM/Physics_HWW/interface/ElectronMaker.h"
#include "DQM/Physics_HWW/interface/MuonMaker.h"
#include "DQM/Physics_HWW/interface/PFJetMaker.h"
#include "DQM/Physics_HWW/interface/HypDilepMaker.h"
#include "DQM/Physics_HWW/interface/PFCandidateMaker.h"
#include "DQM/Physics_HWW/interface/PFElectronMaker.h"
#include "DQM/Physics_HWW/interface/PFElToElAssMaker.h"
#include "DQM/Physics_HWW/interface/GSFTrackMaker.h"
#include "DQM/Physics_HWW/interface/RecoConversionMaker.h"
#include "DQM/Physics_HWW/interface/RhoMaker.h"
#include "DQM/Physics_HWW/interface/PFMETMaker.h"
#include "DQM/Physics_HWW/interface/TrkMETMaker.h"
#include "DQM/Physics_HWW/interface/PileupJetIdAlgo.h"
#include "DQM/Physics_HWW/interface/MVAJetIdMaker.h"

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

      EGammaMvaEleEstimator* egammaMvaEleEstimator;
      MuonMVAEstimator* muonMVAEstimator;
  
      EventMonitor eventMonitor;

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

      MonitorElement* cutflowHist[4];
        
      void FillHistograms();

};

#endif
