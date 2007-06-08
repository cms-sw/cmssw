// -*- C++ -*-


// system include files
#include <memory>
#include <string>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"
//#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
//#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "Calibration/Tools/interface/TrackAssociator.h"
#include "Calibration/Tools/interface/TimerStack.h"

#include "TH1F.h"

//
// class declaration
//

class AlCaIsoTracksProducer : public edm::EDProducer {
   public:
      explicit AlCaIsoTracksProducer(const edm::ParameterSet&);
      ~AlCaIsoTracksProducer();

      virtual void produce(edm::Event &, const edm::EventSetup&);
      void endJob(void);

   private:

      HTrackAssociator trackAssociator_;

      edm::InputTag hoLabel_;
      bool allowMissingInputs_;
      std::string m_inputTrackLabel;
      std::string m_ecalLabel;
      std::string m_ebInstance;
      std::string m_eeInstance;
      std::string m_hcalLabel;
      double m_dvCut;
      double m_ddirCut;
      double m_pCut;
      double m_ptCut;
      double m_ecalCut;
      int m_histoFlag;
      TFile* m_Hfile;
      struct{
        TH1F* Ntrk;
        TH1F* vx;
        TH1F* vy;
        TH1F* vz;
        TH1F* vr;
        TH1F* eta;
        TH1F* phi;
        TH1F* p;
        TH1F* pt;
        TH1F* Dvertx;
        TH1F* Dverty;
        TH1F* Dvertz;
        TH1F* Dvert;
        TH1F* Dtheta;
        TH1F* Dphi;
        TH1F* Ddir;
        TH1F* Nisotr;
        TH1F* Dering;
        TH1F* eecal;
        TH1F* ehcal;
      } IsoHists;

};

