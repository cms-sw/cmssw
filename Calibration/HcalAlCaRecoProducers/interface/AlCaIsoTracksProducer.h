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
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

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
      std::string m_inputTrackLabel;
      TFile* m_Hfile;
      struct{
        TH1F* Ntrk;
        TH1F* vx;
        TH1F* vy;
        TH1F* vz;
        TH1F* eta;
        TH1F* phi;
        TH1F* p;
        TH1F* pt;
        TH1F* Dvertx;
        TH1F* Dverty;
        TH1F* Dvertz;
        TH1F* Dvert;
        TH1F* Deta;
        TH1F* Dphi;
        TH1F* Ddir;
        TH1F* Nisotr;
      } IsoHists;

};
