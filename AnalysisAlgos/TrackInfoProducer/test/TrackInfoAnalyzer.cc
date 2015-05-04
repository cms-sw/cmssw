#include <memory>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
//#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include <iostream>
#include <string>
#include "TH1F.h"
#include "TFile.h"

using namespace edm;

class TrackInfoAnalyzer : public edm::EDAnalyzer {
 public:
  TrackInfoAnalyzer(const edm::ParameterSet& pset);
  void beginJob(){
    //       cout << "beginJob" <<endl;
    //     edm::ESHandle<TrackerGeometry> tkgeom;
    //    c.get<TrackerDigiGeometryRecord>().get( tkgeom );
    //     tracker=&(* tkgeom);
    output = new TFile(filename.c_str(), "RECREATE");
    tib1int = new TH1F("Tib1Int", "Tib Layer 1 Int", 100, -1, 1);
    tib1ext = new TH1F("Tib1Ext", "Tib Layer 1 Ext", 100, -1, 1);
    tib2int = new TH1F("Tib2Int", "Tib Layer 2 Int", 100, -1, 1);
    tib2ext = new TH1F("Tib2Ext", "Tib Layer 2 Ext", 100, -1, 1);
    tib3int = new TH1F("Tib3Int", "Tib Layer 3 Int", 100, -0.5, 0.5);
    tib3ext = new TH1F("Tib3Ext", "Tib Layer 3 Ext", 100, -0.5, 0.5);
    tib4int = new TH1F("Tib4Int", "Tib Layer 4 Int", 100, -0.5, 0.5);
    tib4ext = new TH1F("Tib4Ext", "Tib Layer 4 Ext", 100, -0.5, 0.5);
  }

  ~TrackInfoAnalyzer(){
    delete output;
  }

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){ //analyze

    using namespace reco;

    //Retrieve tracker topology from geometry
    edm::ESHandle<TrackerTopology> tTopoHandle;
    setup.get<TrackerTopologyRcd>().get(tTopoHandle);
    const TrackerTopology* const tTopo = tTopoHandle.product();

    //std::cout << "\nEvent ID = "<< event.id() << std::endl ;
    edm::Handle<reco::TrackInfoCollection> trackCollection;
    event.getByToken(trackCollectionToken, trackCollection);

    reco::TrackInfoCollection tC = *(trackCollection.product());

    edm::LogInfo("TrackInfoAnalyzer") <<"number of infos "<< tC.size();
    for (reco::TrackInfoCollection::iterator track=tC.begin(); track!=tC.end(); track++){

      //const reco::TrackInfo::TrajectoryInfo tinfo=track->trajstate();
      reco::TrackInfo::TrajectoryInfo::const_iterator iter;
      edm::LogInfo("TrackInfoAnalyzer") <<"N hits in the seed: "<<track->seed().nHits();
      edm::LogInfo("TrackInfoAnalyzer") <<"Starting state "<<track->seed().startingState().parameters().position();
      if(track->trajStateMap().size()>0){
      for(iter=track->trajStateMap().begin();iter!=track->trajStateMap().end();iter++){
	edm::LogInfo("TrackInfoAnalyzer") <<"LocalMomentum: "<<(track->stateOnDet(Combined,(*iter).first)->parameters()).momentum();
	edm::LogInfo("TrackInfoAnalyzer") <<"LocalPosition: "<<(track->stateOnDet(Combined,(*iter).first)->parameters()).position();
	edm::LogInfo("TrackInfoAnalyzer") <<"LocalPosition (rechit): "<<((*iter).first)->localPosition();
	DetId id= (*iter).first->geographicalId();
	unsigned int iSubDet = StripSubdetector(id).subdetId();
        if (iSubDet == StripSubdetector::TIB){
          int layer = tTopo->tibLayer(id);
          unsigned int order = tTopo->tibOrder(id);
          if(layer==1){
            if(order==0)tib1int->Fill((track->stateOnDet(Combined,(*iter).first)->parameters()).position().x()-((*iter).first)->localPosition().x());
	      else if(order==1)tib1ext->Fill((track->stateOnDet(Combined,(*iter).first)->parameters()).position().x()-((*iter).first)->localPosition().x());
	  }
	  else if(layer==2){
	    if(order==0)tib2int->Fill((track->stateOnDet(Combined,(*iter).first)->parameters()).position().x()-((*iter).first)->localPosition().x());
	    else if(order==1)tib2ext->Fill((track->stateOnDet(Combined,(*iter).first)->parameters()).position().x()-((*iter).first)->localPosition().x());
	  }
	  else if(layer==3){
	    if(order==0)tib3int->Fill((track->stateOnDet(Combined,(*iter).first)->parameters()).position().x()-((*iter).first)->localPosition().x());
	    else if(order==1)tib3ext->Fill((track->stateOnDet(Combined,(*iter).first)->parameters()).position().x()-((*iter).first)->localPosition().x());
	  }
	  else if(layer==4){
	    if(order==0)tib4int->Fill((track->stateOnDet(Combined,(*iter).first)->parameters()).position().x()-((*iter).first)->localPosition().x());
	    else if(order==1)tib4ext->Fill((track->stateOnDet(Combined,(*iter).first)->parameters()).position().x()-((*iter).first)->localPosition().x());
	  }
	}
      }
    }
  }
}
  void endJob(){
    output->Write();
    output->Close();
  }
 private:
  TFile * output;
  TH1F* tib1int;
  TH1F* tib1ext;
  TH1F* tib2int;
  TH1F* tib2ext;
  TH1F* tib3int;
  TH1F* tib3ext;
  TH1F* tib4int;
  TH1F* tib4ext;
  std::string filename;
  edm::EDGetTokenT<reco::TrackInfoCollection> trackCollectionToken;

};

TrackInfoAnalyzer::TrackInfoAnalyzer(const edm::ParameterSet& pset)
: filename(pset.getParameter<std::string>("OutFileName"))
, trackCollectionToken(consumes<reco::TrackInfoCollection>(pset.getParameter<edm::InputTag>("TrackInfo")))
{}


DEFINE_FWK_MODULE(TrackInfoAnalyzer);

