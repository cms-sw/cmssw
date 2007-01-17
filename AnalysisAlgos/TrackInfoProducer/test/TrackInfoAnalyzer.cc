#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
//#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// #include "TrackingTools/TransientRecHit/interface/TransientRecHit.h"
// #include "TrackingTools/TransientRecHit/interface/TransientRecHitBuilder.h"
// #include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPos.h"
// #include "TrackingTools/TransientTrackerRecHit2D/interface/TSiStripRecHit2DLocalPos.h"
// #include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include <iostream>
#include <string>
#include "TH1F.h"
#include "TFile.h"

using namespace edm;

class TrackInfoAnalyzer : public edm::EDAnalyzer {
 public:
  TrackInfoAnalyzer(const edm::ParameterSet& pset) {conf_=pset;}
  void beginJob(const edm::EventSetup& c){
    //       cout << "beginJob" <<endl;
    //     edm::ESHandle<TrackerGeometry> tkgeom;
    //    c.get<TrackerDigiGeometryRecord>().get( tkgeom );
    //     tracker=&(* tkgeom);
    std::string filename = conf_.getParameter<std::string>("OutFileName");
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
  edm::ParameterSet conf_;

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){ //analyze

    using namespace std;

    //std::cout << "\nEvent ID = "<< event.id() << std::endl ;
    edm::InputTag TkTag = conf_.getParameter<edm::InputTag>("TrackInfo");
    edm::Handle<reco::TrackInfoCollection> trackCollection;
    event.getByLabel(TkTag, trackCollection);
      
    reco::TrackInfoCollection tC = *(trackCollection.product());

    edm::LogInfo("TrackInfoAnalyzer") <<"number of infos "<< tC.size();
    for (reco::TrackInfoCollection::iterator track=tC.begin(); track!=tC.end(); track++){
      
      //const reco::TrackInfo::TrajectoryInfo tinfo=track->trajstate();
      reco::TrackInfo::TrajectoryInfo::const_iterator iter;
      edm::LogInfo("TrackInfoAnalyzer") <<"N hits in the seed: "<<track->seed().nHits();
      edm::LogInfo("TrackInfoAnalyzer") <<"Starting state "<<track->seed().startingState().parameters().position();
      if(track->trajStateMap().size()>0){
      for(iter=track->trajStateMap().begin();iter!=track->trajStateMap().end();iter++){
	edm::LogInfo("TrackInfoAnalyzer") <<"LocalMomentum: "<<((*iter).second.parameters()).momentum();
	edm::LogInfo("TrackInfoAnalyzer") <<"LocalPosition: "<<((*iter).second.parameters()).position();
	edm::LogInfo("TrackInfoAnalyzer") <<"LocalPosition (rechit): "<<((*iter).first)->localPosition();
	DetId id=((BaseSiTrackerRecHit2DLocalPos *) (&(*(*iter).first)))->geographicalId();
	unsigned int iSubDet = StripSubdetector(id).subdetId();
        if (iSubDet == StripSubdetector::TIB){
	  TIBDetId* tibId = new TIBDetId(id);
	  int layer = tibId->layer();
	  std::vector<unsigned int> string = tibId->string();
	  if(layer==1){
	    if(string[1]==0)tib1int->Fill(((*iter).second.parameters()).position().x()-((*iter).first)->localPosition().x());
	    else if(string[1]==1)tib1ext->Fill(((*iter).second.parameters()).position().x()-((*iter).first)->localPosition().x());
	  }
	  else if(layer==2){
	    if(string[1]==0)tib2int->Fill(((*iter).second.parameters()).position().x()-((*iter).first)->localPosition().x());
	    else if(string[1]==1)tib2ext->Fill(((*iter).second.parameters()).position().x()-((*iter).first)->localPosition().x());
	  }
	  else if(layer==3){
	    if(string[1]==0)tib3int->Fill(((*iter).second.parameters()).position().x()-((*iter).first)->localPosition().x());
	    else if(string[1]==1)tib3ext->Fill(((*iter).second.parameters()).position().x()-((*iter).first)->localPosition().x());
	  }
	  else if(layer==4){
	    if(string[1]==0)tib4int->Fill(((*iter).second.parameters()).position().x()-((*iter).first)->localPosition().x());
	    else if(string[1]==1)tib4ext->Fill(((*iter).second.parameters()).position().x()-((*iter).first)->localPosition().x());
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
  
};

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TrackInfoAnalyzer);

