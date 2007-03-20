#include "FastSimulation/Tracking/test/FastTrackAnalyzer.h"
#include "Math/GenVector/BitReproducible.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h" 
#include <memory>
#include <iostream>
#include <string>

using namespace edm;
using namespace std;
    

FastTrackAnalyzer::FastTrackAnalyzer(edm::ParameterSet const& conf) : 
  conf_(conf) {}

FastTrackAnalyzer::~FastTrackAnalyzer() {}

void FastTrackAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup)
  {
    edm::ESHandle<TrackerGeometry> theG;
    setup.get<TrackerDigiGeometryRecord>().get(theG);
    
    std::cout << "\nEvent ID = "<< event.id() << std::endl ;
    
    edm::Handle<reco::TrackCollection> trackCollection;
    event.getByType(trackCollection);
    
    //get simtrack info
    edm::Handle<SimTrackContainer> theSimTracks;
    event.getByType<SimTrackContainer>(theSimTracks);

    edm::Handle<SimVertexContainer> theSimVtx;
    event.getByType(theSimVtx);

    std::vector<unsigned int> SimTrackIds;
    const reco::TrackCollection tC = *(trackCollection.product());
    
    std::cout << "Reconstructed "<< tC.size() << " tracks" << std::endl ;
    
    int i=1;
    for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++){

      std::cout << "Track number "<< i << std::endl ;
      std::cout << "\tmomentum: " << track->momentum()<< "\tPT: " << track->pt()<< std::endl;
      std::cout << "\tvertex: " << track->vertex() << "\timpact parameter: " << track->d0()<< std::endl;
      std::cout << "\tcharge: " << track->charge() << "\tnormalizedChi2: " << track->normalizedChi2()<< std::endl;

      std::cout <<"\t\tNumber of RecHits "<<track->recHitsSize() << std::endl;
      SimTrackIds.clear();
      int ri=0;
      for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
	ri++;
	if ((*it)->isValid()){
	  /*
	  std::cout <<"\t\t\tRecHit " << ri << " on det "<<(*it)->geographicalId().rawId()<<std::endl;
	  std::cout <<"\t\t\tRecHit in LP "<<(*it)->localPosition()<<std::endl;
	  std::cout <<"\t\t\tRecHit in GP "
		    <<theG->idToDet((*it)->geographicalId())->surface().toGlobal((*it)->localPosition()) <<std::endl;
	  */
	  if(const SiTrackerGSRecHit2D * rechit = dynamic_cast<const SiTrackerGSRecHit2D *> (it->get()))	  
	    {
	      int currentId = rechit->simtrackId();	
	      std::cout << "\t\t\tRecHit # " << ri << "\t SimTrackId = " << currentId << std::endl;
	      SimTrackIds.push_back(currentId);
	    }
	}else{
	  cout <<"\t\t Invalid Hit On "<<(*it)->geographicalId().rawId()<<endl;
	} 
      }
      
      int nmax = 0;
      int idmax = -1;
      for(size_t j=0; j<SimTrackIds.size(); j++){
	int n =0;
	n = std::count(SimTrackIds.begin(), SimTrackIds.end(), SimTrackIds[j]);
	if(n>nmax){
	  nmax = n;
	  idmax = SimTrackIds[i];
	}
      }
      float totsim = nmax;
      float tothits = track->recHitsSize();//include pixel as well..
      float fraction = totsim/tothits ;
      
      std::cout << "Track id # " << i << "\tmatched to Simtrack id= " << idmax  << "\t momentum = " << track->momentum() << std::endl;
      std::cout << "\tN(matches)= " << totsim <<  "\t# of rechits = " << track->recHitsSize() 
		<< "\tfraction = " << fraction << std::endl;

      //now found the simtrack information
      for(SimTrackContainer::const_iterator iTrack = theSimTracks->begin(); iTrack != theSimTracks->end(); iTrack++)
	{ 
	  if(iTrack->trackId() == idmax) {
	    std::cout << "\t\tSim track mom = " << iTrack->momentum() << " charge = " <<  iTrack->charge() << std::endl;
	  }
	}
      i++;
      
    }
  }

//define this as a plug-in
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(FastTrackAnalyzer);

