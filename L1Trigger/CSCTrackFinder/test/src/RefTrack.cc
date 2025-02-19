#include "L1Trigger/CSCTrackFinder/test/src/RefTrack.h"
#include <iostream>
namespace csctf_analysis
{
  RefTrack::RefTrack():Track() {}
  RefTrack::RefTrack(const SimTrack& track):Track()
  {
	mom.SetPxPyPzE(track.momentum().x(), 
		track.momentum().y(), 
		track.momentum().z(), 
		track.momentum().t());
	Quality=-1;
	type = track.type();
  }
  RefTrack::RefTrack(const reco::Muon& muon):Track()
  {
	mom.SetPxPyPzE(muon.p4().x(),
		muon.p4().y(),
		muon.p4().z(),
		muon.p4().t());
	Quality=-1;
	type = 13;
  }
  double RefTrack::distanceTo(const TFTrack* tftrack) const
  {
	double newR;
	double dEta =getEta()-tftrack->getEta();
	double dPhi =getPhi()-tftrack->getPhi();

	newR = sqrt( dEta*dEta + dPhi*dPhi ); //Changed to do distance style metric by Daniel 07/02

	return newR;
  }
  void RefTrack::matchedTo(int i, double newR,int newQ, double newTFPt)
  {
	if(matched==false || newR<R){
		matched=true;
		R=newR;
		matchedIndex=i;}
	Quality=newQ;
	TFPt=newTFPt;
  }
  void RefTrack::ghostMatchedTo(const TFTrack& track,int i, double newR)
  {
	ghostMatchedToIndex->push_back(i);
	ghostR->push_back(newR);
	ghostQ->push_back(track.getQuality());	
	if (ghostMatchedToIndex->size() > 1)
		ghost=true;
  }
  void RefTrack::print()
  {
    std::cout << "RefTrack Info" << std::endl;
    std::cout << "  Pt: "<< getPt() << std::endl;
    std::cout << "  Phi: "<< getPhi() << std::endl;
    std::cout << "  Eta: "<< getEta() << std::endl;
    std::cout << "  P: "<< getP() << std::endl;
    std::cout << "  Pz: "<< getPz() << std::endl;
    std::cout << "  Type: "<< getType() << std::endl;
  }
  
		
  void RefTrack::setMatch(TFTrack& trackToMatch)
  {
  	matchedTrack = &trackToMatch;
  }

}
