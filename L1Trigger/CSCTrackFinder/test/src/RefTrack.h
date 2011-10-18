
#ifndef jhugon_RefTrack_h
#define jhugon_RefTrack_h
// system include files
#include <vector>
#include <string>

#include <SimDataFormats/Track/interface/SimTrackContainer.h>
#include <SimDataFormats/Track/interface/SimTrack.h>
#include <DataFormats/MuonReco/interface/Muon.h>

#include <TMath.h>
#include <TLorentzVector.h>

#include <L1Trigger/CSCTrackFinder/test/src/Track.h>
#include <L1Trigger/CSCTrackFinder/test/src/TFTrack.h>

namespace csctf_analysis
{
  class RefTrack : public Track
  {
    public:
	RefTrack();
	RefTrack(const SimTrack& track);
	RefTrack(const reco::Muon& muon);
	void print();
	void matchedTo(int i, double newR, int newQ, double newTFPt);
	void ghostMatchedTo(const TFTrack& track,int i, double newR);
	double distanceTo(const TFTrack* tftrack) const;

	double getPt() const {return mom.Pt();};
	double getEta() const {return mom.PseudoRapidity();};
	double getPhi() const {return (mom.Phi() > 0) ? mom.Phi() : mom.Phi() + 2*M_PI;};
	double getP() const {return mom.P();};
	double getPz() const {return mom.Pz();};
	int getType() const {return type;}
	
	void setMatch(TFTrack& trackToMatch);
	TFTrack getMatchedTrack() const {return *matchedTrack;}

    private:
	TLorentzVector mom;
	int type;
	TFTrack* matchedTrack;
  };
}
#endif



