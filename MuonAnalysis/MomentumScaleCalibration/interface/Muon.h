#ifndef MuScleFitMuon_h
#define MuScleFitMuon_h
 
#include "TObject.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "TLorentzVector.h"

#include <iostream>

//Adding a stupid comment 
// Another one
typedef reco::Particle::LorentzVector lorentzVector;

  class MuScleFitMuon : public TObject
  {
  public:
    MuScleFitMuon() :
      fP4(lorentzVector(0.,0.,0.,0.)),
      fCharge(0),//+1 or -1
      fPtError(0.),
      fHitsMuon(0),
      fHitsTk(0)
	{}
      
      MuScleFitMuon(const lorentzVector & initP4, const int initCharge=-1, const double initPtError=0, const unsigned int initHitsTk=0, 
		    const unsigned int initHitsMuon=0) :
	fP4(initP4),
	fCharge(initCharge),
	fPtError(initPtError),
	fHitsMuon(initHitsMuon),
	fHitsTk(initHitsTk)
	  {}
	
	/// Used to copy the content of another MuScleFitMuon
	void copy(const MuScleFitMuon & copyMuon)
	{
	  fP4 = copyMuon.p4();
	  fPtError = copyMuon.ptError();
	  fCharge = copyMuon.charge();
	  fHitsMuon = copyMuon.hitsMuon();
	  fHitsTk = copyMuon.hitsTk();
	}

	// Getters
	lorentzVector p4() const {return fP4;}
	Int_t charge() const {return fCharge;}
	Double_t ptError() const {return fPtError;}
	UInt_t hitsMuon() const {return fHitsMuon;}
	UInt_t hitsTk() const {return fHitsTk;}

	// Dummy functions to create compatibility with calls to lorentzVector
	Float_t x() const { return fP4.x(); }
	Float_t y() const { return fP4.y(); }
	Float_t z() const { return fP4.z(); }
	Float_t t() const { return fP4.t(); }
	Float_t e() const { return fP4.e(); }

	Float_t E() const {return fP4.E(); }
	Float_t energy() const {return fP4.energy(); }
	Float_t Pt() const {return fP4.Pt(); }
	Float_t Eta() const {return fP4.Eta(); }
	Float_t Phi() const {return fP4.Phi(); }

	Float_t pt() const {return fP4.pt(); }
	Float_t eta() const {return fP4.eta(); }
	Float_t phi() const {return fP4.phi(); }


	friend std::ostream& operator<< (std::ostream& stream, const MuScleFitMuon& mu) {
	  stream << "p4 = " << mu.p4() << ", q = " << mu.charge() << ", ptError = " << mu.ptError() << ", hitsTk = " <<mu. hitsTk() << ", hitsMuon = " << mu.hitsMuon();
	  return stream;
	}

	lorentzVector fP4;
	Int_t fCharge;
	Double_t fPtError;  
	UInt_t fHitsMuon;
	UInt_t fHitsTk;
	
	ClassDef(MuScleFitMuon, 2)
	  };
ClassImp(MuScleFitMuon)
  

#endif

