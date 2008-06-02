// File: MuonMETAlgo.cc
// Description:  see MuonMETAlgo.h
// Author: M. Schmitt, R. Cavanaugh, The University of Florida
// Creation Date:  MHS May 31, 2005 Initial version.
//
//--------------------------------------------
#include <math.h>
#include <vector>
#include "JetMETCorrections/Type1MET/interface/MuonMETAlgo.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/SpecificCaloMETData.h"
#include "DataFormats/METReco/interface/CorrMETData.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TrackReco/interface/Track.h"


using namespace std;
using namespace reco;

namespace {
  CaloMET makeMet (const CaloMET& fMet, 
		   double fSumEt, 
		   const vector<CorrMETData>& fCorrections, 
		   const MET::LorentzVector& fP4) {
    return CaloMET (fMet.getSpecific (), fSumEt, fCorrections, fP4, fMet.vertex ());
  }
  
  MET makeMet (const MET& fMet, 
	       double fSumEt, 
	       const vector<CorrMETData>& fCorrections, 
	       const MET::LorentzVector& fP4) {
    return MET (fSumEt, fCorrections, fP4, fMet.vertex ());
  }
  
  template <class T>
  void MuonMETAlgo_run(edm::Event& iEvent, 
		       const edm::EventSetup& iSetup,
		       const vector<T>& uncorMET, 
		       const edm::View<reco::Muon>& Muons,
		       double muonPtMin,
		       double muonEtaRange,
		       double muonTrackD0Max,
		       double muonTrackDzMax,
		       int    muonNHitsMin,
		       double muonDPtMax,
		       double muonChiSqMax,
		       bool   muonDepositCor,
		       TrackDetectorAssociator& trackAssociator,
		       TrackAssociatorParameters& trackAssociatorParameters,
		       vector<T>* corMET) 
  {
  
    if (!corMET) {
      std::cerr << "MuonMETAlgo_run-> undefined output MET collection. Stop. " << std::endl;
      return;
    }
    //Jet j = uncorJet->front(); std::cout << j.px() << std::endl;
    double DeltaPx = 0.0;
    double DeltaPy = 0.0;
    double DeltaSumET = 0.0;
    
    double DeltaExDep = 0.0;
    double DeltaEyDep = 0.0;
    double DeltaSumETDep = 0.0;
    // ---------------- Calculate jet corrections, but only for those uncorrected jets
    // ---------------- which are above the given threshold.  This requires that the
    // ---------------- uncorrected jets be matched with the corrected jets.
    for( edm::View<reco::Muon>::const_iterator muon = Muons.begin(); muon != Muons.end(); ++muon) {
      const reco::Track * mu_track = muon->bestTrack();
      if( mu_track->pt() > muonPtMin &&
	  fabs(mu_track->eta()) < muonEtaRange &&
	  fabs(mu_track->d0()) < muonTrackD0Max &&
	  fabs(mu_track->dz()) < muonTrackDzMax &&
	  mu_track->numberOfValidHits() > muonNHitsMin ) {
	float dpt_track = mu_track->error(0)/(mu_track->qoverp());
	float chisq = mu_track->normalizedChi2();
	if (dpt_track < muonDPtMax && 
	    chisq < muonChiSqMax) {
	  DeltaPx +=  muon->px();
	  DeltaPy +=  muon->py();
	  DeltaSumET += muon->et();
	  
	  //----------- Calculate muon energy deposition in the calorimeters
	  if (muonDepositCor) {
	    //std::cout << "   MuonMETAlgo: muon pt=" << mu_track->pt() << std::endl;
	    TrackDetMatchInfo info =
	      trackAssociator.associate(iEvent, iSetup,
					trackAssociator.getFreeTrajectoryState(iSetup, *mu_track),
					trackAssociatorParameters);
	    double ene = info.crossedEnergy(TrackDetMatchInfo::TowerTotal);
	    DeltaExDep += ene*sin((*mu_track).theta())*cos((*mu_track).phi());
	    DeltaEyDep += ene*sin((*mu_track).theta())*sin((*mu_track).phi());
	    DeltaSumETDep += ene*sin((*mu_track).theta());		  
	    //std::cout << "   MuonMETAlgo: deposit ene=" << ene 
	    //      << ", ex=" << ene*sin((*mu_track).theta())*cos((*mu_track).phi())
	    //      << ", ey=" << ene*sin((*mu_track).theta())*sin((*mu_track).phi())
	    //      << ", sumet=" << ene*sin((*mu_track).theta()) << std::endl;
	  }
	}
      }
    }
    //----------------- Calculate and set deltas for new MET correction
    CorrMETData delta;
    delta.mex    = - DeltaPx;    //correction to MET (from muons) is negative,    
    delta.mey    = - DeltaPy;    //since MET points in direction of muons
    delta.sumet  = DeltaSumET; 
    delta.mex   +=   DeltaExDep;    //correction to MET (from muon depositions) is positive,    
    delta.mey   +=   DeltaEyDep;    //since MET points in opposite direction of muons
    delta.sumet -= DeltaSumETDep; 
    //----------------- Fill holder with corrected MET (= uncorrected + delta) values
    const T* u = &(uncorMET.front());
    double corrMetPx = u->px()+delta.mex;
    double corrMetPy = u->py()+delta.mey;
    MET::LorentzVector correctedMET4vector( corrMetPx, corrMetPy, 0., 
					    sqrt (corrMetPx*corrMetPx + corrMetPy*corrMetPy) );
    //----------------- get previous corrections and push into new corrections 
    std::vector<CorrMETData> corrections = u->mEtCorr();
    corrections.push_back( delta );
    //----------------- Push onto MET Collection
    T result = makeMet (*u, u->sumEt()+delta.sumet, corrections,correctedMET4vector); 
    corMET->push_back(result);
  }
}


//----------------------------------------------------------------------------
MuonMETAlgo::MuonMETAlgo() {}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
MuonMETAlgo::~MuonMETAlgo() {}
//----------------------------------------------------------------------------

void MuonMETAlgo::run(edm::Event& iEvent, 
		      const edm::EventSetup& iSetup,
		      const CaloMETCollection& uncorMET, 
		      const edm::View<reco::Muon>& Muons,
		      double muonPtMin,
		      double muonEtaRange,
		      double muonTrackD0Max,
		      double muonTrackDzMax,
		      int    muonNHitsMin,
		      double muonDPtMax,
		      double muonChiSqMax,
		      bool   muonDepositCor,
		      TrackDetectorAssociator& trackAssociator,
		      TrackAssociatorParameters& trackAssociatorParameters,
		      CaloMETCollection* corMET) 
{
  return MuonMETAlgo_run (iEvent, iSetup, 
			  uncorMET, Muons, muonPtMin, muonEtaRange, 
			  muonTrackD0Max, muonTrackDzMax, 
			  muonNHitsMin, muonDPtMax, muonChiSqMax, 
			  muonDepositCor, trackAssociator, trackAssociatorParameters,
			  corMET);
}

void MuonMETAlgo::run(edm::Event& iEvent, 
		      const edm::EventSetup& iSetup,
		      const METCollection& uncorMET, 
		      const edm::View<reco::Muon>& Muons,
		      double muonPtMin,
		      double muonEtaRange,
		      double muonTrackD0Max,
		      double muonTrackDzMax,
		      int    muonNHitsMin,
		      double muonDPtMax,
		      double muonChiSqMax,
		      bool   muonDepositCor,
		      TrackDetectorAssociator& trackAssociator,
		      TrackAssociatorParameters& trackAssociatorParameters,
		      METCollection* corMET) 
{
  return MuonMETAlgo_run (iEvent, iSetup, 
			  uncorMET, Muons, muonPtMin, muonEtaRange, 
			  muonTrackD0Max, muonTrackDzMax, 
			  muonNHitsMin, muonDPtMax, muonChiSqMax, 
			  muonDepositCor, trackAssociator, trackAssociatorParameters,
			  corMET);
}  
