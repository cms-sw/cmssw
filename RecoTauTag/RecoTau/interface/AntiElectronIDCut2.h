//--------------------------------------------------------------------------------------------------
// AntiElectronIDCut2
//
// Helper Class for applying simple cut based anti-electron discrimination
//
// Authors: A Nayak
//--------------------------------------------------------------------------------------------------

#ifndef RECOTAUTAG_RECOTAU_AntiElectronIDCut2_H
#define RECOTAUTAG_RECOTAU_AntiElectronIDCut2_H

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <vector>

typedef std::pair<double, double> pdouble;

class AntiElectronIDCut2 
{
  public:

    AntiElectronIDCut2();
    ~AntiElectronIDCut2(); 

    double Discriminator(float TauPt,
                         float TauEta,
                         float TauLeadChargedPFCandPt,
                         float TauLeadChargedPFCandEtaAtEcalEntrance,
                         float TauLeadPFChargedHadrEoP,
                         float TauHcal3x3OverPLead,
                         float TauGammaEtaMom,
                         float TauGammaPhiMom,
                         float TauGammaEnFrac
			 );

    double Discriminator(float TauPt,
			 float TauEta,
			 float TauLeadChargedPFCandPt,
			 float TauLeadChargedPFCandEtaAtEcalEntrance,
			 float TauLeadPFChargedHadrEoP,
			 float TauHcal3x3OverPLead,
			 const std::vector<float>& GammasdEta,
			 const std::vector<float>& GammasdPhi,
			 const std::vector<float>& GammasPt
			 );

    template<typename T> 
      double Discriminator(const T& thePFTau)
      {
	float TauLeadChargedPFCandEtaAtEcalEntrance = -99.;
	float TauLeadChargedPFCandPt = -99.;
	const std::vector<reco::PFCandidatePtr>& signalPFCands = thePFTau.signalPFCands();
	for ( std::vector<reco::PFCandidatePtr>::const_iterator pfCandidate = signalPFCands.begin();
	      pfCandidate != signalPFCands.end(); ++pfCandidate ) {
	  const reco::Track* track = nullptr;
	  if ( (*pfCandidate)->trackRef().isNonnull() ) track = (*pfCandidate)->trackRef().get();
	  else if ( (*pfCandidate)->muonRef().isNonnull() && (*pfCandidate)->muonRef()->innerTrack().isNonnull()  ) track = (*pfCandidate)->muonRef()->innerTrack().get();
	  else if ( (*pfCandidate)->muonRef().isNonnull() && (*pfCandidate)->muonRef()->globalTrack().isNonnull() ) track = (*pfCandidate)->muonRef()->globalTrack().get();
	  else if ( (*pfCandidate)->muonRef().isNonnull() && (*pfCandidate)->muonRef()->outerTrack().isNonnull()  ) track = (*pfCandidate)->muonRef()->outerTrack().get();
	  else if ( (*pfCandidate)->gsfTrackRef().isNonnull() ) track = (*pfCandidate)->gsfTrackRef().get();
	  if ( track ) {
	    if ( track->pt() > TauLeadChargedPFCandPt ) {
	      TauLeadChargedPFCandEtaAtEcalEntrance = (*pfCandidate)->positionAtECALEntrance().eta();
	      TauLeadChargedPFCandPt = track->pt();
	    }
	  }
	}
	
	float TauPt = thePFTau.pt();
	float TauEta = thePFTau.eta();
	//float TauLeadPFChargedHadrHoP = 0.;
	float TauLeadPFChargedHadrEoP = 0.;
	if ( thePFTau.leadPFChargedHadrCand()->p() > 0. ) {
	  //TauLeadPFChargedHadrHoP = thePFTau.leadPFChargedHadrCand()->hcalEnergy()/thePFTau.leadPFChargedHadrCand()->p();
	  TauLeadPFChargedHadrEoP = thePFTau.leadPFChargedHadrCand()->ecalEnergy()/thePFTau.leadPFChargedHadrCand()->p();
	}
	
	std::vector<float> GammasdEta;
	std::vector<float> GammasdPhi;
	std::vector<float> GammasPt;
	for ( unsigned i = 0 ; i < thePFTau.signalPFGammaCands().size(); ++i ) {
	  reco::PFCandidatePtr gamma = thePFTau.signalPFGammaCands().at(i);
	  if ( thePFTau.leadPFChargedHadrCand().isNonnull() ) {
	    GammasdEta.push_back(gamma->eta() - thePFTau.leadPFChargedHadrCand()->eta());
	    GammasdPhi.push_back(gamma->phi() - thePFTau.leadPFChargedHadrCand()->phi());
	  } else {
	    GammasdEta.push_back(gamma->eta() - thePFTau.eta());
	    GammasdPhi.push_back(gamma->phi() - thePFTau.phi());
	  }
	  GammasPt.push_back(gamma->pt());
	}
	
	float TauHcal3x3OverPLead = thePFTau.hcal3x3OverPLead();
	
	return Discriminator(TauPt,
			     TauEta,
			     TauLeadChargedPFCandPt,
			     TauLeadChargedPFCandEtaAtEcalEntrance,
			     TauLeadPFChargedHadrEoP,
			     TauHcal3x3OverPLead,
			     GammasdEta,
			     GammasdPhi,
			     GammasPt
			     );
      };
    
    void SetBarrelCutValues(float TauLeadPFChargedHadrEoP_min,
			    float TauLeadPFChargedHadrEoP_max,
			    float TauHcal3x3OverPLead_max,
			    float TauGammaEtaMom_max,
			    float TauGammaPhiMom_max,
			    float TauGammaEnFrac_max
			    );

    void SetEndcapCutValues(float TauLeadPFChargedHadrEoP_min_1,
                            float TauLeadPFChargedHadrEoP_max_1,
			    float TauLeadPFChargedHadrEoP_min_2,
                            float TauLeadPFChargedHadrEoP_max_2,
                            float TauHcal3x3OverPLead_max,
                            float TauGammaEtaMom_max,
                            float TauGammaPhiMom_max,
                            float TauGammaEnFrac_max
                            );
    void ApplyCut_EcalCrack(bool keepAll_, bool rejectAll_){
      keepAllInEcalCrack_ = keepAll_;
      rejectAllInEcalCrack_ = rejectAll_;
    };
    
    void ApplyCuts(bool applyCut_hcal3x3OverPLead, 
		   bool applyCut_leadPFChargedHadrEoP, 
		   bool applyCut_GammaEtaMom, 
		   bool applyCut_GammaPhiMom, 
		   bool applyCut_GammaEnFrac, 
		   bool applyCut_HLTSpecific
		   );

    void SetEcalCracks(const std::vector<pdouble>& etaCracks)
    {
      ecalCracks_.clear();
      for(size_t i = 0; i < etaCracks.size(); i++)
	ecalCracks_.push_back(etaCracks[i]);
    }
    
 private:

    bool isInEcalCrack(double eta) const; 
    
    float TauLeadPFChargedHadrEoP_barrel_min_;
    float TauLeadPFChargedHadrEoP_barrel_max_;
    float TauHcal3x3OverPLead_barrel_max_;
    float TauGammaEtaMom_barrel_max_;
    float TauGammaPhiMom_barrel_max_;
    float TauGammaEnFrac_barrel_max_;
    float TauLeadPFChargedHadrEoP_endcap_min1_;
    float TauLeadPFChargedHadrEoP_endcap_max1_;
    float TauLeadPFChargedHadrEoP_endcap_min2_;
    float TauLeadPFChargedHadrEoP_endcap_max2_;
    float TauHcal3x3OverPLead_endcap_max_;
    float TauGammaEtaMom_endcap_max_;
    float TauGammaPhiMom_endcap_max_;
    float TauGammaEnFrac_endcap_max_;

    bool keepAllInEcalCrack_;
    bool rejectAllInEcalCrack_;

    bool Tau_applyCut_hcal3x3OverPLead_;
    bool Tau_applyCut_leadPFChargedHadrEoP_;
    bool Tau_applyCut_GammaEtaMom_;
    bool Tau_applyCut_GammaPhiMom_;
    bool Tau_applyCut_GammaEnFrac_;
    bool Tau_applyCut_HLTSpecific_;

    std::vector<pdouble> ecalCracks_;
    
    int verbosity_;
};

#endif
