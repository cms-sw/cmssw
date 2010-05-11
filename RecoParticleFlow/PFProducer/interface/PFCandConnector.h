#ifndef PFProducer_PFCandConnector_H_
#define PFProducer_PFCandConnector_H_

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// \author : M. Gouzevitch
// \date : May 2010

/// Based on a class from : V. Roberfroid, February 2008

class PFCandConnector {
    
    public :
       
       PFCandConnector( ) { 
	 pfC_ = std::auto_ptr<reco::PFCandidateCollection>(new reco::PFCandidateCollection); 
	 debug_ = false;
	 bCorrect_ = false;
	 bCalibPrimary_ =  false;
	 bCalibSecondary_ =  false;
	 fConst_ = 1, fNorm_ = 0, fExp_ = 0;
       }
       
       void setParameters(const edm::ParameterSet& iCfgCandConnector){
	 /// Flag to apply the correction procedure for nuclear interactions
	 bCorrect_ = iCfgCandConnector.getParameter<bool>("bCorrect");
	 /// Flag to calibrate the reconstructed nuclear interactions with primary or merged tracks
	 bCalibPrimary_ =  iCfgCandConnector.getParameter<bool>("bCalibPrimary");
	 /// Flag to calibrate the reconstructed nuclear interactions where only secondary tracks are present
	 bCalibSecondary_ =  iCfgCandConnector.getParameter<bool>("bCalibPrimary");
	 std::vector<double> nuclCalibFactors = iCfgCandConnector.getParameter<std::vector<double> >("nuclCalibFactors");  
	 if (nuclCalibFactors.size() == 3) {
	   fConst_ =  nuclCalibFactors[0]; fNorm_ = nuclCalibFactors[1]; fExp_ = nuclCalibFactors[2];
	 } else {
	   std::cout << "Wrong calibration factors for nuclear interactions. The calibration procedure would not be applyed." << std::endl;
	   bCalibPrimary_ =  false;
	   bCalibSecondary_ =  false;
	 }
       }

       void setDebug( bool debug ) {debug_ = debug;}

       

       std::auto_ptr<reco::PFCandidateCollection> connect(std::auto_ptr<reco::PFCandidateCollection>& pfCand);

       
 
    private :

       /// Analyse nuclear interactions where a primary or merged track is present
       void analyseNuclearWPrim(std::auto_ptr<reco::PFCandidateCollection>&, unsigned int);

       /// Analyse nuclear interactions where a secondary track is present
       void analyseNuclearWSec(std::auto_ptr<reco::PFCandidateCollection>&, unsigned int);

       bool isPrimaryNucl( const reco::PFCandidate& pf ) const;

       bool isSecondaryNucl( const reco::PFCandidate& pf ) const;

       /// Return a calibration factor for a reconstructed nuclear interaction
       double rescaleFactor( const double pt ) const;

       /// Collection of primary PFCandidates to be transmitted to the Event
       std::auto_ptr<reco::PFCandidateCollection> pfC_;
       /// A mask to define the candidates which shall not be transmitted
       std::vector<bool> bMask_;

       /// Parameters
       bool debug_;
       bool bCorrect_;

       /// Calibration parameters for the reconstructed nuclear interactions
       bool bCalibPrimary_;
       bool bCalibSecondary_;
       double fConst_, fNorm_, fExp_;

       /// Useful constants
       static const double pion_mass2;
       static const reco::PFCandidate::Flags fT_TO_DISP_;
       static const reco::PFCandidate::Flags fT_FROM_DISP_;
};

#endif
