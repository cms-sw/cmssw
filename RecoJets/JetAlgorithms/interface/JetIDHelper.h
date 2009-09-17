#ifndef RecoJets_JetAlgorithms_interface_JetIDHelper_h
#define RecoJets_JetAlgorithms_interface_JetIDHelper_h


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"


namespace reco {

  namespace helper {

    class JetIDHelper {

    public : 
      // construction
      JetIDHelper() {}
      JetIDHelper( edm::ParameterSet const & pset );
      ~JetIDHelper() {} 

      void fillDescription(edm::ParameterSetDescription& iDesc);

      void initValues ();

      // interface
      void calculate( const edm::Event& event, const reco::CaloJet &jet, const int iDbg = 0 );

      // member access
      
      // these require RecHits, so can not be evaluated from AOD
      double fHPD()          const    { return    fHPD_;}           
      double fRBX()          const    { return    fRBX_;}           
      int    n90Hits()       const    { return    n90Hits_;}
      // these are tower based
      double fSubDetector1() const    { return    fSubDetector1_;}  
      double fSubDetector2() const    { return    fSubDetector2_;}  
      double fSubDetector3() const    { return    fSubDetector3_;}  
      double fSubDetector4() const    { return    fSubDetector4_;}  
      double restrictedEMF() const    { return    restrictedEMF_;}
      int    nHCALTowers()   const    { return    nHCALTowers_;}    
      int    nECALTowers()   const    { return    nECALTowers_;}    
      // tower based approximations / inferior options
      double approximatefHPD() const { return    approximatefHPD_;}           
      double approximatefRBX() const { return    approximatefRBX_;}           
      int    hitsInN90()        const { return    hitsInN90_;}        
 
      struct subtower { // contents of a sub-detector's tower
	double E;
	int Nhit;
	
	subtower( double xE, int xN ) { E = xE; Nhit = xN; }
      };
      
  
    private:

     
      // helper functions
      void calculateSubDetectorEnergies( const edm::Event& event, const reco::CaloJet &jet, 
					 std::vector< double > &subdet_energies );
      void classifyJetComponents( const edm::Event& event, const reco::CaloJet &jet, 
				  std::vector< double > &energies,
				  std::vector< double > &Ecal_energies, std::vector< double > &Hcal_energies, 
				  std::vector< double > &HO_energies,
				  std::vector< double > &HPD_energies,  std::vector< double > &RBX_energies,
				  const bool recHitBased = true, const int iDbg = 0);

      void classifyJetTowers( const edm::Event& event, const reco::CaloJet &jet, 
			      std::vector< subtower > &subtowers,      
			      std::vector< subtower > &Ecal_subtowers, 
			      std::vector< subtower > &Hcal_subtowers, 
			      std::vector< subtower > &HO_subtowers,
			      std::vector< double > &HPD_energies,  
			      std::vector< double > &RBX_energies,
			      const int iDbg = 0);

      unsigned int nCarrying( double fraction, std::vector< double > descending_energies );
      unsigned int hitsInNCarrying( double fraction, std::vector< subtower > descending_towers );
      
      enum Region{
	unknown_region = -1,
	HFneg, HEneg, HBneg, HBpos, HEpos, HFpos };
      
      int HBHE_oddness( int iEta, int depth );
      Region HBHE_region( int iEta, int depth );
      // tower-based. -1 means can't figure it out
      int HBHE_oddness( int iEta );
      Region region( int iEta );
      

      double fHPD_;
      double fRBX_;
      int    n90Hits_;
      double fSubDetector1_;
      double fSubDetector2_;
      double fSubDetector3_;
      double fSubDetector4_;
      double restrictedEMF_;
      int    nHCALTowers_;
      int    nECALTowers_;
      double approximatefHPD_;
      double approximatefRBX_;
      int    hitsInN90_;
      
      bool useRecHits_;
      edm::InputTag hbheRecHitsColl_;
      edm::InputTag hoRecHitsColl_;
      edm::InputTag hfRecHitsColl_;
      edm::InputTag ebRecHitsColl_;
      edm::InputTag eeRecHitsColl_;
    };
  }
}
#endif
