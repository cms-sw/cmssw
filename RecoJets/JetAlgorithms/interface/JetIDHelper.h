#ifndef RecoJets_JetAlgorithms_interface_JetIDHelper_h
#define RecoJets_JetAlgorithms_interface_JetIDHelper_h


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/JetReco/interface/CaloJet.h"




namespace reco {

  namespace helper {


    // select2nd exists only in some std and boost implementations, so let's control our own fate
    // and it can't be a non-static member function.
    static double select2nd (std::map<int,double>::value_type const &pair) {return pair.second;}

    class JetID {

    public : 
      // construction
      JetID();
      ~JetID() {} 

      // interface
      void calculate( const edm::Event& event, const reco::CaloJet &jet, const int iDbg = 0 );

      // member access
      
     double fHPD()          const   { return    fHPD_;}           
     double fRBX()          const   { return    fRBX_;}           
     int    n90Hits()       const   { return    n90Hits_;}        
     double fSubDetector1() const   { return    fSubDetector1_;}  
     double fSubDetector2() const   { return    fSubDetector2_;}  
     double fSubDetector3() const   { return    fSubDetector3_;}  
     double fSubDetector4() const   { return    fSubDetector4_;}  
     double restrictedEMF() const   { return    restrictedEMF_;}  
     int    nHCALTowers()   const   { return    nHCALTowers_;}    
     int    nECALTowers()   const   { return    nECALTowers_;}    
 
    private:

      // helper functions
      void classifyJetComponents( const edm::Event& event, const reco::CaloJet &jet, 
				  std::vector< double > &energies,      std::vector< double > &subdet_energies,
				  std::vector< double > &Ecal_energies, std::vector< double > &Hcal_energies, 
				  std::vector< double > &HO_energies,
				  std::vector< double > &HPD_energies,  std::vector< double > &RBX_energies,
				  unsigned int& nHadTowers, unsigned int& nEMTowers, int iDbg = 0);

      unsigned int nCarrying( double fraction, std::vector< double > descending_energies );
 
      // if implementing as a class, the following should probably be private:

      int HBHE_oddness (int iEta, int depth);
      int HBHE_region (int iEta, int depth);
      

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
  
    };

  }

}

#endif
