#ifndef RecoJets_JetProducers_interface_CastorJetIDHelper_h
#define RecoJets_JetProducers_interface_CastorJetIDHelper_h


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/JetReco/interface/BasicJet.h"

namespace reco {

  namespace helper {

    class CastorJetIDHelper {

    public : 
      // construction
      CastorJetIDHelper();
      ~CastorJetIDHelper() {} 


      void initValues ();

      // interface
      void calculate( const edm::Event& event, const reco::BasicJet &jet );

      // member access
     
      double emEnergy()          const    { return    emEnergy_;}           
      double hadEnergy()          const    { return    hadEnergy_;}           
      double    fem()       const    { return    fem_;}
      double width() const    { return    width_;}  
      double depth() const    { return    depth_;}  
      double fhot() const    { return    fhot_;}  
      double sigmaz() const    { return    sigmaz_;}  
      int nTowers()           const    { return    nTowers_;}
      
  
    private:

     
      // helper functions
      double phiangle (double testphi);

      double emEnergy_;
      double hadEnergy_; 
      double fem_;
      double width_;
      double depth_;
      double fhot_;
      double sigmaz_;
      int nTowers_;

      static int sanity_checks_left_;
    };
  }
}
#endif
