// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      METSignificance
// 
/**\class METSignificance METSignificance.cc RecoMET/METAlgorithms/src/METSignificance.cc
Description: [one line class summary]
Implementation:
[Notes on implementation]
*/
//
// Original Author:  Nathan Mirman (Cornell University)
//         Created:  Thu May 30 16:39:52 CDT 2013
//
//
#ifndef METAlgorithms_METSignificance_h
#define METAlgorithms_METSignificance_h
//____________________________________________________________________________||
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "TMatrixD.h"

//____________________________________________________________________________||
class METSignificance {
   public:
      explicit METSignificance() {  };
      ~METSignificance() {  };

      TMatrixD getCovariance();
      double getSignificance(TMatrixD&);

      void addJets( const std::vector<reco::Jet>& );
      void addLeptons( const std::vector<reco::Candidate::LorentzVector>& );
      void addCandidates( const std::vector<reco::Candidate::LorentzVector>& );
      void addMET( const reco::MET& );

      void setThreshold( const double& );
      void setJetEtaBins( const std::vector<double>& );
      void setJetParams( const std::vector<double>& );
      void setPJetParams( const std::vector<double>& );
      void setResFiles( const std::string&, const std::string& );

   private:
      std::vector<reco::Jet> cleanJets(double, double);

      std::vector<reco::Jet> jets;
      std::vector<reco::Candidate::LorentzVector> leptons;
      std::vector<reco::Candidate::LorentzVector> candidates;
      reco::MET met;

      double jetThreshold;
      std::vector<double> jetetas;
      std::vector<double> jetparams;
      std::vector<double> pjetparams;

      std::string ptResFileName;
      std::string phiResFileName;

};

//____________________________________________________________________________||
#endif // METAlgorithms_METSignificance_h
