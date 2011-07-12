#ifndef RECOJETS_JETALGORITHMS_SUBJETFILTERALGORITHM_H
#define RECOJETS_JETALGORITHMS_SUBJETFILTERALGORITHM_H 1

#include <vector>

#include "RecoJets/JetAlgorithms/interface/CompoundPseudoJet.h"
#include "FWCore/Framework/interface/Event.h"

#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>




class SubjetFilterAlgorithm
{
  //
  // construction / destruction
  //
public:
  SubjetFilterAlgorithm(const std::string& moduleLabel,
			const std::string& jetAlgorithm,
			unsigned nFatMax, double rParam, double jetPtMin,
			double massDropCut, double asymmCut,
			bool asymmCutLater);
  virtual ~SubjetFilterAlgorithm();
  

  //
  // member functions
  //
public:
  void run(const std::vector<fastjet::PseudoJet>& inputs, 
	   std::vector<CompoundPseudoJet>& fatJets,
	   const edm::EventSetup& iSetup);
  
  std::string summary() const;
  
  
  //
  // member data
  //
private:
  std::string             moduleLabel_;
  std::string             jetAlgorithm_;
  unsigned                nFatMax_;
  double                  rParam_;
  double                  jetPtMin_;
  double                  massDropCut_;
  double                  asymmCut2_;
  bool                    asymmCutLater_;
  
  unsigned                ntotal_;
  unsigned                nfound_;

  fastjet::JetDefinition* fjJetDef_;

};


#endif
