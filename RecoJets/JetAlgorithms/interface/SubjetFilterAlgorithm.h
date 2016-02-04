#ifndef RECOJETS_JETALGORITHMS_SUBJETFILTERALGORITHM_H
#define RECOJETS_JETALGORITHMS_SUBJETFILTERALGORITHM_H 1


/*
  Implementation of the subjet/filter jet reconstruction algorithm
  which is described in: http://arXiv.org/abs/0802.2470
  
  CMSSW implementation by David Lopes-Pegna           <david.lopes-pegna@cern.ch>
                      and Philipp Schieferdecker <philipp.schieferdecker@cern.ch>

  see: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSubjetFilterJetProducer

*/


#include <vector>

#include "RecoJets/JetAlgorithms/interface/CompoundPseudoJet.h"
#include "FWCore/Framework/interface/Event.h"

#include <fastjet/JetDefinition.hh>
#include <fastjet/AreaDefinition.hh>
#include <fastjet/PseudoJet.hh>




class SubjetFilterAlgorithm
{
  //
  // construction / destruction
  //
public:
  SubjetFilterAlgorithm(const std::string& moduleLabel,
			const std::string& jetAlgorithm,
			unsigned nFatMax,      double rParam,
			double   rFilt,        double jetPtMin,
			double   massDropCut,  double asymmCut,
			bool     asymmCutLater,bool   doAreaFastjet,
			double   ghostEtaMax,  int    activeAreaRepeats,
			double   ghostArea,    bool   verbose);
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
  std::string              moduleLabel_;
  std::string              jetAlgorithm_;
  unsigned                 nFatMax_;
  double                   rParam_;
  double                   rFilt_;
  double                   jetPtMin_;
  double                   massDropCut_;
  double                   asymmCut2_;
  bool                     asymmCutLater_;
  bool                     doAreaFastjet_;
  double                   ghostEtaMax_;
  int                      activeAreaRepeats_;
  double                   ghostArea_;
  bool                     verbose_;
  
  unsigned                 nevents_;
  unsigned                 ntotal_;
  unsigned                 nfound_;
  
  fastjet::JetDefinition*  fjJetDef_;
  fastjet::AreaDefinition* fjAreaDef_;

};


#endif
