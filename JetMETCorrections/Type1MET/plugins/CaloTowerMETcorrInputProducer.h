#ifndef JetMETCorrections_Type1MET_CaloTowerMETcorrInputProducer_h
#define JetMETCorrections_Type1MET_CaloTowerMETcorrInputProducer_h

/** \class CaloTowerMETcorrInputProducer
 *
 * Sum CaloTowers not within jets ("unclustered energy"),
 * needed as input for Type 2 MET corrections
 *
 * \authors Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: CaloTowerMETcorrInputProducer.h,v 1.1 2013/02/22 15:38:43 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"

#include "DataFormats/METReco/interface/CorrMETData.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <string>

class CaloTowerMETcorrInputProducer : public edm::EDProducer  
{
 public:

  explicit CaloTowerMETcorrInputProducer(const edm::ParameterSet&);
  ~CaloTowerMETcorrInputProducer();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  edm::InputTag src_; // CaloTower input collection

  struct binningEntryType
  {
    binningEntryType()
      : binLabel_(""),
	binLabel_em_("em"),
	binLabel_had_("had"),
        binSelection_(0)
    {}
    binningEntryType(const edm::ParameterSet& cfg)
    : binLabel_(cfg.getParameter<std::string>("binLabel")),
      binSelection_(new StringCutObjectSelector<reco::Candidate::LorentzVector>(cfg.getParameter<std::string>("binSelection")))
    {
      binLabel_em_ = std::string(binLabel_).append("em");
      binLabel_had_ = std::string(binLabel_).append("had");
    }
    ~binningEntryType() 
    {
      delete binSelection_;
    }
    std::string binLabel_;
    std::string binLabel_em_;
    std::string binLabel_had_;
    StringCutObjectSelector<reco::Candidate::LorentzVector>* binSelection_;
    CorrMETData binUnclEnergySum_;
    CorrMETData binUnclEnergySum_em_;
    CorrMETData binUnclEnergySum_had_;
  };
  std::vector<binningEntryType*> binning_;

  std::string residualCorrLabel_;
  double residualCorrEtaMax_;
  double residualCorrOffset_;
  double extraCorrFactor_;
  FactorizedJetCorrector* residualCorrectorFromFile_;
  bool isMC_;

  double globalThreshold_;
  bool noHF_;

  int verbosity_;
};

#endif


 

