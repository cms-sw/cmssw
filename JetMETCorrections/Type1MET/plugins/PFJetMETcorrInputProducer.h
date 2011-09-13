#ifndef JetMETCorrections_Type1MET_PFJetMETcorrInputProducer_h
#define JetMETCorrections_Type1MET_PFJetMETcorrInputProducer_h

/** \class PFJetMETcorrInputProducer
 *
 * Produce Type 1 + 2 MET corrections corresponding to differences
 * between raw PFJets and PFJets with jet energy corrections (JECs) applied
 *
 * \authors Michael Schmitt, Richard Cavanaugh, The University of Florida
 *          Florent Lacroix, University of Illinois at Chicago
 *          Christian Veelken, LLR
 *
 * \version $Revision: 1.00 $
 *
 * $Id: PFJetMETcorrInputProducer.h,v 1.18 2011/05/30 15:19:41 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include <string>

class PFJetMETcorrInputProducer : public edm::EDProducer  
{
 public:

  explicit PFJetMETcorrInputProducer(const edm::ParameterSet&);
  ~PFJetMETcorrInputProducer();
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  edm::InputTag src_; // PFJet input collection

  std::string offsetCorrLabel_; // e.g. 'ak5PFJetL1Fastjet'
  std::string jetCorrLabel_;    // e.g. 'ak5PFJetL1FastL2L3' (MC) / 'ak5PFJetL1FastL2L3Residual' (Data)

  double jetCorrEtaMax_; // do not use JEC factors for |eta| above this threshold (recommended default = 4.7),
                         // in order to work around problem with CMSSW_4_2_x JEC factors at high eta,
                         // reported in
                         //  https://hypernews.cern.ch/HyperNews/CMS/get/jes/270.html
                         //  https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/1259/1.html

  double type1JetPtThreshold_; // threshold to distinguish between jets entering Type 1 MET correction
                               // and jets entering "unclustered energy" sum
                               // NOTE: threshold is applied on **corrected** jet energy (recommended default = 10 GeV)

  bool skipEM_; // flag to exclude jets with large fraction of electromagnetic energy (electrons/photons) 
                // from Type 1 + 2 MET corrections
  double skipEMfractionThreshold_;

  bool skipMuons_; // flag to subtract momentum of muons (provided muons pass selection cuts) which are within jets
                   // from jet energy before compute JECs/propagating JECs to Type 1 + 2 MET corrections
  StringCutObjectSelector<reco::Muon>* skipMuonSelection_;
};

#endif



 

