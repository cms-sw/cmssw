#ifndef JetMETCorrections_Type1MET_PFCandMETcorrInputProducer_h
#define JetMETCorrections_Type1MET_PFCandMETcorrInputProducer_h

/** \class PFCandMETcorrInputProducer
 *
 * Sum PFCandidates not within jets ("unclustered energy"),
 * needed as input for Type 2 MET corrections
 *
 * \authors Michael Schmitt, Richard Cavanaugh, The University of Florida
 *          Florent Lacroix, University of Illinois at Chicago
 *          Christian Veelken, LLR
 *
 *
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/METReco/interface/CorrMETData.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <string>

class PFCandMETcorrInputProducer : public edm::stream::EDProducer<> {
public:
  explicit PFCandMETcorrInputProducer(const edm::ParameterSet&);
  ~PFCandMETcorrInputProducer() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  std::string moduleLabel_;

  edm::EDGetTokenT<edm::View<reco::Candidate>> token_;
  edm::EDGetTokenT<edm::ValueMap<float>> weightsToken_;

  struct binningEntryType {
    binningEntryType() : binLabel_(""), binSelection_(nullptr) {}
    binningEntryType(const edm::ParameterSet& cfg)
        : binLabel_(cfg.getParameter<std::string>("binLabel")),
          binSelection_(new StringCutObjectSelector<reco::Candidate::LorentzVector>(
              cfg.getParameter<std::string>("binSelection"))) {}
    ~binningEntryType() {}
    const std::string binLabel_;
    std::unique_ptr<const StringCutObjectSelector<reco::Candidate::LorentzVector>> binSelection_;
    CorrMETData binUnclEnergySum_;
  };
  std::vector<std::unique_ptr<binningEntryType>> binning_;
};

#endif
