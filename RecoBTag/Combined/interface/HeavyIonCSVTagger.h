#ifndef RecoBTag_Combined_HeavyIonCSVTagger_h
#define RecoBTag_Combined_HeavyIonCSVTagger_h

#include "FWCore/Framework/interface/ESConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/MVAUtils/interface/TMVAEvaluator.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SecondaryVertex/interface/CombinedSVComputer.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"

#include <memory>
#include <type_traits>

/** \class HeavyIonCSVTagger
 *  \author M. Nguyen
 *  copied from CharmTagger.h (by M. Verzetti)
 */

class HeavyIonCSVTagger : public JetTagComputer {
public:
  struct Tokens {
    Tokens(const edm::ParameterSet& configuration, edm::ESConsumesCollector&& cc);
    edm::ESGetToken<GBRForest, GBRWrapperRcd> gbrForest_;
  };

  /// explicit ctor
  HeavyIonCSVTagger(const edm::ParameterSet&, Tokens);
  ~HeavyIonCSVTagger() override;
  float discriminator(const TagInfoHelper& tagInfo) const override;
  void initialize(const JetTagComputerRecord& record) override;

  typedef std::vector<edm::ParameterSet> vpset;

  struct MVAVar {
    std::string name;
    reco::btau::TaggingVariableName id;
    size_t index;
    bool has_index;
    float default_value;
  };

private:
  std::unique_ptr<TMVAEvaluator> mvaID_;
  CombinedSVComputer sv_computer_;
  std::vector<MVAVar> variables_;

  std::string mva_name_;
  edm::FileInPath weight_file_;
  bool use_GBRForest_;
  bool use_adaBoost_;
  Tokens tokens_;
};

#endif
