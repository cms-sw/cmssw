#ifndef RecoBTag_CTagging_CharmTagger_h
#define RecoBTag_CTagging_CharmTagger_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/MVAUtils/interface/TMVAEvaluator.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include <mutex>
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "RecoBTag/SecondaryVertex/interface/CombinedSVSoftLeptonComputer.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"

#include <memory>

/** \class CharmTagger
 *  \author M. Verzetti, U. Rochester, N.Y.
 *  copied from ElectronTagger.h
 */

class CharmTagger : public JetTagComputer {
public:
  /// explicit ctor 
	CharmTagger(const edm::ParameterSet & );
	~CharmTagger() override;//{}
  float discriminator(const TagInfoHelper & tagInfo) const override;
	void initialize(const JetTagComputerRecord & record) override;
	
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
	CombinedSVSoftLeptonComputer sl_computer_;
	std::vector<MVAVar> variables_;

	std::string mva_name_;
  bool use_condDB_;
	std::string gbrForest_label_;
	edm::FileInPath weight_file_;
  bool use_GBRForest_;
  bool use_adaBoost_;
  bool defaultValueNoTracks_;
};

#endif
