#ifndef RecoBTag_CTagging_CharmTagger_h
#define RecoBTag_CTagging_CharmTagger_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/TMVAEvaluator.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include <mutex>
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "RecoBTag/SecondaryVertex/interface/CombinedSVSoftLeptonComputer.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

/** \class CharmTagger
 *  \author M. Verzetti, U. Rochester, N.Y.
 *  copied from ElectronTagger.h
 */

class CharmTagger : public JetTagComputer {
public:
  /// explicit ctor 
	CharmTagger(const edm::ParameterSet & );
	~CharmTagger(){}
  virtual float discriminator(const TagInfoHelper & tagInfo) const override;
	typedef std::vector<edm::ParameterSet> vpset;
	
	struct MVAVar {
		std::string name;
		reco::btau::TaggingVariableName id;
		size_t index;
		bool has_index;
		float default_value;
	};

private:
  mutable std::mutex mutex_;
	[[cms::thread_guard("mutex_")]] std::unique_ptr<TMVAEvaluator> mvaID_;
	CombinedSVSoftLeptonComputer sl_computer_;
	std::vector<MVAVar> variables_;
	// std:: variables_;
	// std::vector<TaggingVariableName> name_ids_;
	// std::vector<>
};

#endif
