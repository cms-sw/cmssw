#ifndef HLTJetSortedVBFFilter_h
#define HLTJetSortedVBFFilter_h

/** \class HLTJetSortedVBFFilter
 *
 *
 *  This class is an HLTFilter (-> EDFilter) implementing a
 *  single jet requirement with an Energy threshold (not Et!)
 *  Based on HLTSinglet
 *
 *
 *  \author Jacopo Bernardini
 *
 */

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include <string>
#include <vector>
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Handle.h"
namespace edm {
	class ConfigurationDescriptions;
}

//
// class declaration
//
template<typename T>
class HLTJetSortedVBFFilter : public HLTFilter {
	public:
		typedef std::pair<double,unsigned int> Jpair;
		static bool comparator ( const Jpair& l, const Jpair& r) {
			return l.first < r.first;
		}

		explicit HLTJetSortedVBFFilter(const edm::ParameterSet&);
		~HLTJetSortedVBFFilter();
		static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
		static float findCSV(const  typename std::vector<T>::const_iterator & jet, const reco::JetTagCollection & jetTags);	
		virtual bool hltFilter(edm::Event&, const edm::EventSetup&,trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

	private:
		edm::EDGetTokenT<std::vector<T>> m_theJetsToken;
		edm::EDGetTokenT<reco::JetTagCollection> m_theJetTagsToken;
		edm::InputTag inputJets_;
		edm::InputTag inputJetTags_;
		double mqq_;
		double detaqq_;
		double detabb_;
		double dphibb_;
		double ptsqq_;
		double ptsbb_;
		double seta_;
		double njets_;
		std::string value_;
		int triggerType_;
};

#endif

