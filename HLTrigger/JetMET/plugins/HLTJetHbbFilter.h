#ifndef HLTJetHbbFilter_h
#define HLTJetHbbFilter_h

/** \class HLTJetHbbFilter
 *
 *
 *  This class is an HLTFilter (-> EDFilter) 
 *  implementing a two b-jet invariant mass requirement 
 *  Based on HLTJetSortedVBFFilter
 *
 *
 *  \author Ann Wang, Javier Duarte
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

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "TVector3.h"
#include "TLorentzVector.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//
template<typename T>
class HLTJetHbbFilter : public HLTFilter {
 public:
  typedef std::pair<double,unsigned int> Jpair;
  static bool comparator ( const Jpair& l, const Jpair& r) {
    return l.first < r.first;
  }

  explicit HLTJetHbbFilter(const edm::ParameterSet&);
  ~HLTJetHbbFilter();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  static float findCSV(const  typename std::vector<T>::const_iterator & jet, const reco::JetTagCollection & jetTags);	
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&,trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

 private:
  edm::EDGetTokenT<std::vector<T>> m_theJetsToken;
  edm::EDGetTokenT<reco::JetTagCollection> m_theJetTagsToken;
  edm::InputTag inputJets_;
  edm::InputTag inputJetTags_;
  double minmbb_;
  double maxmbb_;
  double minptb1_;
  double minptb2_;
  double maxetab_; 
  double minptbb_;
  double maxptbb_;
  double mintag1_;
  double mintag2_;
  double maxtag_;
  int triggerType_;
};

#endif
