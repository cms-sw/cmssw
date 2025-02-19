#ifndef HLTJetSortedVBFFilter_h
#define HLTJetSortedVBFFilter_h

/** \class HLTJetSortedVBFFilter
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a
 *  single jet requirement with an Energy threshold (not Et!)
 *  Based on HLTSinglet
 *
 *  $Date: 2012/02/06 15:09:21 $
 *  $Revision: 1.3 $
 *
 *  \author Jacopo Bernardini
 *
 */

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include<string>

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
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&,trigger::TriggerFilterObjectWithRefs& filterproduct);
      
 private:
  edm::InputTag inputJets_; 
  edm::InputTag inputJetTags_; 
  double mqq_;           
  double detaqq_; 
  double detabb_;        
  double ptsqq_;          
  double ptsbb_; 
  double seta_; 
  std::string value_;
  int triggerType_;
};

#endif
