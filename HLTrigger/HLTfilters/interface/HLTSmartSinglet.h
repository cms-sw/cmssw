#ifndef HLTSmartSinglet_h
#define HLTSmartSinglet_h

/** \class HLTSmartSinglet
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a smart HLT
 *  trigger cut, specified as a string such as "pt>15 && -3<eta<3",
 *  for single objects of the same physics type, allowing to cut on
 *  variables relating to both the base class T and the derived actual
 *  class
 *
 *  $Date: 2012/02/24 13:13:47 $
 *  $Revision: 1.11 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include<vector>

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include<string>

//
// class declaration
//

template<typename T>
class HLTSmartSinglet : public HLTFilter {

   public:

      explicit HLTSmartSinglet(const edm::ParameterSet&);
      ~HLTSmartSinglet();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

   private:
      edm::InputTag inputTag_; // input tag identifying product
      int triggerType_;        // triggerType
      std::string   cut_;      // smart cut
      int           min_N_;    // number of objects passing cuts required

      StringCutObjectSelector<T,true> select_; // smart selector
};

#endif //HLTSmartSinglet_h
