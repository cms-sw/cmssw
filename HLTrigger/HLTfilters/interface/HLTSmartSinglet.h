#ifndef HLTSmartSinglet_h
#define HLTSmartSinglet_h

/** \class HLTSmartSinglet
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a smart HLT
 *  trigger cut, specified as a string such as "pt>15 && -3<eta<3",
 *  for single objects of the same physics type, allowing to cut on
 *  variables relating to their 4-momentum representation
 *
 *  $Date: 2008/05/05 15:48:33 $
 *  $Revision: 1.4 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include<vector>

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include<string>

//
// class declaration
//

template<typename T, int Tid>
class HLTSmartSinglet : public HLTFilter {

   public:

      explicit HLTSmartSinglet(const edm::ParameterSet&);
      ~HLTSmartSinglet();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputTag_; // input tag identifying product
      bool          saveTag_;  // whether to save this tag
      std::string   cut_;      // smart cut
      int           min_N_;    // number of objects passing cuts required

      StringCutObjectSelector<T> select_; // smart selector
};

#endif //HLTSmartSinglet_h
