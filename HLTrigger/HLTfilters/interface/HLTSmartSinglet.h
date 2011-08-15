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
 *  $Date: 2010/01/13 12:59:57 $
 *  $Revision: 1.6 $
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
      bool          saveTags_;  // whether to save this tag
      std::string   cut_;      // smart cut
      int           min_N_;    // number of objects passing cuts required

      StringCutObjectSelector<T,true> select_; // smart selector
};

#endif //HLTSmartSinglet_h
