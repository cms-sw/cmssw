#ifndef RecoTauTag_TauTagTools_interface_DiscrminantList
#define RecoTauTag_TauTagTools_interface_DiscrminantList
// Class: DiscriminantList
// 
/*files: RecoTauTag/TauTagTools/interface/DiscriminantList.h RecoTauTag/TauTagTools/src/DiscriminantList.cc
 *
 * Description: Base point to define a list of tau discriminant objects used in an MVA training/computation chain
 *
 * Note: container class owns the Discriminant objects and will delete them upon its destruction.
 *
 * USERS: Define list of desired descriminants in ctor, @ RecoTauTag/TauTagTools/src/DiscriminantList.cc

*/
// Original Author:  Evan K.Friis, UC Davis  (friis@physics.ucdavis.edu)

#include "RecoTauTag/TauTagTools/interface/Discriminants.h"

namespace PFTauDiscriminants {
class DiscriminantList {
   public:
      typedef std::vector<Discriminant*> collection;
      typedef collection::const_iterator const_iterator;
      DiscriminantList();
      ~DiscriminantList();
      ///returns constant reference to full list
      const collection& discriminantList() { return theDiscriminants_; };
      ///iterators over the list
      const_iterator    begin()            { return theDiscriminants_.begin(); };
      const_iterator    end()              { return theDiscriminants_.end(); };

   private:
      collection theDiscriminants_;
};
}//end namespace

#endif
