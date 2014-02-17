#ifndef HLTReco_TriggerFilterObjectWithRefs_h
#define HLTReco_TriggerFilterObjectWithRefs_h

/** \class trigger::TriggerFilterObjectWithRefs
 *
 *  If HLT cuts of intermediate or final HLT filters are satisfied,
 *  instances of this class hold the combination of reconstructed
 *  physics objects (e/gamma/mu/jet/MMet...) satisfying the cuts.
 *
 *  This implementation is not completely space-efficient as some
 *  physics object containers may stay empty. However, the big
 *  advantage is that the solution is generic, i.e., works for all
 *  possible HLT filters. Hence we accept the reasonably small
 *  overhead of empty containers.
 *
 *  $Date: 2010/02/15 17:40:49 $
 *  $Revision: 1.15 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>
#include <vector>
#include <algorithm>

namespace trigger
{

  /// Transient book-keeping EDProduct filled by HLTFilter module to
  /// record physics objects firing the filter (never persistet in
  /// production; same functionality but different implementation
  /// compared to the old HLT data model's HLTFilterObjectWithRefs
  /// class)
  class TriggerFilterObjectWithRefs : public TriggerRefsCollections {

  /// data members
  private:
    int path_;
    int module_;
    std::vector<std::string> collectionTags_;

  /// methods
  public:
    /// constructors
    TriggerFilterObjectWithRefs():
      TriggerRefsCollections(),
      path_(-9),
      module_(-9),
      collectionTags_() { }
    
    TriggerFilterObjectWithRefs(int path, int module):
      TriggerRefsCollections(),
      path_(path),
      module_(module),
      collectionTags_() { }
    
    /// accessors
    int path() const {return path_;}
    int module() const {return module_;}
    
    /// collectionTags
    void addCollectionTag(const edm::InputTag& collectionTag){
      collectionTags_.push_back(collectionTag.encode());
    }
    
    void getCollectionTags(std::vector<edm::InputTag>& collectionTags) const {
      const trigger::size_type n(collectionTags_.size());
      collectionTags.resize(n);
      for (trigger::size_type i=0; i!=n; ++i) {
	collectionTags[i]=edm::InputTag(collectionTags_[i]);
      }
    }

    /// low-level technical accessor
    const std::vector<std::string>& getCollectionTagsAsStrings() const {
      return collectionTags_;
    }

    /// utility
    void swap(TriggerFilterObjectWithRefs & other) {
      TriggerRefsCollections::swap(other);                  // swap base instance
      std::swap(path_,           other.path_);
      std::swap(module_,         other.module_);
      std::swap(collectionTags_, other.collectionTags_);    // use specialized version for STL containers
    }

  };

  // picked up via argument dependent lookup, e-g- by boost::swap()
  inline void swap(TriggerFilterObjectWithRefs & first, TriggerFilterObjectWithRefs & second) {
    first.swap(second);
  }

}

#endif
