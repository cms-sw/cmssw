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
 *  $Date: 2007/12/04 20:40:37 $
 *  $Revision: 1.8 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"

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
    
  /// methods
  public:
    /// constructors
    TriggerFilterObjectWithRefs():
      TriggerRefsCollections(), path_(-9), module_(-9) { }
    TriggerFilterObjectWithRefs(int path, int module):
      TriggerRefsCollections(), path_(path), module_(module) { }

    int path() const {return path_;}
    int module() const {return module_;}

  };

}

#endif
