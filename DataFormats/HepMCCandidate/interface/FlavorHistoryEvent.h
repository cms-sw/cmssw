#ifndef HepMCCandidate_FlavorHistoryEvent_h
#define HepMCCandidate_FlavorHistoryEvent_h

/** \class reco::FlavorHistoryEvent
 *
 * Stores a vector of information about the flavor history of partons
 * as well as a classification of the event. 
 *
 * It will return the following:
 *    nb = number of genjets that are matched to b partons
 *    nc = number of genjets that are matched to c partons
 * 
 * This can be used to classify the event, for instance, as W+bb (2 b partons),
 * W+c (1 c parton), etc.
 *
 * \author: Salvatore Rappoccio (JHU)
 *
 */

// -------------------------------------------------------------
// Identify the class of the event
//
// Reports nb, nc, nlight genjets that are matched
// to partons.
//
// -------------------------------------------------------------

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/HepMCCandidate/interface/FlavorHistory.h"

#include <fstream>

namespace reco {

  namespace helpers {
    // Helper class to decide which type of event this should be classified as.
    //
    // Decision is based on a priority weighting of:
    //  1. flavor (5 > 4)
    //  2. type:
    //      2a. Flavor decay
    //      2b. Matrix element
    //      2c. Flavor excitation
    //      2d. Gluon splitting
    //  3. delta R (if applicable)
    //
    struct FlavorHistoryEventHelper {
      FlavorHistoryEventHelper(int iflavor, FlavorHistory::FLAVOR_T iflavorSource, double idR)
          : flavor(iflavor), flavorSource(iflavorSource), dR(idR) {}
      ~FlavorHistoryEventHelper() {}

      // Data members
      int flavor;                            // pdg id
      FlavorHistory::FLAVOR_T flavorSource;  // flavor source
      double dR;                             // if there is a sister, dR

      // Comparison operators
      bool operator<(FlavorHistoryEventHelper const& right) const {
        if (flavor > right.flavor)
          return false;
        if (flavorSource > right.flavorSource)
          return false;
        if (dR > right.dR)
          return false;
        return true;
      }

      bool operator==(FlavorHistoryEventHelper const& right) const {
        return flavor == right.flavor && flavorSource == right.flavorSource && dR == right.dR;
      }

      friend std::ostream& operator<<(std::ostream& out, FlavorHistoryEventHelper helper) {
        char buff[1000];
        sprintf(buff, "Flavor = %2d, type = %2d, dR = %6f", helper.flavor, helper.flavorSource, helper.dR);
        out << buff << std::endl;
        return out;
      }
    };
  }  // namespace helpers

  class FlavorHistoryEvent {
  public:
    // convenient typedefs
    typedef FlavorHistory value_type;
    typedef std::vector<value_type> collection_type;
    typedef collection_type::size_type size_type;
    typedef collection_type::iterator iterator;
    typedef collection_type::const_iterator const_iterator;
    typedef collection_type::reverse_iterator reverse_iterator;
    typedef collection_type::const_reverse_iterator const_reverse_iterator;
    typedef collection_type::pointer pointer;
    typedef collection_type::const_pointer const_pointer;
    typedef collection_type::reference reference;
    typedef collection_type::const_reference const_reference;
    typedef FlavorHistory::FLAVOR_T flavor_type;

    FlavorHistoryEvent() { clear(); }
    ~FlavorHistoryEvent() {}

    // Set up the heavy flavor content
    void cache();
    bool isCached() const { return cached_; }

    // Accessors to heavy flavor content
    unsigned int nb() const {
      if (isCached())
        return nb_;
      else
        return 0;
    }
    unsigned int nc() const {
      if (isCached())
        return nc_;
      else
        return 0;
    }

    // Accessor to maximum delta R between highest flavor constituents
    double deltaR() const {
      if (isCached())
        return dR_;
      else
        return -1.0;
    }
    unsigned int highestFlavor() const {
      if (isCached())
        return highestFlavor_;
      else
        return 0;
    }
    flavor_type flavorSource() const {
      if (isCached())
        return flavorSource_;
      else
        return FlavorHistory::FLAVOR_NULL;
    }

    // vector interface.. when mutable, make sure cache is set to false.
    // only allow const access via begin, end, rbegin, rend
    size_type size() const { return histories_.size(); }
    const_iterator begin() const { return histories_.begin(); }
    const_iterator end() const { return histories_.end(); }
    const_reverse_iterator rbegin() const { return histories_.rbegin(); }
    const_reverse_iterator rend() const { return histories_.rend(); }
    // here is the proper mutable interface... this is done so that the cache is
    // set by us, not the user
    void push_back(const value_type& v) {
      cached_ = false;
      histories_.push_back(v);
    }
    void resize(size_t n) {
      cached_ = false;
      histories_.resize(n);
    }
    void clear() {
      cached_ = false;
      histories_.clear();
      nb_ = nc_ = 0;
      dR_ = 0.0;
      highestFlavor_ = 0;
      flavorSource_ = FlavorHistory::FLAVOR_NULL;
    }

  protected:
    collection_type histories_;   // FlavorHistory vector
    bool cached_;                 // cached flag
    unsigned int nb_;             // number of b quark partons with a matched jet
    unsigned int nc_;             // number of c quark partons with a matched jet
    double dR_;                   // maximum delta R between highest flavor constituents
    unsigned int highestFlavor_;  // highest flavor, corresponds to dR
    flavor_type flavorSource_;    // flavor source
  };

}  // namespace reco

#endif
