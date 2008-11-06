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

namespace reco {


class FlavorHistoryEvent {
public:

  // convenient typedefs
  typedef FlavorHistory                           value_type;
  typedef std::vector<value_type>                 collection_type;
  typedef collection_type::size_type              size_type;
  typedef collection_type::iterator               iterator;
  typedef collection_type::const_iterator         const_iterator;
  typedef collection_type::reverse_iterator       reverse_iterator;
  typedef collection_type::const_reverse_iterator const_reverse_iterator;
  typedef collection_type::pointer                pointer;
  typedef collection_type::const_pointer          const_pointer;
  typedef collection_type::reference              reference;
  typedef collection_type::const_reference        const_reference;


  FlavorHistoryEvent() { clear(); }
  ~FlavorHistoryEvent(){}

  // Set up the heavy flavor content
  void                 cache();
  bool                 isCached() const { return cached_; }

  // Accessors to heavy flavor content
  unsigned int         nb() const { if ( isCached() ) return nb_; else return calculateNB();}
  unsigned int         nc() const { if ( isCached() ) return nc_; else return calculateNC();}

  // Accessor to maximum delta R between highest flavor constituents
  double               deltaR() const { if ( isCached() ) return dR_; else return calculateDR().first; }
  unsigned int         highestFlavor() const { if ( isCached() ) return highestFlavor_; else return calculateDR().second; }

  // vector interface.. when mutable, make sure cache is set to false.
  // only allow const access via begin, end, rbegin, rend
  size_type               size()   const { return histories_.size(); }
  const_iterator          begin()  const { return histories_.begin(); }
  const_iterator          end()    const { return histories_.end(); }
  const_reverse_iterator  rbegin() const { return histories_.rbegin(); }
  const_reverse_iterator  rend()   const { return histories_.rend(); }
  // here is the proper mutable interface... this is done so that the cache is
  // set by us, not the user
  void                    push_back( value_type v ) { cached_ = false; histories_.push_back(v); }
  void                    resize( size_t n )        { cached_ = false; histories_.resize(n); }
  void                    clear() { cached_ = false; histories_.clear(); nb_ = nc_ = 0; dR_ = 0.0; highestFlavor_ = 0; }

protected:
  unsigned int            calculateNB() const;  // if object can't be cached (i.e. is const), return
  unsigned int            calculateNC() const;  //   count of b and c partons without setting nb and nc on the fly
  std::pair<double,int>   calculateDR() const;  // Calculate the maximum delta R (first) of the highest flavor q-qbar pair (second)
  collection_type         histories_;           // FlavorHistory vector
  bool                    cached_;              // cached flag
  unsigned int            nb_;                  // number of b quark partons with a matched jet
  unsigned int            nc_;                  // number of c quark partons with a matched jet
  double                  dR_;                  // maximum delta R between highest flavor constituents
  unsigned int            highestFlavor_;       // highest flavor, corresponds to dR
};

}

#endif
