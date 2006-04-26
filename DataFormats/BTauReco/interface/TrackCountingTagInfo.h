#ifndef BTauReco_BJetTagTrackCounting_h
#define BTauReco_BJetTagTrackCounting_h


#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfoFwd.h"

namespace reco {
 
class TrackCountingTagInfo
 {
  public:

  TrackCountingTagInfo() {}
  virtual ~TrackCountingTagInfo() {}
  
  virtual float significance(size_t n) const 
   {
    if(n <m_significance.size())
      return m_significance[n];  
    return -10.; 
   }
 
  virtual float discriminator(size_t n) const { return significance(n); }
  
  virtual TrackCountingTagInfo* clone() const { return new TrackCountingTagInfo( * this ); }
  

 
  private:
   edm::Ref<JetTagCollection> m_jetTag; 
   std::vector<double> m_significance;  //create a smarter container instead of 
   std::vector<int> m_trackOrder;       // this  pair of vectors. 
 };

//typedef edm::ExtCollection< TrackCountingTagInfo,JetTagCollection> TrackCountingExtCollection;
//typedef edm::OneToOneAssociation<JetTagCollection, TrackCountingTagInfo> TrackCountingExtCollection;
 
}
#endif
