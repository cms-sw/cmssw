#ifndef DataFormats_BTauReco_BJetTagTrackCounting_h
#define DataFormats_BTauReco_BJetTagTrackCounting_h

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

#include "DataFormats/BTauReco/interface/JTATagInfo.h"

namespace reco {
 
class TrackCountingTagInfo : public JTATagInfo
 {
  public:

 TrackCountingTagInfo(
   const std::vector<double>& significance2d,
   const std::vector<double>& significance3d,
   const std::vector<int>& trackOrder2d,
   const std::vector<int>& trackOrder3d,const JetTracksAssociationRef & jtaRef) : JTATagInfo(jtaRef),
     m_significance2d(significance2d),
     m_significance3d(significance3d),
     m_trackOrder2d(trackOrder2d),
     m_trackOrder3d(trackOrder3d)     {}

  TrackCountingTagInfo() {}
  
  virtual ~TrackCountingTagInfo() {}
  
  /* virtual const Track & track(size_t n,int ipType) const
  {
    return tracks()[trackIndex(n,ipType)];
  } */

  virtual float significance(size_t n,int ip) const 
   {
    if(ip == 0)
    {
     if(n <m_significance3d.size())
      return m_significance3d[n];  
    }
    else
    {
     if(n <m_significance2d.size())
      return m_significance2d[n];  
    }
    return -10.; 
   }

   virtual int trackIndex(size_t n,int ip) const
   {
    if(ip == 0)
    {
     if(n <m_significance3d.size())
      return m_trackOrder3d[n];
    }
    else
    {
     if(n <m_significance2d.size())
      return m_trackOrder2d[n];
    }
    return 0;
   }

   
 /**
  Recompute discriminator using nth track i.p. significance.
  ipType = 0 means 3d impact parameter
  ipType = 1 means transverse impact parameter
 */
  virtual float discriminator(size_t nth, int ipType) const { return significance(nth-1,ipType); }
 
  virtual int selectedTracks(int ipType) const
  {
   if(ipType == 0) return m_significance3d.size();
   else return m_significance2d.size();
  }   
  
  virtual TrackCountingTagInfo* clone() const { return new TrackCountingTagInfo( * this ); }
 
  private:
   std::vector<double> m_significance2d;  //create a smarter container instead of 
   std::vector<double> m_significance3d;  //create a smarter container instead of 
   std::vector<int> m_trackOrder2d;       // this  pair of vectors. 
   std::vector<int> m_trackOrder3d;       // this  pair of vectors. 
 };

//typedef edm::ExtCollection< TrackCountingTagInfo,JetTagCollection> TrackCountingExtCollection;
//typedef edm::OneToOneAssociation<JetTagCollection, TrackCountingTagInfo> TrackCountingExtCollection;
 
DECLARE_EDM_REFS( TrackCountingTagInfo )

}

#endif // DataFormats_BTauReco_BJetTagTrackCounting_h
