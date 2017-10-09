#ifndef BTauReco_BJetTagTrackProbability_h
#define BTauReco_BJetTagTrackProbability_h

#include "DataFormats/BTauReco/interface/RefMacros.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

namespace reco {
 
class TrackProbabilityTagInfo : public JTATagInfo
 {
  public:

 TrackProbabilityTagInfo(
   const std::vector<double>& probability2d,
   const std::vector<double>& probability3d,
   const std::vector<int>& trackOrder2d,
   const std::vector<int>& trackOrder3d,const JetTracksAssociationRef & jtaRef) : JTATagInfo(jtaRef),
     m_probability2d(probability2d),
     m_probability3d(probability3d),
     m_trackOrder2d(trackOrder2d),
     m_trackOrder3d(trackOrder3d)     {}

  TrackProbabilityTagInfo() {}
  
  virtual ~TrackProbabilityTagInfo() {}

int factorial(int n) const
{
 if(n<2) return 1;
 else return n*factorial(n-1);
}
  
  virtual float probability(size_t n,int ip) const 
   {
    if(ip == 0)
    {
     if(n <m_probability3d.size())
      return m_probability3d[n];  
    }
    else
    {
     if(n <m_probability2d.size())
      return m_probability2d[n];  
    }
    return -10.; 
   }

  virtual float jetProbability(int ip, float minTrackProb) const
{

 const std::vector<double> * vp;
   if(ip==0) vp= &m_probability3d;
   else vp= &m_probability2d;
   const std::vector<double> & v =*vp;

   int ngoodtracks=v.size();
   double SumJet=0.;

  for(std::vector<double>::const_iterator q = v.begin(); q != v.end(); q++){
    SumJet+=(*q>minTrackProb)?log(*q):log(minTrackProb);
  }

  double ProbJet;
  double Loginvlog=0;

  if(SumJet<0.){
    if(ngoodtracks>=2){
      Loginvlog=log(-SumJet);
    }
    double Prob=1.;
    for(int l=1; l!=ngoodtracks; l++){

      Prob+=exp(l*Loginvlog-log(1.*factorial(l)));
    }
    double LogProb=log(Prob);
    ProbJet=
      std::min(exp(std::max(LogProb+SumJet,-30.)),1.);
  }else{
    ProbJet=1.;
  }
  if(ProbJet>1)
   std::cout << "ProbJet too high: "  << ProbJet << std::endl;

  //double LogProbJet=-log(ProbJet);
  //return 1.-ProbJet;
  return -log10(ProbJet)/4.;
}

   
 /**
  Recompute discriminator 
  ipType = 0 means 3d impact parameter
  ipType = 1 means transverse impact parameter
  
  minProb is the minimum probability allowed  for a single track. Tracks with lower probability 
  are considered with a probability = minProb.
 */
  virtual float discriminator(int ipType, float minProb) const { return jetProbability(ipType,minProb); }

  virtual int selectedTracks(int ipType) const
  {
   if(ipType == 0) return m_probability3d.size();
   else return m_probability2d.size();
  }
     virtual int trackIndex(size_t n,int ip) const
   {
    if(ip == 0)
    {
     if(n <m_probability3d.size())
      return m_trackOrder3d[n];
    }
    else
    {
     if(n <m_probability2d.size())
      return m_trackOrder2d[n];
    }
    return 0;
   }
 
  virtual const Track & track(size_t n,int ipType) const
  {
    return *tracks()[trackIndex(n,ipType)];
  }
 
  virtual TrackProbabilityTagInfo* clone() const { return new TrackProbabilityTagInfo( * this ); }
  
  private:
   std::vector<double> m_probability2d;     //
   std::vector<double> m_probability3d;     // create a smarter container instead of 
   std::vector<int> m_trackOrder2d;         // this pair of vectors. 
   std::vector<int> m_trackOrder3d;         //
 };

//typedef edm::ExtCollection< TrackProbabilityTagInfo,JetTagCollection> TrackProbabilityExtCollection;
//typedef edm::OneToOneAssociation<JetTagCollection, TrackProbabilityTagInfo> TrackProbabilityExtCollection;
 
DECLARE_EDM_REFS( TrackProbabilityTagInfo )

}
#endif
