#ifndef BTauReco_BJetTagTrackProbability_h
#define BTauReco_BJetTagTrackProbability_h


#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfoFwd.h"

//FIXME: check what to use
#define MAX(a, b) (a > b ? a : b)
#define MIN(a, b) (a < b ? a : b)

//FIXME: fix this

namespace reco {
 
class TrackProbabilityTagInfo
 {
  public:

 TrackProbabilityTagInfo(
   std::vector<double> probability2d,
   std::vector<double> probability3d,
   std::vector<int> trackOrder2d,
   std::vector<int> trackOrder3d) :
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

  virtual float jetProbability(int ip) const
  {
   const std::vector<double> * vp;
   if(ip==0) vp= &m_probability3d;
   else vp= &m_probability2d;
   const std::vector<double> & v =*vp;

   int ngoodtracks=v.size();
   double SumJet=0.;

  for(std::vector<double>::const_iterator q = v.begin(); q != v.end(); q++){
    SumJet+=(log(*q)>-20.)?log(*q):-20;
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
      MIN(exp(MAX(LogProb+SumJet,-30.)),1.);
  }else{
    ProbJet=1.;
  }
  if(ProbJet>1)
   std::cout << "ProbJet too high: "  << ProbJet << std::endl;

  double LogProbJet=-log(ProbJet);

  return LogProbJet;
  }

 /**
  Recompute discriminator 
  ipType = 0 means 3d impact parameter
  ipType = 1 means transverse impact parameter
 */
  virtual float discriminator(int ipType) const { return jetProbability(ipType); }

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

    return *m_jetTag->tracks()[trackIndex(n,ipType)];
  }
 
  virtual TrackProbabilityTagInfo* clone() const { return new TrackProbabilityTagInfo( * this ); }
  
  void setJetTag(const JetTagRef ref) { 
        m_jetTag = ref;
   }
 
  private:
   edm::Ref<JetTagCollection> m_jetTag; 
   std::vector<double> m_probability2d;  //create a smarter container instead of 
   std::vector<double> m_probability3d;  //create a smarter container instead of 
   std::vector<int> m_trackOrder2d;       // this  pair of vectors. 
   std::vector<int> m_trackOrder3d;       // this  pair of vectors. 
 };

//typedef edm::ExtCollection< TrackProbabilityTagInfo,JetTagCollection> TrackProbabilityExtCollection;
//typedef edm::OneToOneAssociation<JetTagCollection, TrackProbabilityTagInfo> TrackProbabilityExtCollection;
 
}
#endif
