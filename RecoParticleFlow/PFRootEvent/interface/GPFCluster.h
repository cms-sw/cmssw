#ifndef Graphic_PFCluster_h
#define Graphic_PFCluster_h

/*! \file interface/GPFCluster.h
    class to create graphic object
    from physical object PFCluster
*/  
 
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "TMarker.h"


class GPFCluster : public TMarker {
  public:
    GPFCluster() ;
    GPFCluster(const reco::PFCluster* clus,
	       double x,double y,int color);
    virtual ~GPFCluster() {;}
    
    //override ROOT method 
    virtual void     Print();     // *MENU*
    virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
    
    const GPFCluster& operator=( const GPFCluster& other ) {
      clus_ = other.clus_;
      return *this;
    }

  private:
    const reco::PFCluster*   clus_;
    
};  
#endif
