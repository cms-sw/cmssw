#ifndef slimmedcluster_h
#define slimmedcluster_h

#include "TObject.h"

/**
   @short use to strip down the information of a vertex
 */
class SlimmedCluster : public TObject
{
 public:
 SlimmedCluster() : en_(0), eta_(0), phi_(0), nhits_(0){ }
 SlimmedCluster(float en, float eta, float phi, int nhits):en_(en), eta_(eta), phi_(phi), nhits_(nhits) {}
  SlimmedCluster(const SlimmedCluster &other)
    {
      en_=other.en_;
      eta_=other.eta_;
      phi_=other.phi_;
      nhits_=other.nhits_;
      roiidx_=other.roiidx_;
      center_x_=other.center_x_;
      center_y_=other.center_y_; 
      center_z_=other.center_z_;
      axis_x_=other.axis_x_;
      axis_y_=other.axis_y_;   
      axis_z_=other.axis_z_;
      ev_1_=other.ev_1_;    
      ev_2_=other.ev_2_;     
      ev_3_=other.ev_3_;
      sigma_1_=other.sigma_1_;  
      sigma_2_=other.sigma_2_;  
      sigma_3_=other.sigma_3_;
    }
  virtual ~SlimmedCluster() { }

  float en_, eta_, phi_;
  float center_x_, center_y_, center_z_;
  float axis_x_,   axis_y_,   axis_z_;
  float ev_1_,     ev_2_,     ev_3_;
  float sigma_1_,  sigma_2_,  sigma_3_;
  int nhits_,roiidx_;

  ClassDef(SlimmedCluster,1)
};

typedef std::vector<SlimmedCluster> SlimmedClusterCollection;

#endif
