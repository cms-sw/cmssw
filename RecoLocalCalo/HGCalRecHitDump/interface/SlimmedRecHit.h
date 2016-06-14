#ifndef slimmedrechit_h
#define slimmedrechit_h

#include "TObject.h"

#include<iostream>
#include <vector>

/**
   @short use to strip down the information of a RecHit
 */
class SlimmedRecHit : public TObject
{
 public:
 SlimmedRecHit() : detId_(0), layerId_(0), clustId_(-1), isIn3x3_(false), isIn5x5_(false), isIn7x7_(false), x_(0), y_(0), z_(0), en_(0), t_(0), cellSize_(1.0) {}
 SlimmedRecHit(int detId, float x=0, float y=0, float z=0, float en=0, float t=0, float cellSize=1) : detId_(detId),
    clustId_(-1), isIn3x3_(false), isIn5x5_(false), isIn7x7_(false),
    x_(x), y_(y), z_(z), en_(en), t_(t), cellSize_(cellSize) 
  { 
    layerId_=(detId>>19)&0x1f;
  }
  SlimmedRecHit(const SlimmedRecHit &other)
    {
      detId_    = other.detId_;
      layerId_  = other.layerId_;
      clustId_  = other.clustId_;
      isIn3x3_  = other.isIn3x3_;
      isIn5x5_  = other.isIn5x5_;
      isIn7x7_  = other.isIn7x7_;
      x_        = other.x_;
      y_        = other.y_;
      z_        = other.z_;
      en_       = other.en_;
      t_        = other.t_;
      simEn_    = other.simEn_;
      simT_     = other.simT_;
      cellSize_ = other.cellSize_;
    }

  friend bool operator== ( const SlimmedRecHit &lhs, const SlimmedRecHit &rhs) { return lhs.detId_==rhs.detId_; }
  static bool IsNotClustered(const SlimmedRecHit &hit) { return hit.clustId_<0; }

  unsigned int detId() { return detId_; }
  float x()  { return x_; }
  float y()  { return y_; }
  float z()  { return z_; }
  float en() { return en_; }
  float t()  { return t_; } 

  void addSimHit(float en, float toa,float emf) 
  { 
    simEn_.push_back(en);
    simT_.push_back(toa);
    simEmFrac_.push_back(emf);
  }

  virtual ~SlimmedRecHit() { }
  
  unsigned int detId_,layerId_;
  int clustId_;
  bool isIn3x3_, isIn5x5_, isIn7x7_;
  float x_,y_,z_,en_,t_;
  float cellSize_;
  std::vector<float> simEn_, simT_, simEmFrac_;

  ClassDef(SlimmedRecHit,1)
};

typedef std::vector<SlimmedRecHit> SlimmedRecHitCollection;

#endif
