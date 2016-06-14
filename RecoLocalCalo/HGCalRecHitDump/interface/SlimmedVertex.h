#ifndef slimmedvertex_h
#define slimmedvertex_h

#include "TObject.h"

/**
   @short use to strip down the information of a vertex
 */
class SlimmedVertex : public TObject
{
 public:
  SlimmedVertex(){ }
 SlimmedVertex(int ntks,float x, float y, float z, float pt, float normchi2) :
  ntks_(ntks), x_(x), y_(y), z_(z), pt_(pt), normchi2_(normchi2){}
  SlimmedVertex(const SlimmedVertex &other)
    {
      ntks_     = other.ntks_;
      x_        = other.x_;
      y_        = other.y_;
      z_        = other.z_;      
      pt_       = other.pt_;   
      normchi2_ = other.normchi2_;
    }
  virtual ~SlimmedVertex() { }

  int ntks_;
  float x_,y_,z_,pt_,normchi2_;

  ClassDef(SlimmedVertex,1)
};

#endif
