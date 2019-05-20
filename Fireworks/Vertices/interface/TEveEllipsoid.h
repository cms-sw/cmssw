#ifndef ROOT_TEveEllipsoid
#define ROOT_TEveEllipsoid

#include "TEveShape.h"
#include "TEveVector.h"
#include "TEveTrans.h"
#include "TMatrixDSym.h"

//------------------------------------------------------------------------------
// TEveEllipsoid
//------------------------------------------------------------------------------

class TEveEllipsoid : public TEveShape {
  friend class TEveEllipsoidProjected;
  friend class TEveEllipsoidGL;
  friend class TEveEllipsoidProjectedGL;

private:
  TEveEllipsoid(const TEveEllipsoid&);             // Not implemented
  TEveEllipsoid& operator=(const TEveEllipsoid&);  // Not implemented

protected:
  TEveVector fPos;
  TEveVector fExtent3D;
  TEveTrans fEMtx;

  float fEScale;

public:
  TEveEllipsoid(const Text_t* n = "TEveEllipsoid", const Text_t* t = "");
  ~TEveEllipsoid() override {}

  void ComputeBBox() override;
  TClass* ProjectedClass(const TEveProjection* p) const override;

  TEveVector& RefPos() { return fPos; }
  TEveVector& RefExtent3D() { return fExtent3D; }  // cached member for bbox and 3D rendering
  TEveTrans& RefEMtx() { return fEMtx; }

  void SetScale(float x) { fEScale = x; }

  ClassDefOverride(TEveEllipsoid, 0);  // Short description.
};

//------------------------------------------------------------------------------
// TEveEllipsoidProjected
//------------------------------------------------------------------------------

class TEveEllipsoidProjected : public TEveShape, public TEveProjected {
  friend class TEveEllipsoidProjectedGL;

private:
  TEveEllipsoidProjected(const TEveEllipsoidProjected&);             // Not implemented
  TEveEllipsoidProjected& operator=(const TEveEllipsoidProjected&);  // Not implemented

protected:
  void SetDepthLocal(Float_t d) override;

public:
  TEveEllipsoidProjected(const char* n = "TEveEllipsoidProjected", const char* t = "");
  ~TEveEllipsoidProjected() override;

  // For TAttBBox:
  void ComputeBBox() override;

  // Projected:
  void SetProjection(TEveProjectionManager* mng, TEveProjectable* model) override;
  void UpdateProjection() override;

  TEveElement* GetProjectedAsElement() override { return this; }

  ClassDefOverride(TEveEllipsoidProjected, 0);  // Projection of TEveEllipsoid.
};

#endif
