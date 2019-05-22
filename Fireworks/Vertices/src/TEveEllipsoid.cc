#include "Fireworks/Vertices/interface/TEveEllipsoid.h"
#include "TEveTrans.h"
#include "TEveProjectionManager.h"
#include "TMath.h"

//______________________________________________________________________________
TEveEllipsoid::TEveEllipsoid(const Text_t* n, const Text_t* t) : TEveShape(n, t) {
  // Constructor.
}

//______________________________________________________________________________
void TEveEllipsoid::ComputeBBox() {
  // Compute bounding-box of the data.

  BBoxInit();

  Float_t a = TMath::Max(TMath::Max(TMath::Abs(fExtent3D[0]), TMath::Abs(fExtent3D[1])), TMath::Abs(fExtent3D[2]));

  fBBox[0] = -a + fPos[0];
  fBBox[1] = a + fPos[0];

  fBBox[2] = -a + fPos[1];
  fBBox[3] = a + fPos[1];

  fBBox[4] = -a + fPos[2];
  fBBox[5] = a + fPos[2];
}

//______________________________________________________________________________
TClass* TEveEllipsoid::ProjectedClass(const TEveProjection*) const {
  // Virtual from TEveProjectable, returns TEveEllipsoidProjected class.

  return TEveEllipsoidProjected::Class();
}

//==============================================================================
// TEveEllipsoidProjected
//==============================================================================

//______________________________________________________________________________
//
// Projection of TEveEllipsoid.

//______________________________________________________________________________
TEveEllipsoidProjected::TEveEllipsoidProjected(const char* n, const char* t) : TEveShape(n, t) {
  // Constructor.
}

//______________________________________________________________________________
TEveEllipsoidProjected::~TEveEllipsoidProjected() {
  // Destructor.
}

//______________________________________________________________________________
void TEveEllipsoidProjected::ComputeBBox() {
  // Compute bounding-box, virtual from TAttBBox.

  BBoxInit();

  TEveEllipsoid* e3d = dynamic_cast<TEveEllipsoid*>(fProjectable);

  //printf("project bbox  %p\n", (void*)e3d->GetBBox());
  if (e3d) {
    TEveProjection* proj = GetManager()->GetProjection();
    Float_t a = TMath::Max(TMath::Max(TMath::Abs(e3d->RefExtent3D()[0]), TMath::Abs(e3d->RefExtent3D()[1])),
                           TMath::Abs(e3d->RefExtent3D()[2]));
    float* p = e3d->RefPos().Arr();
    float b[] = {-a + p[0], a + p[0], -a + p[1], a + p[1], -a + p[2], a + p[2]};
    TEveVector v;
    v.Set(b[0], b[2], b[4]);
    proj->ProjectVector(v, fDepth);
    BBoxCheckPoint(v);
    v.Set(b[1], b[2], b[4]);
    proj->ProjectVector(v, fDepth);
    BBoxCheckPoint(v);
    v.Set(b[0], b[3], b[4]);
    proj->ProjectVector(v, fDepth);
    BBoxCheckPoint(v);
    v.Set(b[1], b[3], b[4]);
    proj->ProjectVector(v, fDepth);
    BBoxCheckPoint(v);
    v.Set(b[0], b[2], b[5]);
    proj->ProjectVector(v, fDepth);
    BBoxCheckPoint(v);
    v.Set(b[1], b[2], b[5]);
    proj->ProjectVector(v, fDepth);
    BBoxCheckPoint(v);
    v.Set(b[0], b[3], b[5]);
    proj->ProjectVector(v, fDepth);
    BBoxCheckPoint(v);
    v.Set(b[1], b[3], b[5]);
    proj->ProjectVector(v, fDepth);
    BBoxCheckPoint(v);

    // for Z dimesion
    fBBox[4] -= a;
    fBBox[5] += a;
  }
  // printf("(%f, %f) (%f, %f) (%f, %f)\n",fBBox[0],fBBox[1],fBBox[2],fBBox[3],fBBox[4],fBBox[5] );
}

//______________________________________________________________________________
void TEveEllipsoidProjected::SetDepthLocal(Float_t d) {
  // This is virtual method from base-class TEveProjected.

  SetDepthCommon(d, this, fBBox);
}

//______________________________________________________________________________
void TEveEllipsoidProjected::SetProjection(TEveProjectionManager* mng, TEveProjectable* model) {
  // This is virtual method from base-class TEveProjected.

  TEveProjected::SetProjection(mng, model);
  CopyVizParams(dynamic_cast<TEveElement*>(model));
}

//______________________________________________________________________________
void TEveEllipsoidProjected::UpdateProjection() {
  // Override from abstract function.
}
