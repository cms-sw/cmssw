#ifndef ROOT_TEveEllipsoidGL
#define ROOT_TEveEllipsoidGL

#include "TGLObject.h"
#include "TEveVector.h"

class TGLViewer;
class TGLScene;

class TEveEllipsoid;
class TEveEllipsoidProjected;

//------------------------------------------------------------------------------
// TEveEllipsoid
//------------------------------------------------------------------------------

class TEveEllipsoidGL : public TGLObject
{
private:
   TEveEllipsoidGL(const TEveEllipsoidGL&);            // Not implemented
   TEveEllipsoidGL& operator=(const TEveEllipsoidGL&); // Not implemented
 
   
protected:
   TEveEllipsoid                   *fE;  // Model object.

public:
   TEveEllipsoidGL();
   virtual ~TEveEllipsoidGL() {}

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0);
   virtual void   SetBBox();

   virtual void   DirectDraw(TGLRnrCtx & rnrCtx) const;
virtual Bool_t IgnoreSizeForOfInterest() const { return kTRUE; }

  ClassDef(TEveEllipsoidGL, 0); // GL renderer class for TEveEllipsoid.
};


//------------------------------------------------------------------------------
// TEveEllipsoidProjectedGL
//------------------------------------------------------------------------------

class TEveEllipsoidProjectedGL : public TEveEllipsoidGL
{
private:
   TEveEllipsoidProjectedGL(const TEveEllipsoidProjectedGL&);            // Not implemented
   TEveEllipsoidProjectedGL& operator=(const TEveEllipsoidProjectedGL&); // Not implemented
   
   void DrawRhoPhi() const;
   void DrawRhoZ() const;
   //  void DrawYZ() const;
   void   drawArch(float pStart, float pEnd, float phiStep, TEveVector& v0,  TEveVector& v1, TEveVector& v2) const;
   void drawRhoZAxis(TEveVector& v, TEveVector&) const;
protected:
   TEveEllipsoidProjected  *fM;  // Model object.

public:
   TEveEllipsoidProjectedGL();
   virtual ~TEveEllipsoidProjectedGL() {}

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0);
   virtual void   SetBBox();

   virtual void   DirectDraw(TGLRnrCtx & rnrCtx) const;
   ClassDef(TEveEllipsoidProjectedGL, 0); // GL renderer class for TEveEllipsoid.
};

#endif
