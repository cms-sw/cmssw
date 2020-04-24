#ifndef Fireworks_Core_FWTextProjected_h
#define Fireworks_Core_FWTextProjected_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWTextProjected
// 
/**\class FWTextProjected FWTextProjected.h Fireworks/Core/interface/FWTextProjected.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel 
//         Created:  Fri Aug 12 01:12:28 CEST 2011
//

#include "TNamed.h"
#include "TAtt3D.h"
#include "TAttBBox.h"

#include "TEveProjectionBases.h"
#include "TEveText.h"
#include "TEveTextGL.h"


class FWEveText : public TEveText,
                  public TEveProjectable
{
private:
   FWEveText(const FWEveText&); // stop default
   const FWEveText& operator=(const FWEveText&); // stop default

public:
   float m_offsetZ;
   float m_textPad;
   FWEveText(const char* txt=""):TEveText(txt), m_offsetZ(0), m_textPad(5) {}
   ~FWEveText() override {}

   TClass* ProjectedClass(const TEveProjection* p) const override  ;
   ClassDefOverride(FWEveText, 0); // Class for visualisation of text with FTGL font.
};

//==============================================================================
//==============================================================================


class FWEveTextProjected : public FWEveText,
                           public TEveProjected
{
private:
   FWEveTextProjected(const FWEveTextProjected&);            // Not implemented
   FWEveTextProjected& operator=(const FWEveTextProjected&); // Not implemented

public:
   FWEveTextProjected() {}
   ~FWEveTextProjected() override {}

   void UpdateProjection() override;
   TEveElement* GetProjectedAsElement() override { return this; }

   ClassDefOverride(FWEveTextProjected, 0); // Projected replica of a FWEveText.
};

//______________________________________________________________________________

class FWEveTextGL : public TEveTextGL
{
public:
   FWEveTextGL(){}
   ~FWEveTextGL() override {}

   void DirectDraw(TGLRnrCtx & rnrCtx) const override;

   ClassDefOverride(FWEveTextGL, 0); // GL renderer class for TEveText.
};


#endif
