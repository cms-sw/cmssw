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
// $Id: FWTextProjected.h,v 1.2 2011/08/16 21:43:27 amraktad Exp $
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
   virtual ~FWEveText() {}

   virtual TClass* ProjectedClass(const TEveProjection* p) const  ;
   ClassDef(FWEveText, 0); // Class for visualisation of text with FTGL font.
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
   virtual ~FWEveTextProjected() {}

   virtual void UpdateProjection();
   virtual TEveElement* GetProjectedAsElement() { return this; }

   ClassDef(FWEveTextProjected, 0); // Projected replica of a FWEveText.
};

//______________________________________________________________________________

class FWEveTextGL : public TEveTextGL
{
public:
   FWEveTextGL(){}
   virtual ~FWEveTextGL() {}

   virtual void DirectDraw(TGLRnrCtx & rnrCtx) const;

   ClassDef(FWEveTextGL, 0); // GL renderer class for TEveText.
};


#endif
