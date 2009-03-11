#ifndef Fireworks_Core_FW3DLegoView_h
#define Fireworks_Core_FW3DLegoView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DLegoView
//
/**\class FW3DLegoView FW3DLegoView.h Fireworks/Core/interface/FW3DLegoView.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:37 EST 2008
// $Id: FW3DLegoView.h,v 1.6 2009/01/23 21:35:40 amraktad Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWViewBase.h"

// forward declarations
class TGFrame;
class TRootEmbeddedCanvas;
class THStack;
class TCanvas;
class TObject;

class FW3DLegoView : public FWViewBase
{

public:
   FW3DLegoView(TEveWindowSlot*);
   virtual ~FW3DLegoView();

   // ---------- const member functions ---------------------
   TGFrame* frame() const;
   const std::string& typeName() const;

   // ---------- static member functions --------------------
   static const std::string& staticTypeName();

   // ---------- member functions ---------------------------
   void draw(THStack*);

   void DynamicCoordinates();
   void connect(const char* receiver_class, void* receiver, const char* slot);

   virtual void saveImageTo(const std::string& iName) const;

private:
   FW3DLegoView(const FW3DLegoView&);    // stop default

   const FW3DLegoView& operator=(const FW3DLegoView&);    // stop default

   // ---------- member data --------------------------------
   TRootEmbeddedCanvas* m_frame;
   TCanvas* m_legoCanvas;
};


#endif
