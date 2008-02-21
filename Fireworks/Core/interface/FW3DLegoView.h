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
// $Id: FW3DLegoView.h,v 1.1 2008/02/21 19:20:05 chrjones Exp $
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
      FW3DLegoView(TGFrame*);
      virtual ~FW3DLegoView();

      // ---------- const member functions ---------------------
      TGFrame* frame() const;
      const std::string& typeName() const;
     
      // ---------- static member functions --------------------
      static const std::string& staticTypeName();
   
      // ---------- member functions ---------------------------
      void draw(THStack*);

      void DynamicCoordinates();
      void exec3event(int event, int x, int y, TObject *selected);
      void pixel2wc(const Int_t PixelX, const Int_t PixelY, 
                    Double_t& WCX, Double_t& WCY, const Double_t WCZ = 0);
   
   private:
      FW3DLegoView(const FW3DLegoView&); // stop default

      const FW3DLegoView& operator=(const FW3DLegoView&); // stop default

      // ---------- member data --------------------------------
      TRootEmbeddedCanvas* m_frame;
      TCanvas* m_legoCanvas;
};


#endif
