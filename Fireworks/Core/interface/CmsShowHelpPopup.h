// -*- C++ -*-
#ifndef Fireworks_Core_CmsShowHelpPopup_h
#define Fireworks_Core_CmsShowHelpPopup_h
//
// Package:     Core
// Class  :     CmsShowHelpPopup
//
/**\class CmsShowHelpPopup CmsShowHelpPopup.h Fireworks/Core/interface/CmsShowHelpPopup.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Fri Jun 27 11:23:31 EDT 2008
//

// system include files
#include "GuiTypes.h"
#include "TGFrame.h"
#include <string>

// forward declarations
class TGHtml;

class CmsShowHelpPopup : public TGTransientFrame {
public:
   CmsShowHelpPopup (const std::string &filename,
                     const std::string &windowname, const TGWindow* p = nullptr,
                     UInt_t w = 1, UInt_t h = 1);
   ~CmsShowHelpPopup() override;
   void CloseWindow() override { UnmapWindow(); }

 
protected:
   TGHtml     *m_helpHtml;
};


#endif
