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
// $Id: CmsShowHelpPopup.h,v 1.5 2009/05/20 16:33:39 amraktad Exp $
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
                     const std::string &windowname, const TGWindow* p = 0,
                     UInt_t w = 1, UInt_t h = 1);
   virtual ~CmsShowHelpPopup();
   virtual void CloseWindow() { UnmapWindow(); }

 
protected:
   TGHtml     *m_helpHtml;
};


#endif
