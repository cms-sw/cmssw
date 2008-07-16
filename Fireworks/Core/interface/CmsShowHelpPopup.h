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
// $Id: CmsShowHelpPopup.h,v 1.6 2008/07/08 17:47:00 chrjones Exp $
//

// system include files
#include "GuiTypes.h"
#include "TGFrame.h"

// forward declarations
class TGHtml;

class CmsShowHelpPopup : public TGTransientFrame {
public:
     CmsShowHelpPopup(const TGWindow* p = 0, UInt_t w = 1, UInt_t h = 1);
     virtual ~CmsShowHelpPopup();

     static const char *helpFileName ();

protected:
     TGHtml	*m_helpHtml;
};


#endif
