// -*- C++ -*-
#ifndef Fireworks_Core_CmsShowSearchFiles_h
#define Fireworks_Core_CmsShowSearchFiles_h
//
// Package:     Core
// Class  :     CmsShowSearchFiles
//
/**\class CmsShowSearchFiles CmsShowSearchFiles.h Fireworks/Core/interface/CmsShowSearchFiles.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Fri Jun 27 11:23:31 EDT 2008
// $Id: CmsShowSearchFiles.h,v 1.4 2009/01/23 21:35:40 amraktad Exp $
//

// system include files
#include "GuiTypes.h"
#include "TGFrame.h"
#include "TGHtmlBrowser.h"
#include <string>

// forward declarations

class TGHtmlBrowser;

class CmsShowSearchFiles : public TGTransientFrame {

public:

  CmsShowSearchFiles (const char *filename,
                     const std::string &windowname, const TGWindow* p = 0,
                     UInt_t w = 1, UInt_t h = 1);
   virtual ~CmsShowSearchFiles();

  TGHtmlBrowser *browser () { return m_searchFileHtml; }

protected:
 
  TGHtmlBrowser  *m_searchFileHtml;
};


#endif
