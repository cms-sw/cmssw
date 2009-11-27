#include <stdexcept>
#include "TGClient.h"
#include "TGHtml.h"
#include "TGHtmlBrowser.h"
#include "TGText.h"
#include "TSystem.h"
#include "Fireworks/Core/interface/CmsShowSearchFiles.h"



CmsShowSearchFiles::CmsShowSearchFiles (const char *filename,
                                    const std::string &windowname,
                                    const TGWindow* p, UInt_t w, UInt_t h)
   : TGTransientFrame(gClient->GetDefaultRoot(), p, w, h),
     m_searchFileHtml(new TGHtmlBrowser(filename,this,w, h))
{
   AddFrame(m_searchFileHtml, new TGLayoutHints(kLHintsTop | kLHintsLeft |
                                          kLHintsExpandX | kLHintsExpandY));
   SetWindowName(windowname.c_str());
   MapSubwindows();
   m_searchFileHtml->Layout();
}


CmsShowSearchFiles::~CmsShowSearchFiles()
{
   delete m_searchFileHtml;
}

