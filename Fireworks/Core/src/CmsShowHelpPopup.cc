#include <stdexcept>
#include <cassert>
#include "TGClient.h"
#include "TGHtml.h"
#include "TGText.h"
#include "TSystem.h"
#include "Fireworks/Core/interface/CmsShowHelpPopup.h"
#include "Fireworks/Core/interface/fwPaths.h"

CmsShowHelpPopup::CmsShowHelpPopup (const std::string &filename,
                                    const std::string &windowname,
                                    const TGWindow* p, UInt_t w, UInt_t h)
   : TGTransientFrame(gClient->GetDefaultRoot(), p, w, h),
     m_helpHtml(new TGHtml(this, w, h))
{
   AddFrame(m_helpHtml, new TGLayoutHints(kLHintsTop | kLHintsLeft |
                                          kLHintsExpandX | kLHintsExpandY));
   SetWindowName(windowname.c_str());

   TString path =  "data/" + filename;
   fireworks::setPath(path);
  
   TGText text;
   m_helpHtml->ParseText((char *)text.AsString().Data());

   MapSubwindows();
   m_helpHtml->Layout();
}

CmsShowHelpPopup::~CmsShowHelpPopup()
{
   delete m_helpHtml;
}
