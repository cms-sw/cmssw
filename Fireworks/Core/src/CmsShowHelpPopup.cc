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
 
   TString dirPath =  "data/";
   fireworks::setPath(dirPath); 
   m_helpHtml->SetBaseUri(dirPath.Data());
   // printf("%s ... %s\n",  m_helpHtml->GetBaseUri(), dirPath.Data());

   TGText text;
   TString filePath =  dirPath + filename;
   text.Load(filePath.Data());

   m_helpHtml->ParseText((char *)text.AsString().Data());

   MapSubwindows();
   m_helpHtml->Layout();
}

CmsShowHelpPopup::~CmsShowHelpPopup()
{
   delete m_helpHtml;
}
