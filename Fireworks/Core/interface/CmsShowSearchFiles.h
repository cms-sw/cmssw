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
// $Id: CmsShowSearchFiles.h,v 1.6 2012/03/02 03:18:25 amraktad Exp $
//

// system include files
#include "GuiTypes.h"
#include "TGFrame.h"
#include <string>
#include <vector>

// forward declarations

class FWHtml;
class TGComboBox;
class TGTextButton;
class TGPopupMenu;
class TGTextEntry;

class CmsShowSearchFiles : public TGTransientFrame {

public:

   CmsShowSearchFiles (const char *filename,
                       const char* windowname, const TGWindow* p = 0,
                       UInt_t w = 1, UInt_t h = 1);
   virtual ~CmsShowSearchFiles();

   ///This opens the dialog window and returns once the user has choosen, returns an empty string if canceled
   std::string chooseFileFromURL();
   
   //NOTE: Do not call any of the following, they are only public because 'signals' are attached to them
   void showPrefixes();
   void prefixChoosen(Int_t);
   void fileEntryChanged(const char*);
   void updateBrowser();
   void openClicked();
   
   void hyperlinkClicked(const char*);

   ClassDef(CmsShowSearchFiles, 0);

private:
   void sendToWebBrowser(std::string& iWebFile);
   void readInfo();
   void readError();

   TGTextButton* m_choosePrefix;
   TGPopupMenu* m_prefixMenu;
   TGTextEntry* m_file; 
   FWHtml  *m_webFile;
   std::vector<std::string> m_prefixes;
   std::vector<bool> m_prefixComplete;
   TGTextButton* m_openButton;
   bool m_openCalled;
};


#endif
