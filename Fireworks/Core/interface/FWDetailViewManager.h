#ifndef Fireworks_Core_FWDetailViewManager_h
#define Fireworks_Core_FWDetailViewManager_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDetailViewManager
//
/**\class FWDetailViewManager FWDetailViewManager.h Fireworks/Core/interface/FWDetailViewManager.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Wed Mar  5 09:13:43 EST 2008
// $Id: FWDetailViewManager.h,v 1.19 2009/09/23 20:26:27 chrjones Exp $
//
#include <map>
#include <string>

class FWColorManager;
class TEveCompositeFrameInMainFrame;
class FWDetailViewBase;
class FWModelId;
class TEveWindow;

class FWDetailViewManager
{
public:
   FWDetailViewManager(FWColorManager*);
   virtual ~FWDetailViewManager();
   bool haveDetailViewFor(const FWModelId&) const;
   void openDetailViewFor(const FWModelId& );
   void colorsChanged();

private:
   FWDetailViewManager(const FWDetailViewManager&);    // stop default
   const FWDetailViewManager& operator=(const FWDetailViewManager&);    // stop default
   std::string findViewerFor(const std::string&) const;

protected:
   FWColorManager                *m_colorManager;

   TGMainFrame                   *m_mainFrame;
   TEveCompositeFrameInMainFrame *m_eveFrame; // cached
   FWDetailViewBase              *m_detailView;

   std::map<std::string, FWDetailViewBase *>  m_detailViews;
   mutable std::map<std::string, std::string> m_typeToViewers;
};
#endif
