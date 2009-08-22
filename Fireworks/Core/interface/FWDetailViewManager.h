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
// $Id: FWDetailViewManager.h,v 1.17 2009/07/28 17:25:55 amraktad Exp $
//

class TEveCompositeFrameInMainFrame;
class FWDetailViewBase;
class FWModelId;
class TEveWindow;

class FWDetailViewManager
{
public:
   FWDetailViewManager(const TGWindow*);
   virtual ~FWDetailViewManager();
   bool haveDetailViewFor(const FWModelId&) const;
   void openDetailViewFor(const FWModelId& );

private:
   FWDetailViewManager(const FWDetailViewManager&);    // stop default
   const FWDetailViewManager& operator=(const FWDetailViewManager&);    // stop default
   std::string findViewerFor(const std::string&) const;

protected:
   // const TGWindow* m_cmsShowMainFrame;
   TGMainFrame                   *m_mainFrame;
   TEveCompositeFrameInMainFrame *m_eveFrame; // cached
   FWDetailViewBase              *m_detailView;

   std::map<std::string, FWDetailViewBase *>  m_detailViews;
   mutable std::map<std::string, std::string> m_typeToViewers;
};
#endif
