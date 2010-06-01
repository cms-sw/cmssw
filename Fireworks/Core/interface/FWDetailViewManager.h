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
// $Id: FWDetailViewManager.h,v 1.22 2010/05/12 16:35:10 amraktad Exp $
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

   std::vector<std::string> detailViewsFor(const FWModelId&) const;
   void assertMainFrame();
   void openDetailViewFor(const FWModelId&, const std::string&);
   void colorsChanged();
   void newEventCallback();
   void eveWindowDestroyed();
private:
   FWDetailViewManager(const FWDetailViewManager&);    // stop default
   const FWDetailViewManager& operator=(const FWDetailViewManager&);    // stop default
   std::vector<std::string> findViewersFor(const std::string&) const;

protected:
   FWColorManager                *m_colorManager;
   TEveCompositeFrameInMainFrame *m_eveFrame;
   FWDetailViewBase              *m_detailView;
   std::map<std::string, FWDetailViewBase *>  m_detailViews;

   mutable std::map<std::string, std::vector<std::string> > m_typeToViewers;
};
#endif
