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
// $Id: FWDetailViewManager.h,v 1.23 2010/06/01 19:02:00 amraktad Exp $
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
   //  void assertMainFrame();
   void openDetailViewFor(const FWModelId&, const std::string&);
   void colorsChanged();
   void newEventCallback();
   void eveWindowDestroyed();

protected:
   FWColorManager                *m_colorManager;

private:

   FWDetailViewManager(const FWDetailViewManager&);    // stop default
   const FWDetailViewManager& operator=(const FWDetailViewManager&);    // stop default

   std::vector<std::string> findViewersFor(const std::string&) const;

   struct ViewFrame 
   {
      TEveCompositeFrameInMainFrame *m_eveFrame;
      FWDetailViewBase              *m_detailView;

      ViewFrame(TEveCompositeFrameInMainFrame *f, FWDetailViewBase* v):
         m_eveFrame(f), m_detailView(v) {}
   };

   typedef std::vector<ViewFrame> vViews_t;
   typedef vViews_t::iterator     vViews_i;
   vViews_t   m_views;

   mutable std::map<std::string, std::vector<std::string> > m_typeToViewers;
};

#endif
