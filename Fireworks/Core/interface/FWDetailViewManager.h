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
// $Id: FWDetailViewManager.h,v 1.25 2011/05/27 04:03:42 amraktad Exp $
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
   void eveWindowDestroyed(TEveWindow*);

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
      TEveWindow                    *m_eveWindow;

      ViewFrame(TEveCompositeFrameInMainFrame *f, FWDetailViewBase* v, TEveWindow* w):
         m_eveFrame(f), m_detailView(v), m_eveWindow(w) {}
   };

   typedef std::vector<ViewFrame> vViews_t;
   typedef vViews_t::iterator     vViews_i;
   vViews_t   m_views;

   mutable std::map<std::string, std::vector<std::string> > m_typeToViewers;
};

#endif
