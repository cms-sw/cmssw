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
//
#include <map>
#include <memory>
#include <string>
#include <vector>

class FWColorManager;
class TEveCompositeFrameInMainFrame;
class FWDetailViewBase;
class FWModelId;
class TEveWindow;

namespace fireworks {
  class Context;
}

class FWDetailViewManager {
public:
  FWDetailViewManager(fireworks::Context*);
  virtual ~FWDetailViewManager();

  std::vector<std::string> detailViewsFor(const FWModelId&) const;
  //  void assertMainFrame();
  void openDetailViewFor(const FWModelId&, const std::string&);
  void colorsChanged();
  void newEventCallback();
  void eveWindowDestroyed(TEveWindow*);

  struct ViewFrame {
    TEveCompositeFrameInMainFrame* m_eveFrame;
    std::unique_ptr<FWDetailViewBase> m_detailView;
    TEveWindow* m_eveWindow;

    ViewFrame(TEveCompositeFrameInMainFrame* f, std::unique_ptr<FWDetailViewBase> v, TEveWindow* w);
    ~ViewFrame();
    ViewFrame(const ViewFrame&) = delete;
    ViewFrame& operator=(const ViewFrame&) = delete;
    ViewFrame(ViewFrame&&) = default;
    ViewFrame& operator=(ViewFrame&&) = default;
  };

protected:
  fireworks::Context* m_context;

private:
  FWDetailViewManager(const FWDetailViewManager&) = delete;                   // stop default
  const FWDetailViewManager& operator=(const FWDetailViewManager&) = delete;  // stop default

  std::vector<std::string> findViewersFor(const std::string&) const;

  typedef std::vector<ViewFrame> vViews_t;
  typedef vViews_t::iterator vViews_i;
  vViews_t m_views;

  mutable std::map<std::string, std::vector<std::string> > m_typeToViewers;
};

#endif
