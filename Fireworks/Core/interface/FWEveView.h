#ifndef Fireworks_Core_FWEveView_h
#define Fireworks_Core_FWEveView_h

// user include files
#include "Fireworks/Core/interface/FWViewBase.h"
#include "Fireworks/Core/interface/FWDoubleParameter.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWEvePtr.h"

// forward declarations
class TGLViewer;
class TEveViewer;
class TEveElementList;
class TEveScene;
class TEveWindowSlot;
class FWEventAnnotation;
class CmsAnnotation;
class FWViewContextMenuHandlerGL;

class FWEveView : public FWViewBase
{
public:
   FWEveView(TEveWindowSlot*);
   virtual ~FWEveView();

   virtual void addTo(FWConfiguration&) const;
   virtual void setFrom(const FWConfiguration&);
   virtual FWViewContextMenuHandlerBase* contextMenuHandler() const;

   virtual void saveImageTo(const std::string& iName) const;
   virtual void setBackgroundColor(Color_t);
   virtual void eventEnd();

   void resetCamera();

protected:
   TEveViewer* viewer() { return m_viewer; }
   TEveScene*  scene()  { return m_scene; }
   TGLViewer*  viewerGL() const;

   virtual void lineWidthChanged();

private:
   FWEveView(const FWEveView&);    // stop default
   const FWEveView& operator=(const FWEveView&);    // stop default
  
   virtual const std::string& typeName() const;

   // ---------- member data --------------------------------
   TEveViewer*          m_viewer;
   TEveScene*           m_scene;

   FWEventAnnotation* m_overlayEventInfo;  
   CmsAnnotation*     m_overlayLogo;

   // parameters
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,26,0)
   FWDoubleParameter   m_imageScale;
#endif
   FWLongParameter   m_eventInfoLevel;
   FWBoolParameter   m_drawCMSLogo;

#if ROOT_VERSION_CODE < ROOT_VERSION(5,26,0)
   FWDoubleParameter m_lineWidth;
#endif

   boost::shared_ptr<FWViewContextMenuHandlerGL>   m_viewContextMenu;
};


#endif
