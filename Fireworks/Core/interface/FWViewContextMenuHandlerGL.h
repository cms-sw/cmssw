#ifndef Fireworks_Core_FWViewContextMenuHandlerBaseGL_h
#define Fireworks_Core_FWViewContextMenuHandlerBaseGL_h

#include "Fireworks/Core/interface/FWViewContextMenuHandlerBase.h"

class TEveViewer;
class FWModelId;

class FWViewContextMenuHandlerGL
{
public:
   enum GLViewerAction { kAnnotate, kCameraCenter, kNone };

   FWViewContextMenuHandlerGL(TEveViewer* v);
   virtual ~FWViewContextMenuHandlerGL() {}
   virtual void select(int iEntryIndex, const FWModelId &id, int iX, int iY);

   void    setPickCameraCenter(bool x) { m_pickCameraCenter = x; }
   
private:
   FWViewContextMenuHandlerGL(const FWViewContextMenuHandlerGL&); // stop default   
   const FWViewContextMenuHandlerGL& operator=(const FWViewContextMenuHandlerGL&); // stop default

   virtual void init(FWViewContextMenuHandlerBase::MenuEntryAdder&);
 
   TEveViewer* m_viewer;
   bool        m_pickCameraCenter;
};

#endif
