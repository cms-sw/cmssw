#ifndef Fireworks_Core_FWViewContextMenuHandlerBaseGL_h
#define Fireworks_Core_FWViewContextMenuHandlerBaseGL_h

#include "Fireworks/Core/interface/FWViewContextMenuHandlerBase.h"

class TEveViewer;

class FWViewContextMenuHandlerGL
{
public:
   enum GLViewerAction { kAnnotate, kPickCenter, kNone };

   FWViewContextMenuHandlerGL(TEveViewer* v): m_viewer(v) {}
   virtual ~FWViewContextMenuHandlerGL() {}
   virtual void select(int iEntryIndex, int iX, int iY);

private:
   FWViewContextMenuHandlerGL(const FWViewContextMenuHandlerGL&); // stop default   
   const FWViewContextMenuHandlerGL& operator=(const FWViewContextMenuHandlerGL&); // stop default

   virtual void init(FWViewContextMenuHandlerBase::MenuEntryAdder&);
 
   TEveViewer* m_viewer;
};

#endif
