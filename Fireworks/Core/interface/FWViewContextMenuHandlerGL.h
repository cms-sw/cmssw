#ifndef Fireworks_Core_FWViewContextMenuHandlerBaseGL_h
#define Fireworks_Core_FWViewContextMenuHandlerBaseGL_h

#include "Fireworks/Core/interface/FWViewContextMenuHandlerBase.h"

class FWEveView;
class FWModelId;

class FWViewContextMenuHandlerGL : public FWViewContextMenuHandlerBase
{
public:
   enum GLViewerAction { kAnnotate, kCameraCenter, kResetCameraCenter,kOrigin,  kNone };

   FWViewContextMenuHandlerGL(FWEveView* v);
   virtual ~FWViewContextMenuHandlerGL() {}
   virtual void select(int iEntryIndex, const FWModelId &id, int iX, int iY);
   
private:
   FWViewContextMenuHandlerGL(const FWViewContextMenuHandlerGL&); // stop default   
   const FWViewContextMenuHandlerGL& operator=(const FWViewContextMenuHandlerGL&); // stop default

   virtual void init(FWViewContextMenuHandlerBase::MenuEntryAdder&, const FWModelId &id);
 
    FWEveView*  m_view;
};

#endif
