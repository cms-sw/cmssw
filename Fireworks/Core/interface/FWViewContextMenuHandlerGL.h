#ifndef Fireworks_Core_FWViewContextMenuHandlerBaseGL_h
#define Fireworks_Core_FWViewContextMenuHandlerBaseGL_h

#include "Fireworks/Core/interface/FWViewContextMenuHandlerBase.h"

class FWEveView;
class FWModelId;

class FWViewContextMenuHandlerGL : public FWViewContextMenuHandlerBase
{
public:
   enum GLViewerAction { kAnnotate, kCameraCenter, kResetCameraCenter, kOrigin, kNone };

   FWViewContextMenuHandlerGL(FWEveView* v);
   ~FWViewContextMenuHandlerGL() override {}
   void select(int iEntryIndex, const FWModelId &id, int iX, int iY) override;

private:
   FWViewContextMenuHandlerGL(const FWViewContextMenuHandlerGL&) = delete; // stop default   
   const FWViewContextMenuHandlerGL& operator=(const FWViewContextMenuHandlerGL&) = delete; // stop default

   void init(FWViewContextMenuHandlerBase::MenuEntryAdder&, const FWModelId &id) override;

   FWEveView   *m_view;
};

#endif
