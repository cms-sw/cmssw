
//Add includes for your classes here
#include "Fireworks/Core/interface/FWPhysicsObjectDesc.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWDisplayProperties.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
// #include "Fireworks/Core/interface/FWGeometryTable.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWDetailViewManager.h"
#include "Fireworks/Core/interface/FWEveViewManager.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/src/FWDoubleParameterSetter.h"
#include "Fireworks/Core/interface/FWDoubleParameter.h"
#include "Fireworks/Core/src/FWLongParameterSetter.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/src/FWBoolParameterSetter.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"
#include "Fireworks/Core/src/FWStringParameterSetter.h"
#include "Fireworks/Core/interface/FWStringParameter.h"
#include "Fireworks/Core/interface/FWEnumParameter.h"
#include "Fireworks/Core/interface/FWEnumParameterSetter.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/src/CmsShowMain.h"
#include "Fireworks/Core/interface/FWTableView.h"
#include "Fireworks/Core/interface/FWTriggerTableView.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWGUIEventSelector.h"
#include "Fireworks/Core/interface/FWTEventList.h"
#include "Fireworks/Core/interface/FWTSelectorToEventList.h"
#include "Fireworks/Core/src/FWEveDigitSetScalableMarker.cc"
#include "Fireworks/Core/src/FW3DViewDistanceMeasureTool.h"
namespace {
   struct Fireworks_Core {
      //add 'dummy' Wrapper variable for each class type you put into the Event
      //FWDisplayEvent de;
      FWBoolParameter bp;
      FWDoubleParameter dp;
      FWLongParameter lp;
      FWStringParameter sp;
      FWConfiguration::KeyValues kv;
   };
}
