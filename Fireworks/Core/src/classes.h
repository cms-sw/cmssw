
//Add includes for your classes here
#include "Fireworks/Core/interface/FWPhysicsObjectDesc.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWDisplayProperties.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
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
#include "Fireworks/Core/src/FWEveDigitSetScalableMarker.h"
#include "Fireworks/Core/src/FW3DViewDistanceMeasureTool.h"

#include "Fireworks/Core/src/CSGConnector.h"
#include "Fireworks/Core/interface/FWIntValueListenerBase.h"
#include "Fireworks/Core/interface/FWSummaryManager.h"
#include "Fireworks/Core/src/FWGUIEventDataAdder.h"
#include "Fireworks/Core/src/FWColorSelect.h"
#include "Fireworks/Core/src/FWNumberEntry.h"
#include "Fireworks/Core/src/FWCollectionSummaryWidget.h"
#include "Fireworks/Core/src/FWCompactVerticalLayout.h"
#include "Fireworks/Core/src/FWModelContextMenuHandler.h"

#include "Fireworks/Core/interface/CmsShowEDI.h"
#include "Fireworks/Core/interface/FWViewEnergyScaleEditor.h"
#include "Fireworks/Core/interface/CmsShowSearchFiles.h"

#include "Fireworks/Core/interface/FWGUISubviewArea.h"
#include "Fireworks/Core/interface/CmsShowMainFrame.h"
#include "Fireworks/Core/src/FWGUIValidatingTextEntry.h"

#include "Fireworks/Core/interface/CmsShowModelPopup.h"
#include "Fireworks/Core/interface/FWGUIEventFilter.h"
#include "Fireworks/Core/interface/CmsShowCommonPopup.h"
#include "Fireworks/Core/interface/CmsShowViewPopup.h"
#include "Fireworks/Core/interface/FWInvMassDialog.h"

#include "Fireworks/Core/interface/FWGeometryTableViewBase.h"
#include "Fireworks/Core/src/FWGeometryTableView.h"
#include "Fireworks/Core/src/FWOverlapTableView.h"
#include "Fireworks/Core/interface/FWTextProjected.h"

#include "Fireworks/Core/interface/FWGeoTopNode.h"
#include "Fireworks/Core/src/FWEveOverlap.h"
#include "Fireworks/Core/src/FWEveDetectorGeo.h"
#include "Fireworks/Core/src/FWGeoTopNodeGL.h"

#include "Fireworks/Core/interface/FWPartialConfig.h"

namespace Fireworks_Core {
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
