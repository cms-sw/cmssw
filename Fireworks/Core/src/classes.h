
//Add includes for your classes here
#include "Fireworks/Core/interface/FWDisplayEvent.h"
#include "Fireworks/Core/interface/FWDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWPhysicsObjectDesc.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWDisplayProperties.h"
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"
#include "Fireworks/Core/interface/FW3DLegoViewManager.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/TracksProxy3DBuilder.h"
#include "Fireworks/Core/interface/TracksRecHitsProxy3DBuilder.h"
#include "Fireworks/Core/interface/TEveElementIter.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWGUISubviewArea.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWRhoPhiZViewManager.h"
#include "Fireworks/Core/interface/FW3DLegoView.h"
#include "Fireworks/Core/interface/ElectronsProxySCBuilder.h"
#include "Fireworks/Core/interface/MuonsProxyPUBuilder.h"
#include "Fireworks/Core/interface/FWDetailViewManager.h"
#include "Fireworks/Core/interface/ElectronDetailView.h"
#include "Fireworks/Core/interface/TrackDetailView.h"
#include "Fireworks/Core/interface/MuonDetailView.h"
#include "Fireworks/Core/src/FWListEventItem.h"
#include "Fireworks/Core/src/FWListEventItemEditor.h"
#include "Fireworks/Core/src/FWListModel.h"
#include "Fireworks/Core/src/FWListModelEditor.h"
#include "Fireworks/Core/interface/ElectronView.h"
#include "Fireworks/Core/src/FWListViewObject.h"
#include "Fireworks/Core/src/FWListViewObjectEditor.h"
#include "Fireworks/Core/src/FWListItemBase.h"
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/src/FWDoubleParameterSetter.h"
#include "Fireworks/Core/interface/FWDoubleParameter.h"
#include "Fireworks/Core/src/FWLongParameterSetter.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/GenParticleProxy3DBuilder.h"
#include "Fireworks/Core/interface/GenParticleDetailView.h"


namespace {
   struct Fireworks_Core {
      //add 'dummy' Wrapper variable for each class type you put into the Event
     //FWDisplayEvent de;
      FWConfiguration::KeyValues kv;
   };
}
