
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
#include "Fireworks/Core/interface/MuonDetailView.h"
#include "Fireworks/Core/src/FWListEventItem.h"
#include "Fireworks/Core/src/FWListEventItemEditor.h"
#include "Fireworks/Core/src/FWListModel.h"
#include "Fireworks/Core/src/FWListModelEditor.h"
#include "Fireworks/Core/interface/ElectronView.h"

namespace {
   struct Fireworks_Core {
      //add 'dummy' Wrapper variable for each class type you put into the Event
     //FWDisplayEvent de;
   };
}
