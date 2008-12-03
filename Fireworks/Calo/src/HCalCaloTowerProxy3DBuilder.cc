// -*- C++ -*-
// $Id: HCalCaloTowerProxy3DBuilder.cc,v 1.8 2008/11/26 16:19:12 chrjones Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "RVersion.h"
#include "TEveCalo.h"
#include "TEveCaloData.h"
#include "TH2F.h"
#include "Fireworks/Calo/interface/HCalCaloTowerProxy3DBuilder.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"


// user include files
std::string
HCalCaloTowerProxy3DBuilder::histName() const
{
   return "hcal3D";
}

REGISTER_FWRPZDATAPROXYBUILDERBASE(HCalCaloTowerProxy3DBuilder,CaloTowerCollection,"HCal");
