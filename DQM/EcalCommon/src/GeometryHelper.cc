#include "DQM/EcalCommon/interface/GeometryHelper.h"

#include <utility>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/EcalCommon/interface/Numbers.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

namespace ecaldqm {

  std::map<std::string, MeInfo> MeInfoMap::infos;

  void
  MeInfoMap::set(MonitorElement *me, TClass *cl, ObjectType otype, BinningType btype, int ism)
  {

    if( !me ) return;

    MeInfo info;
    info.isa = cl;
    info.otype = otype;
    info.btype = btype;
    info.ism = ism;

    MeInfoMap::infos[ me->getFullname() ] = info;

  }

  const MeInfo *
  MeInfoMap::get(MonitorElement *me)
  {
    if( !me ) return NULL;

    std::map<std::string, MeInfo>::const_iterator it = MeInfoMap::infos.find( me->getFullname() );

    if( it == MeInfoMap::infos.end() ) return NULL;

    return &(it->second);
  }

  MonitorElement *
  bookME(DQMStore *dqmStore, const std::string &name, const std::string &title, const std::string &className, ObjectType otype, BinningType btype, int ism, double lowZ, double highZ, const char *option)
  {
    if( !dqmStore ) return NULL;

    if( className != "TH2F" && className != "TProfile2D" ) return NULL;

    Double_t xmin, ymin;
    Double_t xmax, ymax;

    xmin = ymin = 0.;
    xmax = ymax = 0.;

    switch(otype){
    case kFullEE:
      xmin = -100;
      xmax = 100.;
      ymax = 100.;
      break;
    case kEEp:
    case kEEm:
      xmax = 100.;
      ymax = 100.;
      break;
    case kEEpFar:
    case kEEmFar:
      xmin = 50.;
      xmax = 100.;
      ymax = 100.;
      break;
    case kEEpNear:
    case kEEmNear:
      xmax = 50.;
      ymax = 100.;
      break;
    case kSM:
      xmin = Numbers::ix0EE(ism);
      xmax = xmin + 50;
      ymin = Numbers::iy0EE(ism);
      ymax = ymin + 50;
      break;
    }

    Int_t nBinsX, nBinsY;

    switch(btype){
    case kCrystal:
    case kTriggerTower:
      nBinsX = (Int_t)(xmax - xmin);
      nBinsY = (Int_t)(ymax - ymin);
      break;
    case kSuperCrystal:
      nBinsX = (Int_t)(xmax - xmin) / 5;
      nBinsY = (Int_t)(ymax - ymin) / 5;
      break;
    default:
      nBinsX = nBinsY = 0;
      break;
    }

    MonitorElement *me;

    if( className == "TH2F" )
      me = dqmStore->book2D(name, title, nBinsX, xmin, xmax, nBinsY, ymin, ymax);
    else
      me = dqmStore->bookProfile2D(name, title, nBinsX, xmin, xmax, nBinsY, ymin, ymax, lowZ, highZ);

    MeInfoMap::set( me, TClass::GetClass(className.c_str()), otype, btype, ism );

    return me;
  }

  void
  fillME(MonitorElement *me, const EEDetId &id, double wz, double wprof)
  {

    if( !me ) return;

    const MeInfo *info = MeInfoMap::get( me );

    if( !info ) return;

    if(info->btype == kCrystal){

      float x = id.ix() - 0.5;
      float y = id.iy() - 0.5;
      if( info->otype == kSM && id.zside() < 0 ) x = 100 - x;

      if( info->isa == TClass::GetClass("TH2F") ) me->Fill( x, y, wz );
      else if( info->isa == TClass::GetClass("TProfile2D") ) me->Fill( x, y, wz, wprof );
      return;

    }else if(info->btype == kSuperCrystal){

      EcalScDetId scid( Numbers::getEcalScDetId( id ) );
      fillME( me, scid, wz, wprof );
      return;

    }else if(info->btype == kTriggerTower){

      const EcalElectronicsMapping *map = Numbers::getElectronicsMapping();
      EcalTriggerElectronicsId teid( map->getTriggerElectronicsId( id ) );
      EcalTrigTowerDetId ttid( map->getTrigTowerDetId( teid.tccId(), teid.ttId() ) );
      fillME( me, ttid, wz, wprof );
      return;

    }

  }

  void
  fillME(MonitorElement *me, const EcalScDetId &id, double wz, double wprof)
  {

    if( !me ) return;

    const MeInfo *info = MeInfoMap::get( me );

    if( !info ) return;

    if(info->btype == kCrystal){

      const EcalElectronicsMapping *map = Numbers::getElectronicsMapping();
      std::pair<int,int> p = map->getDCCandSC( id );
      std::vector<DetId> vcry = map->dccTowerConstituents( p.first, p.second );
      for(unsigned u = 0; u < vcry.size(); u++){
	EEDetId cid( vcry[u] );

	float x = cid.ix() - 0.5;
	float y = cid.iy() - 0.5;
	if( info->otype == kSM && cid.zside() < 0 ) x = 100 - x;

	if( info->isa == TClass::GetClass("TH2F") ) me->Fill( x, y, wz );
	else if( info->isa == TClass::GetClass("TProfile2D") ) me->Fill( x, y, wz, wprof );

      }
      return;

    }else if(info->btype == kSuperCrystal){

      float x = id.ix() * 5 - 2.5;
      float y = id.iy() * 5 - 2.5;
      if( info->otype == kSM && id.zside() < 0 ) x = 100 - x;

      if( info->isa == TClass::GetClass("TH2F") ) me->Fill( x, y, wz );
      else if( info->isa == TClass::GetClass("TProfile2D") ) me->Fill( x, y, wz, wprof );
      return;

    }

  }

  void
  fillME(MonitorElement *me, const EcalTrigTowerDetId &id, double wz, double wprof)
  {

    if( !me ) return;

    const MeInfo *info = MeInfoMap::get( me );

    if( !info ) return;

    if(info->btype == kTriggerTower || info->btype == kCrystal){

      std::vector<DetId> vcry = *( Numbers::crystals( id ) );
      for(unsigned u = 0; u < vcry.size(); u++){
	EEDetId cid( vcry[u] );

	float x = cid.ix() - 0.5;
	float y = cid.iy() - 0.5;
	if( info->otype == kSM && cid.zside() < 0 ) x = 100 - x;

	if( info->isa == TClass::GetClass("TH2F") ) me->Fill( x, y, wz );
	else if( info->isa == TClass::GetClass("TProfile2D") ) me->Fill( x, y, wz, wprof );
      }
      return;

    }

  }

}
