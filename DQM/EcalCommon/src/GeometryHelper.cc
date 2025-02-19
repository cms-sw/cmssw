#include "DQM/EcalCommon/interface/GeometryHelper.h"

#include <utility>

#include "TH1.h"

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
  MeInfoMap::set(MonitorElement *me, ObjectType otype, BinningType btype, int ism)
  {

    if( !me ) return;

    MeInfo info;
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
      xmax = 200.;
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

    MeInfoMap::set( me, otype, btype, ism );

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
      if( info->otype == kFullEE && id.zside() > 0 ) x += 100;

      if( me->kind() == MonitorElement::DQM_KIND_TH2F ) me->Fill( x, y, wz );
      else if( me->kind() == MonitorElement::DQM_KIND_TPROFILE2D ) me->Fill( x, y, wz, wprof );
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
	if( info->otype == kFullEE && cid.zside() > 0 ) x += 100;

	if( me->kind() == MonitorElement::DQM_KIND_TH2F ) me->Fill( x, y, wz );
	else if( me->kind() == MonitorElement::DQM_KIND_TPROFILE2D ) me->Fill( x, y, wz, wprof );

      }
      return;

    }else if(info->btype == kSuperCrystal){

      float x = id.ix() * 5 - 2.5;
      float y = id.iy() * 5 - 2.5;
      if( info->otype == kSM && id.zside() < 0 ) x = 100 - x;
      if( info->otype == kFullEE && id.zside() > 0 ) x += 100;

      if( me->kind() == MonitorElement::DQM_KIND_TH2F ) me->Fill( x, y, wz );
      else if( me->kind() == MonitorElement::DQM_KIND_TPROFILE2D ) me->Fill( x, y, wz, wprof );
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
	if( info->otype == kFullEE && cid.zside() > 0 ) x += 100;

	if( me->kind() == MonitorElement::DQM_KIND_TH2F ) me->Fill( x, y, wz );
	else if( me->kind() == MonitorElement::DQM_KIND_TPROFILE2D ) me->Fill( x, y, wz, wprof );
      }
      return;

    }

  }

  int
  getBinME(MonitorElement *me, const EEDetId &id)
  {
    if( !me ) return -1;
    int kind = me->kind();
    if( kind < MonitorElement::DQM_KIND_TH2F || 
	kind == MonitorElement::DQM_KIND_TPROFILE ||
	kind == MonitorElement::DQM_KIND_TH3F ) return -1;

    const MeInfo *info = MeInfoMap::get( me );
    if( !info ) return -1;

    int ix = id.ix();
    int iy = id.iy();
    int zside = id.zside();
    int nx;
    int ism = info->ism;

    if(info->otype == kSM){

      if(zside > 0){
	ix -= Numbers::ix0EE(ism);
	iy -= Numbers::iy0EE(ism);
      }else{
	ix = 101 - ix - Numbers::ix0EE(ism);
	iy -= Numbers::iy0EE(ism);
      }
      nx = 50;

    }else{

      switch(info->otype){
      case kFullEE:
	if(zside > 0) ix += 100;
	nx = 200;
	break;
      case kEEp:
	if(zside < 0) return -1;
	nx = 100;
	break;
      case kEEm:
	if(zside > 0) return -1;
	nx = 100;
	break;
      case kEEpFar:
	if(zside < 0 || ix > 50) return -1;
	nx = 50;
	break;
      case kEEpNear:
	if(zside < 0 || ix < 51) return -1;
	ix -= 50;
	nx = 50;
	break;
      case kEEmFar:
	if(zside > 0 || ix > 50) return -1;
	nx = 50;
	break;
      case kEEmNear:
	if(zside > 0 || ix < 51) return -1;
	ix -= 50;
	nx = 50;
	break;
      default:
	return -1;
      }

    }

    int scale = info->btype == kSuperCrystal ? 5 : 1;
    ix = (ix - 1) / scale + 1;
    iy = (iy - 1) / scale + 1;
    nx = nx / scale;

    return iy * (nx + 2) + ix;
  }

  int
  getBinME(MonitorElement *me, const EcalScDetId &id)
  {
    if( !me ) return -1;
    int kind = me->kind();
    if( kind < MonitorElement::DQM_KIND_TH2F || 
	kind == MonitorElement::DQM_KIND_TPROFILE ||
	kind == MonitorElement::DQM_KIND_TH3F ) return -1;

    const MeInfo *info = MeInfoMap::get( me );
    if( !info ) return -1;

    if(info->btype != kSuperCrystal) return -1;

    int ix = id.ix();
    int iy = id.iy();
    int zside = id.zside();
    int nx;
    int ism = info->ism;

    if(info->otype == kSM){

      if(zside > 0){
	ix -= Numbers::ix0EE(ism) / 5;
	iy -= Numbers::iy0EE(ism) / 5;
      }else{
	ix = 21 - ix - Numbers::ix0EE(ism) / 5;
	iy -= Numbers::iy0EE(ism) / 5;
      }
      nx = 10;

    }else{

      switch(info->otype){
      case kFullEE:
	if(zside > 0) ix += 20;
	nx = 40;
	break;
      case kEEp:
	if(zside < 0) return -1;
	nx = 20;
	break;
      case kEEm:
	if(zside > 0) return -1;
	nx = 20;
	break;
      case kEEpFar:
	if(zside < 0 || ix > 10) return -1;
	nx = 10;
	break;
      case kEEpNear:
	if(zside < 0 || ix < 11) return -1;
	ix -= 10;
	nx = 10;
	break;
      case kEEmFar:
	if(zside > 0 || ix > 10) return -1;
	nx = 10;
	break;
      case kEEmNear:
	if(zside > 0 || ix < 11) return -1;
	ix -= 10;
	nx = 10;
	break;
      default:
	return -1;
      }

    }

    return iy * (nx + 2) + ix;

  }

  // the choices are either dynamic_casting or overloading. prefer overload in the interest of overhead reduction
  double
  getBinContentME(MonitorElement *me, const EEDetId &id)
  {
    if( !me ) return 0.;
    if( me->kind() < MonitorElement::DQM_KIND_TH1F ) return 0.;

    return ((TH1 *)me->getRootObject())->GetBinContent( getBinME( me, id ) );
  }

  double
  getBinContentME(MonitorElement *me, const EcalScDetId &id)
  {
    if( !me ) return 0.;
    if( me->kind() < MonitorElement::DQM_KIND_TH1F ) return 0.;

    return ((TH1 *)me->getRootObject())->GetBinContent( getBinME( me, id ) );
  }

  double
  getBinContentME(MonitorElement *me, const EcalTrigTowerDetId &id)
  {
    if( !me ) return 0.;
    if( me->kind() < MonitorElement::DQM_KIND_TH1F ) return 0.;

    const MeInfo *info = MeInfoMap::get( me );
    if( !info || info->btype != kTriggerTower ) return 0.;

    std::vector<DetId> *crystals = Numbers::crystals( id );
    if( !crystals->size() ) return 0.;

    return ((TH1 *)me->getRootObject())->GetBinContent( getBinME( me, EEDetId( crystals->at(0) ) ) );
  }

  double
  getBinErrorME(MonitorElement *me, const EEDetId &id)
  {
    if( !me ) return 0.;
    if( me->kind() < MonitorElement::DQM_KIND_TH1F ) return 0.;

    return ((TH1 *)me->getRootObject())->GetBinError( getBinME( me, id ) );
  }

  double
  getBinErrorME(MonitorElement *me, const EcalScDetId &id)
  {
    if( !me ) return 0.;
    if( me->kind() < MonitorElement::DQM_KIND_TH1F ) return 0.;

    return ((TH1 *)me->getRootObject())->GetBinError( getBinME( me, id ) );
  }

  double
  getBinErrorME(MonitorElement *me, const EcalTrigTowerDetId &id)
  {
    if( !me ) return 0.;
    if( me->kind() < MonitorElement::DQM_KIND_TH1F ) return 0.;

    const MeInfo *info = MeInfoMap::get( me );
    if( !info || info->btype != kTriggerTower ) return 0.;

    std::vector<DetId> *crystals = Numbers::crystals( id );
    if( !crystals->size() ) return 0.;

    return ((TH1 *)me->getRootObject())->GetBinError( getBinME( me, EEDetId( crystals->at(0) ) ) );
  }

  double
  getBinEntriesME(MonitorElement *me, const EEDetId &id)
  {
    if( !me ) return 0.;
    if( me->kind() != MonitorElement::DQM_KIND_TPROFILE2D ) return 0.;

    return ((TProfile2D *)me->getRootObject())->GetBinEntries( getBinME( me, id ) );
  }

  double
  getBinEntriesME(MonitorElement *me, const EcalScDetId &id)
  {
    if( !me ) return 0.;
    if( me->kind() != MonitorElement::DQM_KIND_TPROFILE2D ) return 0.;

    return ((TProfile2D *)me->getRootObject())->GetBinEntries( getBinME( me, id ) );
  }

  double
  getBinEntriesME(MonitorElement *me, const EcalTrigTowerDetId &id)
  {
    if( !me ) return 0.;
    if( me->kind() != MonitorElement::DQM_KIND_TPROFILE2D ) return 0.;

    const MeInfo *info = MeInfoMap::get( me );
    if( !info || info->btype != kTriggerTower ) return 0.;

    std::vector<DetId> *crystals = Numbers::crystals( id );
    if( !crystals->size() ) return 0.;


    return ((TProfile2D *)me->getRootObject())->GetBinEntries( getBinME( me, EEDetId( crystals->at(0) ) ) );
  }

  void
  setBinContentME(MonitorElement *me, const EEDetId &id, double content)
  {
    if( !me ) return;
    if( me->kind() < MonitorElement::DQM_KIND_TH1F ) return;

    const MeInfo *info = MeInfoMap::get( me );
    if( !info ) return;

    if(info->btype == kCrystal){

      ((TH1 *)me->getRootObject())->SetBinContent( getBinME( me, id ), content );
      return;

    }else if(info->btype == kSuperCrystal){

      EcalScDetId scid( Numbers::getEcalScDetId( id ) );
      ((TH1 *)me->getRootObject())->SetBinContent( getBinME( me, id ), content );
      return;

    }else if(info->btype == kTriggerTower){

      const EcalElectronicsMapping *map = Numbers::getElectronicsMapping();
      EcalTriggerElectronicsId teid( map->getTriggerElectronicsId( id ) );
      std::vector<DetId> vcry = map->ttConstituents( teid.tccId(), teid.ttId() );
      for(unsigned u = 0; u < vcry.size(); u++)
	((TH1 *)me->getRootObject())->SetBinContent( getBinME( me, EEDetId(vcry[u]) ), content );
      return;

    }
  }

  void
  setBinContentME(MonitorElement *me, const EcalScDetId &id, double content)
  {
    if( !me ) return;
    if( me->kind() < MonitorElement::DQM_KIND_TH1F ) return;

    const MeInfo *info = MeInfoMap::get( me );
    if( !info ) return;

    if(info->btype == kCrystal){

      const EcalElectronicsMapping *map = Numbers::getElectronicsMapping();
      std::pair<int,int> p = map->getDCCandSC( id );
      std::vector<DetId> vcry = map->dccTowerConstituents( p.first, p.second );
      for(unsigned u = 0; u < vcry.size(); u++)
	((TH1 *)me->getRootObject())->SetBinContent( getBinME( me, EEDetId(vcry[u]) ), content );
      return;

    }else if(info->btype == kSuperCrystal){

      ((TH1 *)me->getRootObject())->SetBinContent( getBinME( me, id ), content );
      return;

    }
  }

  void
  setBinContentME(MonitorElement *me, const EcalTrigTowerDetId &id, double content)
  {
    if( !me ) return;
    if( me->kind() < MonitorElement::DQM_KIND_TH1F ) return;

    const MeInfo *info = MeInfoMap::get( me );
    if( !info ) return;

    if(info->btype == kCrystal || info->btype == kTriggerTower){

      std::vector<DetId> *crystals = Numbers::crystals( id );
      for(unsigned u = 0; u < crystals->size(); u++)
	((TH1 *)me->getRootObject())->SetBinContent( getBinME( me, EEDetId(crystals->at(u)) ), content );
      return;

    }
  }

  void
  setBinErrorME(MonitorElement *me, const EEDetId &id, double error)
  {
    if( !me ) return;
    if( me->kind() < MonitorElement::DQM_KIND_TH1F ) return;

    const MeInfo *info = MeInfoMap::get( me );
    if( !info ) return;

    if(info->btype == kCrystal){

      ((TH1 *)me->getRootObject())->SetBinError( getBinME( me, id ), error );
      return;

    }else if(info->btype == kSuperCrystal){

      EcalScDetId scid( Numbers::getEcalScDetId( id ) );
      ((TH1 *)me->getRootObject())->SetBinError( getBinME( me, id ), error );
      return;

    }else if(info->btype == kTriggerTower){

      const EcalElectronicsMapping *map = Numbers::getElectronicsMapping();
      EcalTriggerElectronicsId teid( map->getTriggerElectronicsId( id ) );
      std::vector<DetId> vcry = map->ttConstituents( teid.tccId(), teid.ttId() );
      for(unsigned u = 0; u < vcry.size(); u++)
	((TH1 *)me->getRootObject())->SetBinError( getBinME( me, EEDetId(vcry[u]) ), error );
      return;

    }
  }

  void
  setBinErrorME(MonitorElement *me, const EcalScDetId &id, double error)
  {
    if( !me ) return;
    if( me->kind() < MonitorElement::DQM_KIND_TH1F ) return;

    const MeInfo *info = MeInfoMap::get( me );
    if( !info ) return;

    if(info->btype == kCrystal){

      const EcalElectronicsMapping *map = Numbers::getElectronicsMapping();
      std::pair<int,int> p = map->getDCCandSC( id );
      std::vector<DetId> vcry = map->dccTowerConstituents( p.first, p.second );
      for(unsigned u = 0; u < vcry.size(); u++)
	((TH1 *)me->getRootObject())->SetBinError( getBinME( me, EEDetId(vcry[u]) ), error );
      return;

    }else if(info->btype == kSuperCrystal){

      ((TH1 *)me->getRootObject())->SetBinError( getBinME( me, id ), error );
      return;

    }
  }

  void
  setBinErrorME(MonitorElement *me, const EcalTrigTowerDetId &id, double error)
  {
    if( !me ) return;
    if( me->kind() < MonitorElement::DQM_KIND_TH1F ) return;

    const MeInfo *info = MeInfoMap::get( me );
    if( !info ) return;

    if(info->btype == kCrystal || info->btype == kTriggerTower){

      std::vector<DetId> *crystals = Numbers::crystals( id );
      for(unsigned u = 0; u < crystals->size(); u++)
	((TH1 *)me->getRootObject())->SetBinError( getBinME( me, EEDetId(crystals->at(u)) ), error );
      return;

    }
  }

  void
  setBinEntriesME(MonitorElement *me, const EEDetId &id, double entries)
  {
    if( !me ) return;
    if( me->kind() != MonitorElement::DQM_KIND_TPROFILE2D ) return;

    const MeInfo *info = MeInfoMap::get( me );
    if( !info ) return;

    if(info->btype == kCrystal){

      ((TProfile2D *)me->getRootObject())->SetBinEntries( getBinME( me, id ), entries );
      return;

    }else if(info->btype == kSuperCrystal){

      EcalScDetId scid( Numbers::getEcalScDetId( id ) );
      ((TProfile2D *)me->getRootObject())->SetBinError( getBinME( me, id ), entries );
      return;

    }else if(info->btype == kTriggerTower){

      const EcalElectronicsMapping *map = Numbers::getElectronicsMapping();
      EcalTriggerElectronicsId teid( map->getTriggerElectronicsId( id ) );
      std::vector<DetId> vcry = map->ttConstituents( teid.tccId(), teid.ttId() );
      for(unsigned u = 0; u < vcry.size(); u++)
	((TProfile2D *)me->getRootObject())->SetBinError( getBinME( me, EEDetId(vcry[u]) ), entries );
      return;

    }

  }

  void
  setBinEntriesME(MonitorElement *me, const EcalScDetId &id, double entries)
  {
    if( !me ) return;
    if( me->kind() != MonitorElement::DQM_KIND_TPROFILE2D ) return;

    const MeInfo *info = MeInfoMap::get( me );
    if( !info ) return;

    if(info->btype == kCrystal){

      const EcalElectronicsMapping *map = Numbers::getElectronicsMapping();
      std::pair<int,int> p = map->getDCCandSC( id );
      std::vector<DetId> vcry = map->dccTowerConstituents( p.first, p.second );
      for(unsigned u = 0; u < vcry.size(); u++)
	((TProfile2D *)me->getRootObject())->SetBinEntries( getBinME( me, EEDetId(vcry[u]) ), entries );
      return;

    }else if(info->btype == kSuperCrystal){

      ((TProfile2D *)me->getRootObject())->SetBinError( getBinME( me, id ), entries );
      return;

    }
  }

  void
  setBinEntriesME(MonitorElement *me, const EcalTrigTowerDetId &id, double entries)
  {
    if( !me ) return;
    if( me->kind() != MonitorElement::DQM_KIND_TPROFILE2D ) return;

    const MeInfo *info = MeInfoMap::get( me );
    if( !info ) return;

    if(info->btype == kCrystal || info->btype == kTriggerTower){

      std::vector<DetId> *crystals = Numbers::crystals( id );
      for(unsigned u = 0; u < crystals->size(); u++)
	((TProfile2D *)me->getRootObject())->SetBinError( getBinME( me, EEDetId(crystals->at(u)) ), entries );
      return;

    }
  }

}
