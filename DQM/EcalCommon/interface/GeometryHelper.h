#ifndef DQM_ECALCOMMON_GeomHelperFunctions_H
#define DQM_ECALCOMMON_GeomHelperFunctions_H

/*!
  \file GeomHelperFunctions.h
  \brief Ecal Monitor Utility functions
  \author Yutaro Iiyama
  \version $Revision: 1.2 $
  \date $Date: 2011/08/05 12:07:11 $
*/

#include <string>
#include <map>

class DQMStore;
class MonitorElement;
class EEDetId;
class EcalScDetId;
class EcalTrigTowerDetId;

namespace ecaldqm {

  enum ObjectType {
    kFullEE,
    kEEp,
    kEEm,
    kEEpFar, kEEpNear,
    kEEmFar, kEEmNear,
    kSM
  };

  enum BinningType {
    kCrystal,
    kSuperCrystal,
    kTriggerTower
  };

  struct MeInfo {
    ObjectType otype;
    BinningType btype;
    int ism;
  };

  class MeInfoMap {

  public:
    static void set(MonitorElement *me, ObjectType otype, BinningType btype, int ism);
    static const MeInfo *get(MonitorElement *me);

  private:
    static std::map<std::string, MeInfo> infos;
  };

  // dqmStore must be cd'ed to the desired directory before passing
  MonitorElement *bookME(DQMStore *dqmStore, const std::string &name, const std::string &title, const std::string &className, ObjectType otype, BinningType btype = kCrystal, int ism = 0, double lowZ = 0., double highZ = 0., const char *option = "s");

  void fillME(MonitorElement *me, const EEDetId &id, double wz = 1., double wprof = 1.);
  void fillME(MonitorElement *me, const EcalScDetId &id, double wz = 1., double wprof = 1.);
  void fillME(MonitorElement *me, const EcalTrigTowerDetId &id, double wz = 1., double wprof = 1.);

  int getBinME(MonitorElement *me, const EEDetId &id);
  int getBinME(MonitorElement *me, const EcalScDetId &id);
  int getBinME(MonitorElement *me, const EcalTrigTowerDetId &id);

  double getBinContentME(MonitorElement *me, const EEDetId &id);
  double getBinContentME(MonitorElement *me, const EcalScDetId &id);
  double getBinContentME(MonitorElement *me, const EcalTrigTowerDetId &id);

  double getBinErrorME(MonitorElement *me, const EEDetId &id);
  double getBinErrorME(MonitorElement *me, const EcalScDetId &id);
  double getBinErrorME(MonitorElement *me, const EcalTrigTowerDetId &id);

  double getBinEntriesME(MonitorElement *me, const EEDetId &id);
  double getBinEntriesME(MonitorElement *me, const EcalScDetId &id);
  double getBinEntriesME(MonitorElement *me, const EcalTrigTowerDetId &id);

  void setBinContentME(MonitorElement *me, const EEDetId &id, double content);
  void setBinContentME(MonitorElement *me, const EcalScDetId &id, double content);
  void setBinContentME(MonitorElement *me, const EcalTrigTowerDetId &id, double content);

  void setBinErrorME(MonitorElement *me, const EEDetId &id, double error);
  void setBinErrorME(MonitorElement *me, const EcalScDetId &id, double error);
  void setBinErrorME(MonitorElement *me, const EcalTrigTowerDetId &id, double error);

  void setBinEntriesME(MonitorElement *me, const EEDetId &id, double entries);
  void setBinEntriesME(MonitorElement *me, const EcalScDetId &id, double entries);
  void setBinEntriesME(MonitorElement *me, const EcalTrigTowerDetId &id, double entries);

}

#endif
