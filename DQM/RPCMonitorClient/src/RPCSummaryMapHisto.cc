#include "DQM/RPCMonitorClient/interface/RPCSummaryMapHisto.h"
#include <fmt/format.h>

typedef dqm::implementation::MonitorElement MonitorElement;
typedef dqm::implementation::IBooker IBooker;

MonitorElement* RPCSummaryMapHisto::book(IBooker& booker, const std::string& name, const std::string& title) {
  MonitorElement* me = booker.book2D(name, title, 15, -7.5, 7.5, 12, 0.5, 12.5);
  me->getTH1()->SetMinimum(-1e-5);

  // Customize the 2d histogram
  for (int sec = 1; sec <= 12; ++sec) {
    me->setBinLabel(sec, fmt::format("Sec{}", sec), 2);
  }

  for (int disk = 1; disk <= 4; ++disk) {
    me->setBinLabel(11 + disk, fmt::format("Disk{}", disk), 1);
    me->setBinLabel(5 - disk, fmt::format("Disk{}", -disk), 1);
  }

  for (int wheel = -2; wheel <= 2; ++wheel) {
    me->setBinLabel(8 + wheel, fmt::format("Wheel{}", wheel), 1);
  }

  // Fill with -1 as the default value
  for (int sec = 1; sec <= 12; ++sec) {
    for (int xbin = 1; xbin <= 15; ++xbin) {
      me->setBinContent(xbin, sec, -1);
    }
  }

  return me;
}

void RPCSummaryMapHisto::setBinsBarrel(MonitorElement* me, const double value) {
  for (int wheel = -2; wheel <= 2; ++wheel) {
    for (int sector = 1; sector <= 12; ++sector) {
      setBinBarrel(me, wheel, sector, value);
    }
  }
}

void RPCSummaryMapHisto::setBinsEndcap(MonitorElement* me, const double value) {
  for (int disk = -4; disk <= 4; ++disk) {
    for (int sector = 1; sector <= 6; ++sector) {
      setBinEndcap(me, disk, sector, value);
    }
  }
}

void RPCSummaryMapHisto::setBinBarrel(MonitorElement* me, const int wheel, const int sector, const double value) {
  if (std::abs(wheel) > 2 or sector < 1 or sector > 12)
    return;
  me->setBinContent(8 + wheel, sector, value);
}

void RPCSummaryMapHisto::setBinEndcap(MonitorElement* me, const int disk, const int sector, const double value) {
  if (std::abs(disk) < 1 or std::abs(disk) > 4)
    return;
  if (sector < 1 or sector > 6)
    return;

  if (disk > 0) {
    me->setBinContent(11 + disk, sector, value);
  } else {
    me->setBinContent(5 + disk, sector, value);
  }
}
