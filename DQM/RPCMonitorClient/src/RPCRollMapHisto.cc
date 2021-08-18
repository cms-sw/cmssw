#include "DQM/RPCMonitorClient/interface/RPCRollMapHisto.h"
#include <fmt/format.h>

typedef dqm::reco::MonitorElement MonitorElement;
//typedef DQMEDHarvester::MonitorElement MonitorElement;
typedef dqm::reco::DQMStore::IBooker IBooker;

MonitorElement* RPCRollMapHisto::bookBarrel(
    IBooker& booker, const int wheel, const std::string& name, const std::string& title, const bool useRollInfo) {
  MonitorElement* me = booker.book2D(name, title, 12, 0.5, 12.5, 21, 0.5, 21.5);

  TH2* h = dynamic_cast<TH2*>(me->getTH1());
  h->GetXaxis()->SetNoAlphanumeric(true);
  // Set x-axis labels
  for (int i = 1; i <= 12; ++i) {
    me->setBinLabel(i, fmt::format("Sec{}", i), 1);
  }

  // Set y-axis labels
  RPCRollMapHisto::setBarrelRollAxis(me, wheel, 2, useRollInfo);

  return me;
}

void RPCRollMapHisto::setBarrelRollAxis(MonitorElement* me, const int wheel, const int axis, const bool useRollInfo) {
  TH1* h = dynamic_cast<TH1*>(me->getTH1());
  if (axis == 1)
    h->GetXaxis()->SetNoAlphanumeric(true);
  else if (axis == 2)
    h->GetYaxis()->SetNoAlphanumeric(true);

  const std::array<const std::string, 21> labelsRoll = {
      {"RB1in_B",  "RB1in_F",  "RB1out_B", "RB1out_F", "RB2in_B", "RB2in_F", "RB2in_M",
       "RB2out_B", "RB2out_F", "RB3-_B",   "RB3-_F",   "RB3+_B",  "RB3+_F",  "RB4,-_B",
       "RB4,-_F",  "RB4+_B",   "RB4+_F",   "RB4--_B",  "RB4--_F", "RB4++_B", "RB4++_F"}};
  const std::array<const std::string, 21> labelsCh = {{"RB1in",  "",     "RB1out", "",      "RB2in", "",      "",
                                                       "RB2out", "",     "RB3-",   "",      "RB3+",  "",      "RB4,-",
                                                       "",       "RB4+", "",       "RB4--", "",      "RB4++", ""}};

  for (int i = 0, n = std::min(21, me->getNbinsY()); i < n; ++i) {
    const std::string label = useRollInfo ? labelsRoll[i] : labelsCh[i];
    me->setBinLabel(i + 1, label, 2);
  }
  if (useRollInfo and std::abs(wheel) == 2) {
    // We have RB2out_M for the wheel +-2, otherwise, RB2in_M as in the default array
    me->setBinLabel(7, "RB2out_M", 2);
  }
}

void RPCRollMapHisto::setEndcapRollAxis(MonitorElement* me, const int disk, const int axis, const bool useRollInfo) {
  TH1* h = dynamic_cast<TH1*>(me->getTH1());
  if (axis == 1)
    h->GetXaxis()->SetNoAlphanumeric(true);
  else if (axis == 2)
    h->GetYaxis()->SetNoAlphanumeric(true);

  // NOTE: Let us keep only Ring2 and Ring3 for the Run3
  //       There will be Ring1, RE3/1 and RE4/1 only from the phase-2
  const std::array<const std::string, 6> labelsRoll = {{//"C", "Ring1 B", "A",
                                                        "C",
                                                        "Ring2 B",
                                                        "A",
                                                        "C",
                                                        "Ring3 B",
                                                        "A"}};
  const std::array<const std::string, 6> labelsCh = {{//"", "Ring1", "",
                                                      "",
                                                      "Ring2",
                                                      "",
                                                      "",
                                                      "Ring3",
                                                      ""}};

  //const int offset = std::abs(disk) >= 3 ? 0 : 3;
  for (int i = 0; i < 6; ++i) {
    const std::string label = useRollInfo ? labelsRoll[i] : labelsCh[i];
    me->setBinLabel(i + 1, label, 2);
  }
}

MonitorElement* RPCRollMapHisto::bookEndcap(
    IBooker& booker, const int disk, const std::string& name, const std::string& title, const bool useRollInfo) {
  MonitorElement* me = booker.book2D(name, title, 36, 0.5, 36.5, 6, 0.5, 6.5);
  TH2* h = dynamic_cast<TH2*>(me->getTH1());
  h->GetXaxis()->SetNoAlphanumeric(true);
  h->GetYaxis()->SetNoAlphanumeric(true);

  // Set x-axis labels
  for (int i = 1; i <= 36; ++i) {
    me->setBinLabel(i, fmt::format("{}", i), 1);
  }
  me->setAxisTitle("Segments", 1);

  RPCRollMapHisto::setEndcapRollAxis(me, disk, 2, useRollInfo);

  return me;
}
