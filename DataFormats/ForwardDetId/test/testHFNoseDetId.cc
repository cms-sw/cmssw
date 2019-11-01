#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HFNoseTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetIdToModule.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

void testCell(int type) {
  int N = (type == 0) ? HFNoseDetId::HFNoseFineN : HFNoseDetId::HFNoseCoarseN;
  const int waferu(0), waferv(0), layer(1), zside(1);
  std::map<std::pair<int, int>, int> triggers;
  int ntot(0);
  for (int u = 0; u < 2 * N; ++u) {
    for (int v = 0; v < 2 * N; ++v) {
      if (((v - u) < N) && (u - v) <= N) {
        HFNoseDetId id(zside, type, layer, waferu, waferv, u, v);
        std::cout << "ID " << std::hex << id.rawId() << std::dec << " " << id << " Trigger: " << id.triggerCellU()
                  << ":" << id.triggerCellV() << std::endl;
        std::pair<int, int> trig = id.triggerCellUV();
        std::map<std::pair<int, int>, int>::iterator itr = triggers.find(trig);
        if (itr == triggers.end()) {
          triggers[trig] = 0;
          itr = triggers.find(trig);
        }
        ++(itr->second);
        ++ntot;
      }
    }
  }
  std::cout << "Total of " << ntot << " cells in type " << type << " with " << triggers.size() << " trigger cells"
            << std::endl;
  int k(0);
  for (auto itr : triggers) {
    std::cout << "Trigger[" << k << "]  (" << (itr.first).first << ":" << (itr.first).second << ")  " << itr.second
              << std::endl;
    ++k;
  }
}

void testWafer(int layer, double rin, double rout) {
  const double waferSize(167.4408);
  const double rMaxFine(750.0), rMaxMiddle(1200.0);
  const int zside(1), cellu(0), cellv(0);
  const std::string waferType[2] = {"Virtual", "Real   "};
  double r = 0.5 * waferSize;
  double R = 2.0 * r / std::sqrt(3.0);
  double dy = 0.75 * R;
  double xc[6], yc[6];
  int N = (int)(0.5 * rout / r) + 2;
  int nreal(0), nvirtual(0);
  int ntype[3] = {0, 0, 0};
  for (int v = -N; v <= N; ++v) {
    for (int u = -N; u <= N; ++u) {
      int nr = 2 * v;
      int nc = -2 * u + v;
      double xpos = nc * r;
      double ypos = nr * dy;
      xc[0] = xpos + r;
      yc[0] = ypos + 0.5 * R;
      xc[1] = xpos;
      yc[1] = ypos + R;
      xc[2] = xpos - r;
      yc[2] = ypos + 0.5 * R;
      xc[3] = xpos - r;
      yc[3] = ypos - 0.5 * R;
      xc[4] = xpos;
      yc[4] = ypos - R;
      xc[5] = xpos + r;
      yc[5] = ypos - 0.5 * R;
      int cornerOne(0), cornerAll(1);
      for (int k = 0; k < 6; ++k) {
        double rpos = std::sqrt(xc[k] * xc[k] + yc[k] * yc[k]);
        if (rpos >= rin && rpos <= rout)
          cornerOne = 1;
        else
          cornerAll = 0;
      }
      if (cornerOne > 0) {
        double rr = std::sqrt(xpos * xpos + ypos * ypos);
        int type = (rr < rMaxFine) ? 0 : ((rr < rMaxMiddle) ? 1 : 2);
        HFNoseDetId id(zside, type, layer, u, v, cellu, cellv);
        std::cout << waferType[cornerAll] << " Wafer " << id << std::endl;
        if (cornerAll == 1) {
          ++nreal;
          ++ntype[type];
        } else {
          ++nvirtual;
        }
      }
    }
  }
  std::cout << nreal << " full wafers of type 0:" << ntype[0] << " 1:" << ntype[1] << " 2:" << ntype[2] << " and "
            << nvirtual << " partial wafers for r-range " << rin << ":" << rout << std::endl;
}

void testTriggerCell(int type) {
  int N = (type == 0) ? HFNoseDetId::HFNoseFineN : HFNoseDetId::HFNoseCoarseN;
  const int waferu(0), waferv(0), layer(1);
  std::string error[2] = {"ERROR", "OK"};
  int ntot(0), nerror(0);
  for (int iz = 0; iz <= 1; ++iz) {
    int zside = 2 * iz - 1;
    for (int u = 0; u < 2 * N; ++u) {
      for (int v = 0; v < 2 * N; ++v) {
        if (((v - u) < N) && (u - v) <= N) {
          HFNoseDetId id(zside, type, layer, waferu, waferv, u, v);
          HFNoseTriggerDetId idt((int)(HFNoseTrigger),
                                 id.zside(),
                                 id.type(),
                                 id.layer(),
                                 id.waferU(),
                                 id.waferV(),
                                 id.triggerCellU(),
                                 id.triggerCellV());
          std::cout << "ID " << std::hex << id.rawId() << std::dec << " " << id << " Trigger: " << id.triggerCellU()
                    << ":" << id.triggerCellV() << " Trigger " << idt << std::endl;
          int ok(0);
          std::vector<std::pair<int, int> > uvs = idt.cellUV();
          for (auto const& uv : uvs) {
            HFNoseDetId idn(idt.zside(), idt.type(), idt.layer(), idt.waferU(), idt.waferV(), uv.first, uv.second);
            if (idn == id) {
              ok = 1;
              break;
            }
          }
          std::cout << "Trigger Cell: " << idt << " obtained from cell (" << error[ok] << ")" << std::endl;
          std::cout << "Check " << idt << " from rawId " << HGCalTriggerDetId(idt.rawId()) << " from DetId "
                    << HGCalTriggerDetId(DetId(idt.rawId())) << std::endl;
          ++ntot;
          if (ok == 0)
            ++nerror;
        }
      }
    }
  }
  std::cout << "Total of " << ntot << " cells in type " << type << " with " << nerror << " errors for trigger cells"
            << std::endl;
}

void testModule(HFNoseDetId const& id) {
  HFNoseDetIdToModule hfn;
  HFNoseDetId module = hfn.getModule(id);
  std::vector<HFNoseDetId> ids = hfn.getDetIds(module);
  std::string ok = "***** ERROR *****";
  for (auto const& id0 : ids) {
    if (id0 == id) {
      ok = "";
      break;
    }
  }
  std::cout << "Module ID of " << id << " is " << module << " which has " << ids.size() << " cells " << ok << std::endl;
  for (unsigned int k = 0; k < ids.size(); ++k)
    std::cout << "ID[" << k << "] " << ids[k] << std::endl;
}

int main() {
  testCell(0);
  testWafer(1, 299.47, 1041.45);
  testWafer(8, 312.55, 1086.97);
  testTriggerCell(0);
  testModule(HFNoseDetId(1, 0, 1, 3, 3, 0, 5));
  testModule(HFNoseDetId(-1, 0, 5, 2, -2, 7, 5));
  return 0;
}
