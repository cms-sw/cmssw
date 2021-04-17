#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetIdToModule.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetIdToROC.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

void testCell(int type) {
  int N = (type == 0) ? HGCSiliconDetId::HGCalFineN : HGCSiliconDetId::HGCalCoarseN;
  const int waferu(0), waferv(0), layer(1), zside(1);
  std::map<std::pair<int, int>, int> triggers;
  int ntot(0);
  for (int u = 0; u < 2 * N; ++u) {
    for (int v = 0; v < 2 * N; ++v) {
      if (((v - u) < N) && (u - v) <= N) {
        HGCSiliconDetId id(DetId::HGCalEE, zside, type, layer, waferu, waferv, u, v);
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
        HGCSiliconDetId id(DetId::HGCalEE, zside, type, layer, u, v, cellu, cellv);
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

void testScint(int layer) {
  int type = ((layer <= 8) ? 0 : ((layer <= 17) ? 1 : 2));
  int phimax = (type == 0) ? 360 : 288;
  int irmin = ((layer <= 12) ? 10 : ((layer <= 14) ? 14 : ((layer <= 18) ? 7 : 1)));
  int irmax = ((layer <= 12) ? 33 : ((layer <= 14) ? 37 : 42));
  int sipm = (layer <= 17) ? 0 : 1;
  for (int ring = irmin; ring <= irmax; ++ring) {
    for (int phi = 1; phi <= phimax; ++phi) {
      for (int zp = 0; zp < 2; ++zp) {
        int radius = (2 * zp - 1) * ring;
        HGCScintillatorDetId id(type, layer, radius, phi, false, sipm);
        std::cout << "Input " << type << ":" << layer << ":" << radius << ":" << phi << ":" << sipm << " ID "
                  << std::hex << id << std::dec;
        if ((id.iradius() != radius) || (id.iphi() != phi) || (id.layer() != layer) || (id.type() != type) ||
            (id.sipm() != sipm))
          std::cout << " ***** ERROR *****" << std::endl;
        else
          std::cout << std::endl;
      }
      phi += 9;
    }
    ring += 3;
  }
}

void testTriggerCell(int type) {
  int N = (type == 0) ? HGCSiliconDetId::HGCalFineN : HGCSiliconDetId::HGCalCoarseN;
  const int waferu(0), waferv(0), layer(1);
  std::string error[2] = {"ERROR", "OK"};
  int ntot(0), nerror(0);
  for (int iz = 0; iz < 2; ++iz) {
    int zside = 2 * iz - 1;
    for (int u = 0; u < 2 * N; ++u) {
      for (int v = 0; v < 2 * N; ++v) {
        if (((v - u) < N) && (u - v) <= N) {
          HGCSiliconDetId id(DetId::HGCalEE, zside, type, layer, waferu, waferv, u, v);
          std::cout << "ID " << std::hex << id.rawId() << std::dec << " " << id << " Trigger: " << id.triggerCellU()
                    << ":" << id.triggerCellV() << std::endl;
          HGCalTriggerDetId idt((int)(HGCalEETrigger),
                                id.zside(),
                                id.type(),
                                id.layer(),
                                id.waferU(),
                                id.waferV(),
                                id.triggerCellU(),
                                id.triggerCellV());
          int ok(0);
          std::vector<std::pair<int, int> > uvs = idt.cellUV();
          for (auto const& uv : uvs) {
            HGCSiliconDetId idn(
                DetId::HGCalEE, idt.zside(), idt.type(), idt.layer(), idt.waferU(), idt.waferV(), uv.first, uv.second);
            if (idn == id) {
              ok = 1;
              break;
            }
          }
          std::cout << "Trigger Cell: " << idt << " obtained from cell (" << error[ok] << ")" << std::endl;
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

void testROC() {
  HGCSiliconDetIdToROC idToROC;
  idToROC.print();
  for (int type = 0; type < 2; ++type) {
    int kmax = (type == 0) ? 6 : 3;
    for (int k = 1; k <= kmax; ++k) {
      auto cells = idToROC.getTriggerId(k, type);
      bool error(false);
      std::cout << "ROC " << type << ":" << k << " has " << cells.size() << " trigger cells:";
      unsigned int i(0);
      for (auto cell : cells) {
        int k0 = idToROC.getROCNumber(cell.first, cell.second, type);
        std::cout << " [" << i << "] (" << cell.first << "," << cell.second << "):" << k0;
        ++i;
        if (k0 != k)
          error = true;
      }
      if (error)
        std::cout << " ***** ERROR *****" << std::endl;
      else
        std::cout << std::endl;
    }
  }
}

void testModule(HGCSiliconDetId const& id) {
  HGCSiliconDetIdToModule hgc;
  HGCSiliconDetId module = hgc.getModule(id);
  std::vector<HGCSiliconDetId> ids = hgc.getDetIds(module);
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
  testCell(1);
  testWafer(1, 319.80, 1544.30);
  testWafer(28, 352.46, 1658.68);
  testScint(10);
  testScint(22);
  testTriggerCell(0);
  testTriggerCell(1);
  testROC();
  testModule(HGCSiliconDetId(DetId::HGCalEE, 1, 0, 1, 5, 4, 0, 10));
  testModule(HGCSiliconDetId(DetId::HGCalHSi, -1, 1, 30, -6, -4, 5, 5));
  return 0;
}
