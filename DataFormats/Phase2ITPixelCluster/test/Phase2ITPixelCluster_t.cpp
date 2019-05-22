#include "DataFormats/Phase2ITPixelCluster/interface/Phase2ITPixelCluster.h"
#include <cassert>
#include <cstdio>

typedef Phase2ITPixelCluster::PixelPos PiPos;
typedef Phase2ITPixelCluster::Pixel Pixel;

template <int N>
inline bool verify(PiPos const (&pos)[N], bool ox, bool oy) {
  bool ok = true;
  Phase2ITPixelCluster clus;
  for (auto p : pos)
    clus.add(p, 2);
  printf("\nclus  %d  %d,%d %d,%d %s %s\n\n",
         clus.size(),
         clus.minPixelRow(),
         clus.maxPixelRow(),
         clus.minPixelCol(),
         clus.maxPixelCol(),
         clus.overflowCol() ? "overflowY " : " ",
         clus.overflowRow() ? " overflowX" : "");

  // auto cxmin = clus.minPixelRow();
  // auto cymin = clus.minPixelCol();
  ok &= (ox == clus.overflowRow()) && (clus.overflowCol() == oy);

  // verify new constructor
  uint32_t adc[16]{2};
  uint32_t x[16];
  uint32_t y[16];
  uint32_t xmin = 16000;
  uint32_t ymin = 16000;
  unsigned int isize = 0;
  for (auto p : pos) {
    xmin = std::min(xmin, (uint32_t)(p.row()));
    ymin = std::min(ymin, (uint32_t)(p.col()));
    x[isize] = p.row();
    y[isize++] = p.col();
  }
  printf("pos  %d  %d,%d\n", isize, xmin, ymin);
  Phase2ITPixelCluster clus2(isize, adc, x, y, xmin, ymin);
  printf("clus2 %d  %d,%d %d,%d %s %s\n\n",
         clus2.size(),
         clus2.minPixelRow(),
         clus2.maxPixelRow(),
         clus2.minPixelCol(),
         clus2.maxPixelCol(),
         clus2.overflowCol() ? "overflowY " : " ",
         clus2.overflowRow() ? " overflowX" : "");

  ok &= (clus.size() == clus2.size());
  ok &= (clus.pixelOffset() == clus2.pixelOffset());
  for (int i = 0; i != clus.size(); ++i) {
    auto const p = clus.pixel(i);
    auto const p2 = clus2.pixel(i);
    // EM 2016.06.14: comment next two lines
    // while waiting for an answer from experts
    // https://hypernews.cern.ch/HyperNews/CMS/get/pixelOfflineSW/1231.html
    //    ok &=  (pos[i].row()-cxmin>127) ? true : false; //? p.x==127+cxmin : p.x==pos[i].row();
    //    ok &=  (pos[i].col()-cymin>127) ? true : false; //? p.y==127+cymin : p.y==pos[i].col();
    printf("%d,%d %d,%d %d,%d\n", pos[i].row(), pos[i].col(), p.x, p.y, p2.x, p2.y);
  }

  return ok;
}

int main() {
  bool ok = true;

  PiPos const normal[] = {{3, 3}, {3, 4}, {3, 5}, {5, 4}, {4, 7}, {5, 5}};
  // EM 2016.06.14: comment test of "BigPixels"
  // PiPos const bigX[] = { {3,3}, {3,60}, {3,5}, {161,4} ,{162,62}, {162,5} };
  // PiPos const bigY[] = { {3,3}, {3,100}, {3,5}, {61,234} ,{62,102}, {45,65} };
  // PiPos const ylarge[] = { {3,352}, {3,352}, {3,400}, {20,400} ,{40,363}, {62,350} };
  // PiPos const huge[] = { {3,3}, {3,332}, {3,400}, {201,400} ,{212,323}, {122,350} };

  ok &= verify(normal, false, false);
  assert(ok);
  // EM 2016.06.14: comment test of "BigPixels"
  // ok &=verify(bigX,true,false);
  // assert(ok);
  // ok &=verify(bigY,false,true);
  // assert(ok);
  // ok &=verify(ylarge,false,false);
  // assert(ok);
  // ok &=verify(huge,true,true);
  // assert(ok);

  return ok ? 0 : 1;
}
