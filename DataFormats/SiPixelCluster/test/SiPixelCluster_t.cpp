#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include<cassert>
#include<cstdio>

typedef SiPixelCluster::PixelPos PiPos;
typedef SiPixelCluster::Pixel Pixel;

template<int N>
inline
bool verify(PiPos const (&pos)[N]) {

  bool ok=true;
  SiPixelCluster clus;
  for (auto p : pos)
    clus.add(p,2);
  printf("\n%d,%d %d,%d %s %d\n\n",clus.minPixelRow(),clus.maxPixelRow(),clus.minPixelCol(),clus.maxPixelCol(), clus.overflowCol() ? "overflow":"", clus.computeColSpan());  

 for (int i=0; i!=clus.size(); ++i) {
    auto const p = clus.pixel(i);
    ok &= p.x==pos[i].row() && p.y==pos[i].col();
    printf("%d,%d %d,%d\n",pos[i].row(),pos[i].col(), p.x,p.y);
  }

  return ok;

}


int main() {


  bool ok=true;

  PiPos const normal[] = { {3,3}, {3,4}, {3,5}, {5,4} ,{4,4}, {5,5} };
  PiPos const big[] = { {3,3}, {3,132}, {3,5}, {201,4} ,{212,155}, {122,5} };
  PiPos const ylarge[] = { {3,322}, {3,332}, {3,400}, {201,400} ,{212,323}, {122,350} };
  PiPos const huge[] = { {3,150}, {3,332}, {3,400}, {201,400} ,{212,323}, {122,350} };

  ok &=verify(normal);
  assert(ok);
  ok &=verify(big);
  assert(ok);
  ok &=verify(ylarge);
  assert(ok);
  ok &=verify(huge);
  assert(ok);

  return ok ? 0 : 1;
}
