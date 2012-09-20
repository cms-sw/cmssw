#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include<cassert>
#include<cstdio>

typedef SiPixelCluster::PixelPos PiPos;
typedef SiPixelCluster::Pixel Pixel;

template<int N>
inline
bool verify(PiPos const (&pos)[N], bool ox, bool oy) {

  bool ok=true;
  SiPixelCluster clus;
  for (auto p : pos)
    clus.add(p,2);
  printf("\n%d,%d %d,%d %s %s\n\n",clus.minPixelRow(),clus.maxPixelRow(),clus.minPixelCol(),clus.maxPixelCol(), 
	 clus.overflowCol() ? "overflowY ":" ",clus.overflowRow() ? " overflowX":"");  
  
  auto xmin = clus.minPixelRow();
  auto ymin = clus.minPixelCol();
  ok &= (ox==clus.overflowRow()) && (clus.overflowCol()==oy);

 for (int i=0; i!=clus.size(); ++i) {
    auto const p = clus.pixel(i);
      ok &=  (pos[i].row()-xmin>127) ? p.x==127+xmin : p.x==pos[i].row(); 
      ok &=  (pos[i].col()-ymin>127) ? p.y==127+ymin : p.y==pos[i].col();
    printf("%d,%d %d,%d\n",pos[i].row(),pos[i].col(), p.x,p.y);
  }

  return ok;

}


int main() {


  bool ok=true;

  PiPos const normal[] = { {3,3}, {3,4}, {3,5}, {5,4} ,{4,4}, {5,5} };
  PiPos const bigX[] = { {3,3}, {3,100}, {3,5}, {201,4} ,{212,102}, {122,5} };
  PiPos const bigY[] = { {3,3}, {3,100}, {3,5}, {101,234} ,{112,102}, {45,65} };
  PiPos const ylarge[] = { {3,322}, {3,332}, {3,400}, {70,400} ,{40,323}, {72,350} };
  PiPos const huge[] = { {3,3}, {3,332}, {3,400}, {201,400} ,{212,323}, {122,350} };

  ok &=verify(normal,false,false);
  assert(ok);
  ok &=verify(bigX,true,false);
  assert(ok);
  ok &=verify(bigY,false,true);
  assert(ok);
  ok &=verify(ylarge,false,false);
  assert(ok);
  ok &=verify(huge,true,true);
  assert(ok);

  return ok ? 0 : 1;
}
