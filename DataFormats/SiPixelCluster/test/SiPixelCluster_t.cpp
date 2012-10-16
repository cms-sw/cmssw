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
  printf("\nclus  %d  %d,%d %d,%d %s %s\n\n",clus.size(), clus.minPixelRow(),clus.maxPixelRow(),clus.minPixelCol(),clus.maxPixelCol(), 
	 clus.overflowCol() ? "overflowY ":" ",clus.overflowRow() ? " overflowX":"");  
  
  auto cxmin = clus.minPixelRow();
  auto cymin = clus.minPixelCol();
  ok &= (ox==clus.overflowRow()) && (clus.overflowCol()==oy);

  // verify new constructor
  unsigned short adc[16]{2};
  unsigned short x[16];
  unsigned short y[16];
  unsigned short xmin=16000;
  unsigned short ymin=16000;
  unsigned int isize=0;
  for (auto p : pos) {
    xmin=std::min(xmin,(unsigned short)(p.row()));
    ymin=std::min(ymin,(unsigned short)(p.col()));
    x[isize]=p.row();
    y[isize++]=p.col();
  }
  printf("pos  %d  %d,%d\n", isize, xmin, ymin);
  SiPixelCluster clus2(isize,adc,x,y,xmin,ymin);
  printf("clus2 %d  %d,%d %d,%d %s %s\n\n",clus2.size(), clus2.minPixelRow(),clus2.maxPixelRow(),clus2.minPixelCol(),clus2.maxPixelCol(), 
	 clus2.overflowCol() ? "overflowY ":" ",clus2.overflowRow() ? " overflowX":"");  

  ok &= (clus.size()==clus2.size());
  ok &= (clus.pixelOffset()==clus2.pixelOffset());
 for (int i=0; i!=clus.size(); ++i) {
    auto const p = clus.pixel(i);
    auto const p2 = clus2.pixel(i);
      ok &=  (pos[i].row()-cxmin>63) ? p.x==63+cxmin : p.x==pos[i].row(); 
      ok &=  (pos[i].col()-cymin>63) ? p.y==63+cymin : p.y==pos[i].col();
      printf("%d,%d %d,%d %d,%d\n",pos[i].row(),pos[i].col(), p.x,p.y, p2.x,p2.y);
  }

  return ok;

}


int main() {


  bool ok=true;

  PiPos const normal[] = { {3,3}, {3,4}, {3,5}, {5,4} ,{4,7}, {5,5} };
  PiPos const bigX[] = { {3,3}, {3,60}, {3,5}, {161,4} ,{162,62}, {162,5} };
  PiPos const bigY[] = { {3,3}, {3,100}, {3,5}, {61,234} ,{62,102}, {45,65} };
  PiPos const ylarge[] = { {3,352}, {3,352}, {3,400}, {20,400} ,{40,363}, {62,350} };
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
