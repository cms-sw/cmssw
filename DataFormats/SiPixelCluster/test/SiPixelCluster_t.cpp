#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"


typedef SiPixelCluster::PixelPos PiPos;
typedef SiPixelCluster::Pixel Pixel;

inline
bool verify(PiPos (const &pos)[]) {

  SiPixelCluster clus;
  for (auto p : pos)
    clus.add(p,2);

}


int main() {


  bool ok=true;

  PiPos const normal[] = { {3,3}, {3,4}, {3,5}, {5,4} ,{4,4}, {5,5} };
  PiPos const big[] = { {3,3}, {3,132}, {3,5}, {201,4} ,{212,155}, {122,5} };
  PiPos const ylarge[] = { {3,322}, {3,332}, {3,400}, {201,400} ,{212,323}, {122,350} };
  PiPos const huge[] = { {3,3}, {3,332}, {3,400}, {201,400} ,{212,323}, {122,350} };

  ok &=verify(normal);

    return ok ? 0 : 1;
}
