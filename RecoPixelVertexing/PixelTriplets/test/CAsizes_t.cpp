#include "RecoPixelVertexing/PixelTriplets/plugins/GPUCACell.h"

#include <typeinfo>
#include <iostream>

template <typename T>
void print() {
  std::cout << "size of " << typeid(T).name() << ' ' << sizeof(T) << std::endl;
}

int main() {
  using namespace CAConstants;

  print<GPUCACell>();
  print<CellNeighbors>();
  print<CellTracks>();
  print<OuterHitOfCell>();
  print<TuplesContainer>();
  print<HitToTuple>();
  print<TupleMultiplicity>();

  print<CellNeighborsVector>();

  return 0;
}
