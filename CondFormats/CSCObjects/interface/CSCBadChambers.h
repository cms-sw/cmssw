#ifndef CSCBadChambers_h
#define CSCBadChambers_h

#include <vector>

class CSCBadChambers{
 public:
  CSCBadChambers();
  ~CSCBadChambers();

  int numberOfBadChambers;
  std::vector<int> chambers;
};

#endif
