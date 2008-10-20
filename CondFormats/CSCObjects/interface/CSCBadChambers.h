#ifndef CSCBadChambers_h
#define CSCBadChambers_h

#include <vector>

class CSCBadChambers{
 public:
  CSCBadChambers();
  ~CSCBadChambers();
  
  std::vector<int> chambers() const;

 private:
  std::vector<int> theChambers;
};

#endif
