#ifndef CSCIdentifier_h
#define CSCIdentifier_h

#include <vector>

class CSCIdentifier{
 public:
  CSCIdentifier();
  ~CSCIdentifier();
  
 struct Item{
   int CSCid;
 };

  std::vector<Item> identifier;
};

#endif

