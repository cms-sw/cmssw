#ifndef RAWECAL_ListOfFEDS
#define RAWECAL_ListOfFEDS

#include <vector>

class EcalListOfFEDS {
public:
  EcalListOfFEDS();
  void AddFED(int fed);
  std::vector<int> GetList() const { return list_of_feds; }
  void SetList(std::vector<int>& feds) { list_of_feds = feds; }

private:
  std::vector<int> list_of_feds;
};

typedef std::vector<EcalListOfFEDS> EcalListOfFEDSCollection;

#endif
