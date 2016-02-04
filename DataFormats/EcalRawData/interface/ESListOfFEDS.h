#ifndef RAWECAL_ESListOfFEDS
#define RAWECAL_ESListOfFEDS

#include <vector>

class ESListOfFEDS {

 public:
	ESListOfFEDS();
	void AddFED(int fed);
	std::vector<int> GetList() const { return list_of_feds; }
	void SetList(std::vector<int>& feds) { list_of_feds = feds; }

 private:
	std::vector<int> list_of_feds;
};

typedef std::vector<ESListOfFEDS> ESListOfFEDSCollection;


#endif
