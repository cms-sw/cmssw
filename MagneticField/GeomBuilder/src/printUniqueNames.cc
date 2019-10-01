#include "printUniqueNames.h"

#include <iostream>
#include <iterator>
#include <iomanip>

using namespace std;

void magneticfield::printUniqueNames(handles::const_iterator begin, handles::const_iterator end, bool uniq) {
  std::vector<std::string> names;
  for (handles::const_iterator i = begin; i != end; ++i) {
    if (uniq)
      names.push_back((*i)->name);
    else
      names.push_back((*i)->name + ":" + std::to_string((*i)->copyno));
  }

  sort(names.begin(), names.end());
  if (uniq) {
    std::vector<std::string>::iterator i = unique(names.begin(), names.end());
    int nvols = int(i - names.begin());
    cout << nvols << " ";
    copy(names.begin(), i, ostream_iterator<std::string>(cout, " "));
  } else {
    cout << names.size() << " ";
    copy(names.begin(), names.end(), ostream_iterator<std::string>(cout, " "));
  }
  cout << endl;
}
