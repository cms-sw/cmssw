#include "CondFormats/Common/interface/SmallWORMDict.h"

#include <iostream>

namespace test {
  namespace SmallWORMDict {
    int test() {
      std::vector<std::string> dict;
      size_t tot = 0;
      dict.push_back("Sneezy");
      tot += dict.back().size();
      dict.push_back("Sleepy");
      tot += dict.back().size();
      dict.push_back("Dopey");
      tot += dict.back().size();
      dict.push_back("Doc");
      tot += dict.back().size();
      dict.push_back("Happy");
      tot += dict.back().size();
      dict.push_back("Bashful");
      tot += dict.back().size();
      dict.push_back("Grumpy");
      tot += dict.back().size();

      cond::SmallWORMDict worm(dict);

      std::cout << dict.size() << " " << worm.m_index.size() << std::endl;
      std::cout << tot << " " << worm.m_data.size() << std::endl;
      std::copy(worm.m_index.begin(), worm.m_index.end(), std::ostream_iterator<int>(std::cout, " "));
      std::cout << std::endl;
      std::copy(worm.m_data.begin(), worm.m_data.end(), std::ostream_iterator<char>(std::cout, ""));
      std::cout << std::endl;

      cond::SmallWORMDict::Frame f = worm[2];
      std::copy(f.b, f.b + f.l, std::ostream_iterator<char>(std::cout, ""));
      std::cout << std::endl;

      int i = worm.index("Doc");
      f = worm[i];
      std::copy(f.b, f.b + f.l, std::ostream_iterator<char>(std::cout, ""));
      std::cout << " at " << i << std::endl;

      f = *worm.find("Sleepy");
      std::copy(f.b, f.b + f.l, std::ostream_iterator<char>(std::cout, ""));
      std::cout << " at " << f.ind << std::endl;

      return 0;
    }
  }  // namespace SmallWORMDict
}  // namespace test

int main() { return test::SmallWORMDict::test(); }
