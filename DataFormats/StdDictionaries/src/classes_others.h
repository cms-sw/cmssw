#include <bitset>
#include <deque>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <functional>

namespace DataFormats_StdDictionaries {
  struct dictionaryothers {
  std::allocator<char> achar;
  std::allocator<double> adouble;
  std::allocator<int> aint;
  std::allocator<short> ashort;
  std::basic_string<char> bschar;
  std::bidirectional_iterator_tag bidirectiter;
  std::bitset<6> dummybitset6;
  std::bitset<7> dummybitset7;
  std::bitset<15> dummybitset15;
  std::deque<int> dummy18;
  std::forward_iterator_tag fowitertag;
  std::input_iterator_tag initertag;
  std::less<int> lsint;
  std::list<int> dummy17;
  std::multimap<double,double> dummyypwmv6;
  std::output_iterator_tag outitertag;
  std::random_access_iterator_tag randaccitertag;
  std::set<int> dummy19;
  std::set<std::basic_string<char> > dummy20;
  std::binary_function<int,int,bool> dummy25;
  };
}
