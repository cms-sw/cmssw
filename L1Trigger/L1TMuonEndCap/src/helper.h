#include <bitset>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <memory>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Need a safe assertion function
#ifdef NDEBUG
# define assert_no_abort(expr) ((void)0)
#else
# define assert_no_abort(expr) ((void)((expr) || (__assert_no_abort(#expr, __FILE__, __LINE__, __PRETTY_FUNCTION__),0)))
template<typename T=void>
void __assert_no_abort(const char *assertion, const char *file, unsigned int line, const char * function) {
  //std::cout << file << ":" << line << ": " << function << ": Assertion `" << assertion << "' failed. (no abort)" << std::endl;
  edm::LogWarning("L1T") << file << ":" << line << ": " << function << ": Assertion `" << assertion << "' failed. (no abort)";
}
#endif


namespace {

  // Return an integer as a hex string
  template<typename INT>
  std::string to_hex(INT i) {
    std::stringstream s;
    s << "0x" << std::hex << i;
    return s.str();
  }

  // Return an integer as a binary string
  template<typename INT>
  std::string to_binary(INT i, int n) {
    std::stringstream s;
    if (sizeof(i) <= 4) {
      std::bitset<32> b(i);
      s << "0b" << b.to_string().substr(32-n,32);
    } else if (sizeof(i) <= 8) {
      std::bitset<64> b(i);
      s << "0b" << b.to_string().substr(64-n,64);
    }
    return s.str();
  }

  // Return the size of a 1D plain array
  template<typename T, size_t N>
  constexpr size_t array_size(T(&)[N]) { return N; }

  // Return the elements of a 1D plain array as a string (elements are separated by ' ')
  template<typename T, size_t N>
  std::string array_as_string(const T(&arr)[N]) {
    std::stringstream s;
    const char* sep = "";
    for (size_t i=0; i<N; ++i) {
      s << sep << arr[i];
      sep = " ";
    }
    return s.str();
  }

  // This function allows one to loop over a container in reversed order using C++11 for(auto ...) loop
  // e.g.
  //   for (auto x: reversed(v)) {
  //     // do something
  //   }
  // See http://stackoverflow.com/a/21510185
  namespace details {
    template <class T> struct _reversed {
      T& t; _reversed(T& _t): t(_t) {}
      decltype(t.rbegin()) begin() { return t.rbegin(); }
      decltype(t.rend()) end() { return t.rend(); }
    };
  }
  template <class T> details::_reversed<T> reversed(T& t) { return details::_reversed<T>(t); }

  // Split a string by delimiters (default: ' ') into a vector of string
  // See http://stackoverflow.com/a/53878
  template <class STR=std::string>
  std::vector<STR> split_string(const std::string& s, char c = ' ', char d = ' ') {
    std::vector<STR> result;
    const char* str = s.c_str();
    do {
      const char* begin = str;
      while(*str != c && *str != d && *str)
        str++;
      result.emplace_back(begin, str);
    } while (0 != *str++);
    return result;
  }

  // Flatten a vector<vector<T> > into a vector<T>
  // The input type T can be different from the output type T
  template <class T1, class T2>
  void flatten_container(const T1& input, T2& output) {
    typename T1::const_iterator it;
    for (it = input.begin(); it != input.end(); ++it) {
      output.insert(output.end(), it->begin(), it->end());
    }
  }

  // Check type for map of vector
  template<typename>
  struct is_map_of_vectors : public std::false_type { };

  template<typename T1, typename T2>
  struct is_map_of_vectors<std::map<T1, std::vector<T2> > > : public std::true_type { };

  // Merge a map of vectors (map1) into another map of vectors (map2)
  template<typename Map>
  void merge_map_into_map(const Map& map1, Map& map2) {
    // This is customized for maps of containers.
    typedef typename Map::iterator Iterator;
    typedef typename Map::mapped_type Container;

    for (auto& kv1 : map1) {
      std::pair<Iterator,bool> ins = map2.insert(kv1);
      if (!ins.second) {  // if insertion into map2 was not successful
        if (is_map_of_vectors<Map>::value) {  // special case for map of vectors
          const Container* container1 = &(kv1.second);
          Container* container2 = &(ins.first->second);
          container2->insert(container2->end(), container1->begin(), container1->end());
        } // else do nothing
      }
    }
  }

  // A simple nearest-neighbor clustering algorithm
  // It iterates through a sorted container once, whenever the 'adjacent'
  // comparison between two elements evaluates to true, the 'cluster'
  // operator is called to merge them.
  template <class ForwardIt, class BinaryPredicate, class BinaryOp>
  ForwardIt adjacent_cluster(ForwardIt first, ForwardIt last, BinaryPredicate adjacent, BinaryOp cluster) {
    if (first == last) return last;

    ForwardIt result = first;
    while (++first != last) {
      if (!adjacent(*result, *first)) {
        *++result = std::move(*first);
      } else {
        cluster(*result, *first);
      }
    }
    return ++result;
  }

  // Textbook merge sort algorithm with the same interface as std::sort()
  // An internal buffer of the same size as the container is used internally.
  template<typename RandomAccessIterator, typename Compare = std::less<> >
  void merge_sort_merge(RandomAccessIterator first, RandomAccessIterator middle, RandomAccessIterator last, Compare cmp)
  {
    const std::ptrdiff_t len = std::distance(first, last);
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type value_type;
    typedef typename std::iterator_traits<RandomAccessIterator>::pointer pointer;
    std::pair<pointer, std::ptrdiff_t> p = std::get_temporary_buffer<value_type>(len);
    pointer buf = p.first;
    pointer buf_end = std::next(p.first, p.second);

    RandomAccessIterator first1 = first;
    RandomAccessIterator last1  = middle;
    RandomAccessIterator first2 = middle;
    RandomAccessIterator last2  = last;

    while (first1 != last1 && first2 != last2) {
      if (cmp(*first2, *first1)) {
        *buf++ = *first2++;
      } else {
        *buf++ = *first1++;
      }
    }
    while (first1 != last1) {
      *buf++ = *first1++;
    }
    while (first2 != last2) {
      *buf++ = *first2++;
    }

    buf = p.first;
    std::copy(buf, buf_end, first);
    std::return_temporary_buffer(p.first);
  }

  // See above
  template<typename RandomAccessIterator, typename Compare = std::less<> >
  void merge_sort(RandomAccessIterator first, RandomAccessIterator last, Compare cmp)
  {
    const std::ptrdiff_t len = std::distance(first, last);
    if (len > 1) {
      RandomAccessIterator middle = std::next(first, len / 2);
      merge_sort(first, middle, cmp);
      merge_sort(middle, last, cmp);
      merge_sort_merge(first, middle, last, cmp);
    }
  }

  // An extended version of the merge sort algorithm to incorporate a 3-way
  // comparator. It resorts back to 2-way comparator when one of the three
  // lists to be merged is empty.
  template<typename RandomAccessIterator, typename Compare, typename Compare3>
  void merge_sort_merge3(RandomAccessIterator first, RandomAccessIterator one_third, RandomAccessIterator two_third, RandomAccessIterator last, Compare cmp, Compare3 cmp3)
  {
    const std::ptrdiff_t len = std::distance(first, last);
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type value_type;
    typedef typename std::iterator_traits<RandomAccessIterator>::pointer pointer;
    std::pair<pointer, std::ptrdiff_t> p = std::get_temporary_buffer<value_type>(len);
    pointer buf = p.first;
    pointer buf_end = std::next(p.first, p.second);

    RandomAccessIterator first1 = first;
    RandomAccessIterator last1  = one_third;
    RandomAccessIterator first2 = one_third;
    RandomAccessIterator last2  = two_third;
    RandomAccessIterator first3 = two_third;
    RandomAccessIterator last3  = last;

    while (first1 != last1 && first2 != last2 && first3 != last3) {
      int rr = cmp3(*first1, *first2, *first3);
      if (rr == 0) {
        *buf++ = *first1++;
      } else if (rr == 1) {
        *buf++ = *first2++;
      } else if (rr == 2) {
        *buf++ = *first3++;
      }
    }

    if (first3 == last3) {
      // do nothing
    } else if (first2 == last2) {
      first2 = first3;
      last2  = last3;
    } else if (first1 == last1) {
      first1 = first2;
      last1  = last2;
      first2 = first3;
      last2  = last3;
    }

    while (first1 != last1 && first2 != last2) {
      if (cmp(*first2, *first1)) {
        *buf++ = *first2++;
      } else {
        *buf++ = *first1++;
      }
    }
    while (first1 != last1) {
      *buf++ = *first1++;
    }
    while (first2 != last2) {
      *buf++ = *first2++;
    }

    buf = p.first;
    std::copy(buf, buf_end, first);
    std::return_temporary_buffer(p.first);
  }

  // See above
  template<typename RandomAccessIterator, typename Compare, typename Compare3>
  void merge_sort3(RandomAccessIterator first, RandomAccessIterator last, Compare cmp, Compare3 cmp3)
  {
    const std::ptrdiff_t len = std::distance(first, last);
    if (len > 1) {
      RandomAccessIterator one_third = std::next(first, (len+2) / 3);
      RandomAccessIterator two_third = std::next(first, (len+2) / 3 * 2);
      merge_sort3(first, one_third, cmp, cmp3);
      merge_sort3(one_third, two_third, cmp, cmp3);
      merge_sort3(two_third, last, cmp, cmp3);
      merge_sort_merge3(first, one_third, two_third, last, cmp, cmp3);
    }
  }

  // See above. 'Hint' is provided to force the very first division. This is needed to match FW.
  template<typename RandomAccessIterator, typename Compare, typename Compare3>
  void merge_sort3_with_hint(RandomAccessIterator first, RandomAccessIterator last, Compare cmp, Compare3 cmp3, std::ptrdiff_t d)
  {
    const std::ptrdiff_t len = std::distance(first, last);
    if (len > 1) {
      RandomAccessIterator one_third = std::next(first, d);
      RandomAccessIterator two_third = std::next(first, d * 2);
      merge_sort3(first, one_third, cmp, cmp3);
      merge_sort3(one_third, two_third, cmp, cmp3);
      merge_sort3(two_third, last, cmp, cmp3);
      merge_sort_merge3(first, one_third, two_third, last, cmp, cmp3);
    }
  }

}  // namespace
