//  (C) Copyright Ben Bear, Herve Bronnimann 2007.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/*
 Revision history:
Nov 13, 2007:  Incorporation of boost-devel comments (Jens Seidel, Ben Bear and myself)
Nov 11, 2007:  Rewrite of Ben Bear's Gacap
Dec 27, 2018:  Removed code not used, changed namespace to notboost since this was never incorporated in boost by cdj
*/

#ifndef NOTBOOST_ALGORITHM_COMBINATION_HPP
#define NOTBOOST_ALGORITHM_COMBINATION_HPP

#include <algorithm>

namespace notboost {

  namespace detail {

    template <class BidirectionalIterator>
    bool next_combination(BidirectionalIterator first1,
                          BidirectionalIterator last1,
                          BidirectionalIterator first2,
                          BidirectionalIterator last2) {
      if ((first1 == last1) || (first2 == last2)) {
        return false;
      }

      BidirectionalIterator m1 = last1;
      BidirectionalIterator m2 = last2;
      --m2;

      // Find first m1 not less than *m2 (i.e., lower_bound(first1, last1, *m2)).
      // Actually, this loop stops at one position before that, except perhaps
      // if m1 == first1 (in which case one must compare *first1 with *m2).
      while (--m1 != first1 && !(*m1 < *m2)) {
      }

      // Test if all elements in [first1, last1) not less than *m2.
      bool result = (m1 == first1) && !(*first1 < *m2);

      if (!result) {
        // Find first first2 greater than *m1 (since *m1 < *m2, we know it
        // can't pass m2 and there's no need to test m2).
        while (first2 != m2 && !(*m1 < *first2)) {
          ++first2;
        }

        first1 = m1;
        std::iter_swap(first1, first2);
        ++first1;
        ++first2;
      }

      // Merge [first1, last1) with [first2, last2), given that the rest of the
      // original ranges are sorted and compare appropriately.
      if ((first1 != last1) && (first2 != last2)) {
        for (m1 = last1, m2 = first2; (m1 != first1) && (m2 != last2); ++m2) {
          std::iter_swap(--m1, m2);
        }

        std::reverse(first1, m1);
        std::reverse(first1, last1);

        std::reverse(m2, last2);
        std::reverse(first2, last2);
      }

      return !result;
    }

  }  // namespace detail

  /* PROPOSED STANDARD EXTENSIONS:
 *
 * template<class BidirectionalIterator> 
 *   bool next_combination(BidirectionalIterator first,
 *                         BidirectionalIterator middle,
 *                         BidirectionalIterator last); 
 *
 */

  template <class BidirectionalIterator>
  bool next_combination(BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last) {
    return detail::next_combination(first, middle, middle, last);
  }

}  // namespace notboost

#endif  // NOTBOOST_ALGORITHM_COMBINATION_HPP
