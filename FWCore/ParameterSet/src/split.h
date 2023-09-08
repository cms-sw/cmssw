#ifndef ParameterSet_split_h
#define ParameterSet_split_h

// ----------------------------------------------------------------------
// definition of split() and related templates
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// prolog

// ----------------------------------------------------------------------
// prerequisite source files and headers

#include <string_view>

// ----------------------------------------------------------------------
// contents

namespace edm {

  template <class FwdIter>
  FwdIter contextual_find(FwdIter b, FwdIter e, char first, char sep, char last);

  template <class FwdIter>
  FwdIter contextual_find_not(FwdIter b, FwdIter e, char first, char sep, char last);

  template <class OutIter>
  bool split(OutIter result, std::string_view string_to_split, char first, char sep, char last);

}  // namespace edm

// ----------------------------------------------------------------------
// contextual_find

template <class FwdIter>
FwdIter edm::contextual_find(FwdIter b, FwdIter e, char first, char sep, char last) {
  for (int nested = 0; b != e; ++b) {
    if (*b == first)
      ++nested;
    else if (*b == last)
      --nested;
    else if (*b == sep && nested == 0)
      return b;
  }

  return e;

}  // contextual_find()

// ----------------------------------------------------------------------
// contextual_find_not

template <class FwdIter>
FwdIter edm::contextual_find_not(FwdIter b, FwdIter e, char /* first */, char sep, char /* last */) {
  for (; b != e; ++b) {
    if (*b != sep)
      return b;
  }

  return e;

}  // contextual_find_not()

// ----------------------------------------------------------------------
// split()

template <class OutIter>
bool edm::split(OutIter dest, std::string_view s, char first, char sep, char last) {
  using str_c_iter = std::string_view::const_iterator;
  str_c_iter b = s.cbegin(), e = s.cend();

  if (static_cast<unsigned int>(e - b) < 2u)
    return false;

  if (*b == first)
    ++b;
  else
    return false;

  if (*--e != last)
    return false;

  // invariant:  we've found all items in [b..boi)
  for (str_c_iter  //boi = std::find_if(b, e, is_not_a(sep))
           boi = contextual_find_not(b, e, first, sep, last),
           eoi;
       boi != e
       //; boi = std::find_if(eoi, e, is_not_a(sep))
       ;
       boi = contextual_find_not(eoi, e, first, sep, last)) {
    // find end of current item:
    //eoi = std::find_if(boi, e, is_a(sep));
    eoi = contextual_find(boi, e, first, sep, last);

    // copy the item formed from characters in [boi..eoi):
    *dest++ = std::string_view(boi, eoi - boi);
  }  // for

  return true;
}  // split< >()

// ----------------------------------------------------------------------
// epilog

#endif
