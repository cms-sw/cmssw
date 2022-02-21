#include "Rtypes.h"
#include "vector"

#ifdef __CINT__
#pragma link C++ class vector < vector < int>> + ;
#pragma link C++ class vector < vector < unsigned int>> + ;
#pragma link C++ class vector < vector < float>> + ;
#ifdef G__VECTOR_HAS_CLASS_ITERATOR
#pragma link C++ operators vector < vector < int>> ::iterator;
#pragma link C++ operators vector < vector < int>> ::const_iterator;
#pragma link C++ operators vector < vector < int>> ::reverse_iterator;

#pragma link C++ operators vector < vector < unsigned int>> ::iterator;
#pragma link C++ operators vector < vector < unsigned int>> ::const_iterator;
#pragma link C++ operators vector < vector < unsigned int>> ::reverse_iterator;

#pragma link C++ operators vector < vector < float>> ::iterator;
#pragma link C++ operators vector < vector < float>> ::const_iterator;
#pragma link C++ operators vector < vector < float>> ::reverse_iterator;
#endif
#endif
