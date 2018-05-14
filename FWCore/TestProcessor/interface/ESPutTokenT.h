#ifndef Framework_TestProcessor_ESPutTokenT_h
#define Framework_TestProcessor_ESPutTokenT_h
// -*- C++ -*-
//
// Package:     Framework/TestProcessor
// Class  :     ESPutTokenT
// 
/**\class ESPutTokenT ESPutTokenT.h "ESPutTokenT.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue, 08 May 2018 19:46:32 GMT
//

// system include files

// user include files

// forward declarations
namespace edm {
namespace test {
class TestProcessorConfig;
  
template <typename T>
class ESPutTokenT {
public:
  friend class TestProcessorConfig;
  ESPutTokenT(): index_{undefinedIndex()} {}
  
  int index() const { return index_;}
  
  static int undefinedIndex() { return -1;}
private:
  ESPutTokenT(int iIndex): index_{iIndex} {};
  
  int index_;
};

}
}

#endif
