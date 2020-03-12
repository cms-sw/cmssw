#ifndef Fireworks_Core_TEveElementIter_h
#define Fireworks_Core_TEveElementIter_h
//
//  Description: dumb tree iterator with optional perl style regular expression
//               filter
//
//  Original Author: D.Kovalskyi
//

#include "TPRegexp.h"
#include <vector>

class TEveElement;

class TEveElementIter {
  TPRegexp regexp;
  std::vector<TEveElement*> elements;
  std::vector<TEveElement*>::iterator iter;

public:
  TEveElementIter(TEveElement*, const char* regular_expression = nullptr);
  TEveElement* next();
  TEveElement* current();
  TEveElement* reset();
  unsigned int size() { return elements.size(); }

private:
  void addElement(TEveElement*);
};
#endif
