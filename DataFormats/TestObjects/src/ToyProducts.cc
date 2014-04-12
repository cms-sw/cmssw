/*----------------------------------------------------------------------

Toy EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include "DataFormats/TestObjects/interface/ToyProducts.h"

namespace edmtest {

    Simple::~Simple() {}
    Simple* Simple::clone() const { return new Simple(*this); }

    SimpleDerived::~SimpleDerived() {}
    SimpleDerived* SimpleDerived::clone() const { return new SimpleDerived(*this); }
}
