#include "__class__.h"

// __class__ ctor
__class__::__class__()
{
}

// __class__ dtor
__class__::~__class__()
{
}

// __class__ copy assignment
const __class__&
__class__::operator=(const __class__& rhs)
{
    // Check for self-assignment.
    if (this == &rhs) {
        return *this;
    }
    // free old memory, copy new memory
    return *this;
}

// __class__ copy ctor
__class__::__class__(const __class__& src)
{
    // __class__ copy ctor
}
