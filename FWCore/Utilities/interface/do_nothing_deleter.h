#ifndef FWCore_Utilities_do_nothing_deleter_h
#define FWCore_Utilities_do_nothing_deleter_h

// ----------------------------------------------------------------------
//
// do_nothing_deleter.h
//
// Purpose: do_nothing_deleter provides a way to use boost::shared_ptr
// or boost::shared_array for those cases where the object or array
// may be either in dynamic (heap) storage, or in static storage,
// as long as which of these applies when the shared_ptr or shared_array
// is constructed.
//
// For objects:
//
// If the object is allocated in dynamic storage, use
// boost::shared_ptr<T> (new T(...));

// If the object "t" is in static storage, use
// boost::shared_ptr<T> (&t, do_nothing_deleter());
//
// For arrays:
//
// If the array is allocated in dynamic storage, use
// boost::shared_array<T> (new T(...)[]);

// If the array "t" is in static storage, use
// boost::shared_ptr<T> (t, do_nothing_deleter());
//
//
// ----------------------------------------------------------------------

namespace edm {
  struct do_nothing_deleter {
    void operator()(void const*) const {}
  };
}

#endif
