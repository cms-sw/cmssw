#ifndef FWCore_Utilities_interface_concatenate_h
#define FWCore_Utilities_interface_concatenate_h

// concatenate the macro arguments into a single token
#define EDM_CONCATENATE_(a, b) a##b
#define EDM_CONCATENATE(a, b) EDM_CONCATENATE_(a, b)

#endif  // FWCore_Utilities_interface_concatenate_h
