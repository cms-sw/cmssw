#ifndef FWCore_Utilities_interface_stringize_h
#define FWCore_Utilities_interface_stringize_h

// convert the macro argument to a null-terminated quoted string
#define EDM_STRINGIZE_(token) #token
#define EDM_STRINGIZE(token) EDM_STRINGIZE_(token)

#endif  // FWCore_Utilities_interface_stringize_h
