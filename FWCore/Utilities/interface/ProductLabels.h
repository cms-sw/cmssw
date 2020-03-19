#ifndef FWCore_Utilities_ProductLabels_h
#define FWCore_Utilities_ProductLabels_h
namespace edm {
  struct ProductLabels {
    char const* module;
    char const* productInstance;
    char const* process;
  };
}  // namespace edm
#endif
