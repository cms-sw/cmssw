#ifndef CUDADataFormats_HGCal_HGCConditions_h
#define CUDADataFormats_HGCal_HGCConditions_h

// Declare the struct for the conditions payload to be transferred. 
struct HeterogeneousConditionsESProduct {
  int *layer;
  int *wafer;
};

#endif
