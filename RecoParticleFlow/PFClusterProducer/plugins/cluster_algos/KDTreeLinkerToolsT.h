#ifndef KDTreeLinkerToolsTemplated_h
#define KDTreeLinkerToolsTemplated_h

#include <array>

// Box structure used to define 2D field.
// It's used in KDTree building step to divide the detector
// space (ECAL, HCAL...) and in searching step to create a bounding
// box around the demanded point (Track collision point, PS projection...).
template<unsigned DIM>
struct KDTreeBoxT
{
  std::array<float,DIM> dimmin, dimmax;
  
  template<typename... Ts>
  KDTreeBoxT(Ts... dimargs) {
    static_assert(sizeof...(dimargs) == 2*DIM,"Constructor requires 2*DIM args");
    std::vector<float> dims = {dimargs...};
    for( unsigned i = 0; i < DIM; ++i ) {
      dimmin[i] = dims[2*i];
      dimmax[i] = dims[2*i+1];
    }
  }

  KDTreeBoxT() {}
};

typedef KDTreeBoxT<2> KDTreeBox;
typedef KDTreeBoxT<3> KDTreeCube;

  
// Data stored in each KDTree node.
// The dim1/dim2 fields are usually the duplication of some PFRecHit values
// (eta/phi or x/y). But in some situations, phi field is shifted by +-2.Pi
template<typename DATA,unsigned DIM>
struct KDTreeNodeInfoT 
{
  DATA data;
  std::array<float,DIM> dims;

public:
  KDTreeNodeInfoT()
  {}
  
  template<typename... Ts>
  KDTreeNodeInfoT(const DATA& d,Ts... dimargs)
  : data(d), dims{ {dimargs...} }
  {}
};

// KDTree node.
template <typename DATA, unsigned DIM>
struct KDTreeNodeT
{
  // Data
  KDTreeNodeInfoT<DATA,DIM> info;
  
  // Right/left sons.
  KDTreeNodeT<DATA,DIM> *left, *right;
  
  // Region bounding box.
  KDTreeBoxT<DIM> region;
  
  public:
  KDTreeNodeT()
    : left(0), right(0)
  {}
  
  void setAttributs(const KDTreeBoxT<DIM>& regionBox,
		    const KDTreeNodeInfoT<DATA,DIM>& infoToStore) 
  {
    info = infoToStore;
    region = regionBox;
  }
  
  void setAttributs(const KDTreeBoxT<DIM>&   regionBox) 
  {
    region = regionBox;
  }
};

#endif
