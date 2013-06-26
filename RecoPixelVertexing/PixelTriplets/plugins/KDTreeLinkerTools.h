#ifndef KDTreeLinkerToolsTemplated_h
#define KDTreeLinkerToolsTemplated_h

// Box structure used to define 2D field.
// It's used in KDTree building step to divide the detector
// space (ECAL, HCAL...) and in searching step to create a bounding
// box around the demanded point (Track collision point, PS projection...).
struct KDTreeBox
{
  float dim1min, dim1max;
  float dim2min, dim2max;
  
  public:

  KDTreeBox(float d1min, float d1max, 
	    float d2min, float d2max)
    : dim1min (d1min), dim1max(d1max)
    , dim2min (d2min), dim2max(d2max)
  {}

  KDTreeBox()
    : dim1min (0), dim1max(0)
    , dim2min (0), dim2max(0)
  {}
};

  
// Data stored in each KDTree node.
// The dim1/dim2 fields are usually the duplication of some PFRecHit values
// (eta/phi or x/y). But in some situations, phi field is shifted by +-2.Pi
template <typename DATA>
struct KDTreeNodeInfo 
{
  DATA data;
  float dim1;
  float dim2;

  public:
  KDTreeNodeInfo()
  {}
  
  KDTreeNodeInfo(const DATA&	d,
		 float		dim_1,
		 float		dim_2)
    : data(d), dim1(dim_1), dim2(dim_2)
  {}
};

// KDTree node.
template <typename DATA>
struct KDTreeNode
{
  // Data
  KDTreeNodeInfo<DATA> info;
  
  // Right/left sons.
  KDTreeNode *left, *right;
  
  // Region bounding box.
  KDTreeBox region;
  
  public:
  KDTreeNode()
    : left(0), right(0)
  {}
  
  void setAttributs(const KDTreeBox&		regionBox,
		    const KDTreeNodeInfo<DATA>&	infoToStore) 
  {
    info = infoToStore;
    region = regionBox;
  }
  
  void setAttributs(const KDTreeBox&   regionBox) 
  {
    region = regionBox;
  }
};

#endif
