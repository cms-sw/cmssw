#ifndef KDTreeLinkerToolsTemplated_h
#define KDTreeLinkerToolsTemplated_h

// Box structure used to define 2D field.
// It's used in KDTree building step to divide the detector
// space (ECAL, HCAL...) and in searching step to create a bounding
// box around the demanded point (Track collision point, PS projection...).
struct KDTreeBox
{
  double dim1min, dim1max;
  double dim2min, dim2max;
  
  public:

  KDTreeBox(double d1min, double d1max, 
	    double d2min, double d2max)
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
  double dim1;
  double dim2;

  public:
  KDTreeNodeInfo()
  {}
  
  KDTreeNodeInfo(const DATA&	d,
		 double		dim_1,
		 double		dim_2)
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
