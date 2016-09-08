#ifndef DD_CutTubsFromPoints_h
#define DD_CutTubsFromPoints_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

// This algorithm creates a ring made of CutTubs segments from the phi,z points
// of the rings "corners".
// The algorithm only defines a single Solid with the given name.
class DDCutTubsFromPoints : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDCutTubsFromPoints(); 
  virtual ~DDCutTubsFromPoints();
  
  void initialize(const DDNumericArguments & nArgs,
                  const DDVectorArguments & vArgs,
                  const DDMapArguments & mArgs,
                  const DDStringArguments & sArgs,
                  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:
  struct Section {
    double phi; // phi position of this edge
    double z_l; // -Z end (cuttubs l plane)
    double z_t; // +Z end (cuttubs t plane)
    // radius is implicitly r_min
  };

  double r_min;
  double r_max;

  // a segment is produced between each two consecutive sections that have a 
  // non-zero phi distance. Sections with zero phi distance can be used to 
  // create sharp jumps.
  std::vector<Section> sections;
  // this solid will be defined.
  DDName solidOutput;


};

#endif
