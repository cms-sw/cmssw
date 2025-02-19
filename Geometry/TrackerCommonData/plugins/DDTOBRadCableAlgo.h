#ifndef DD_TOBRadCableAlgo_h
#define DD_TOBRadCableAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDTOBRadCableAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDTOBRadCableAlgo(); 
  virtual ~DDTOBRadCableAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  std::string         idNameSpace;   // Namespace of this and ALL sub-parts
				       
  double              diskDz;        // Disk  thickness
  double              rMax;          // Maximum radius
  double              cableT;        // Cable thickness
  std::vector<double> rodRin;        // Radii for inner rods
  std::vector<double> rodRout;       // Radii for outer rods
  std::vector<std::string> cableM;   // Materials for cables
  double              connW;         // Connector width
  double              connT;         // Connector thickness
  std::vector<std::string> connM;    // Materials for connectors
  std::vector<double> coolR1;        // Radii for cooling manifold
  std::vector<double> coolR2;        // Radii for return cooling manifold
  double              coolRin;       // Inner radius of cooling manifold
  double              coolRout1;     // Outer radius of cooling manifold
  double              coolRout2;     // Outer radius of cooling fluid in cooling manifold
  double              coolStartPhi1; // Starting Phi of cooling manifold
  double              coolDeltaPhi1; // Phi Range of cooling manifold
  double              coolStartPhi2; // Starting Phi of cooling fluid in of cooling manifold
  double              coolDeltaPhi2; // Phi Range of of cooling fluid in cooling manifold
  std::string         coolM1;        // Material for cooling manifold
  std::string         coolM2;        // Material for cooling fluid
  std::vector<std::string> names;    // Names of layers
};

#endif
