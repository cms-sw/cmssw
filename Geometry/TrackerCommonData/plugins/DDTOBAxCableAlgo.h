#ifndef DD_TOBAxCableAlgo_h
#define DD_TOBAxCableAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDTOBAxCableAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDTOBAxCableAlgo(); 
  virtual ~DDTOBAxCableAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  std::string         idNameSpace;    // Namespace of this and ALL sub-parts
  
  std::vector<std::string> sectorNumber;     // Id. Number of the sectors
  
  double sectorRin;                          // Inner radius of service sectors  
  double sectorRout;                         // Outer radius of service sectors  
  double sectorDz;                           // Sector half-length
  double sectorDeltaPhi_B;                   // Sector B phi width [A=C=0.5*(360/sectors)]
  std::vector<double> sectorStartPhi;        // Starting phi for the service sectors
  std::vector<std::string> sectorMaterial_A; // Material for the A sectors
  std::vector<std::string> sectorMaterial_B; // Material for the B sectors
  std::vector<std::string> sectorMaterial_C; // Material for the C sectors
  
};

#endif
