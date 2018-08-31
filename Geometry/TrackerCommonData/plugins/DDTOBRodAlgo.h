#ifndef DD_TOBRodAlgo_h
#define DD_TOBRodAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDTOBRodAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDTOBRodAlgo(); 
  ~DDTOBRodAlgo() override;
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:

  std::string              central;        // Name of the central piece
					      
  double                   shift;          // Shift in z
  std::vector<std::string> sideRod;        // Name of the Side Rod
  std::vector<double>      sideRodX;       // x-positions
  std::vector<double>      sideRodY;       // y-positions
  std::vector<double>      sideRodZ;       // z-positions
  std::string              endRod1;        // Name of the End Rod of type 1
  std::vector<double>      endRod1Y;       // y-positions
  std::vector<double>      endRod1Z;       // z-positions
  std::string              endRod2;        // Name of the End Rod of type 2
  double                   endRod2Y;       // y-position
  double                   endRod2Z;       // z-position
					      
  std::string              cable;          // Name of the Mother cable
  double                   cableZ;         // z-position
					      
  std::string              clamp;          // Name of the clamp
  std::vector<double>      clampX;         // x-positions
  std::vector<double>      clampZ;         // z-positions
  std::string              sideCool;       // Name of the Side Cooling Tube
  std::vector<double>      sideCoolX;      // x-positions
  std::vector<double>      sideCoolY;      // y-positions to avoid overlap with the module (be at the same level of EndCool)
  std::vector<double>      sideCoolZ;      // z-positions
  std::string              endCool;        // Name of the End Cooling Tube
  std::string              endCoolRot;     // Rotation matrix name for end cool
  double                   endCoolY;       // y-position to avoid overlap with the module
  double                   endCoolZ;       // z-position
					      
  std::string              optFibre;       // Name of the Optical Fibre
  std::vector<double>      optFibreX;      // x-positions
  std::vector<double>      optFibreZ;      // z-positions
					      
  std::string              sideClamp1;     // Name of the side clamp of type 1
  std::vector<double>      sideClampX;     // x-positions
  std::vector<double>      sideClamp1DZ;   // Delta(z)-positions
  std::string              sideClamp2;     // Name of the side clamp of type 2
  std::vector<double>      sideClamp2DZ;   // Delta(z)-positions
					      
  std::string              module;         // Name of the detector modules
  std::vector<std::string> moduleRot;      // Rotation matrix name for module
  std::vector<double>      moduleY;        // y-positions
  std::vector<double>      moduleZ;        // z-positions
  std::vector<std::string> connect;        // Name of the connectors
  std::vector<double>      connectY;       // y-positions
  std::vector<double>      connectZ;       // z-positions
					      
  std::string              aohName;        // AOH name
  std::vector<double>      aohCopies;      // AOH copies to be positioned on each ICC
  std::vector<double>      aohX;           // AOH translation with respect small-ICC center (X)
  std::vector<double>      aohY;           // AOH translation with respect small-ICC center (Y)
  std::vector<double>      aohZ;           // AOH translation with respect small-ICC center (Z)
};

#endif
