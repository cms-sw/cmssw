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

  void execute();

private:

  std::string         idNameSpace;  //Namespace of this and ALL sub-parts

  double              diskDz;       //Disk  thickness
  double              rMax;         //Maximum radius
  double              cableT;       //Cable thickness
  std::vector<double> rodRin;       //Radii for inner rods
  std::vector<double> rodRout;      //Radii for outer rods
  std::vector<std::string> cableM;  //Materials for cables
  double              connW;        //Connector width
  double              connT;        //Connector thickness
  std::vector<std::string> connM;   //Materials for connectors
  double              coolW;        //Width     of  cooling manifold
  double              coolT;        //Thickness of  cooling manifold
  std::vector<double> coolR;        //Radii     for cooling manifold
  std::vector<std::string> coolM;   //Materials for cooling manifold
  std::vector<std::string> names;   //Names of layers

};

#endif
