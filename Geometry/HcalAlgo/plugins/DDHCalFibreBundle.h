#ifndef HcalAlgo_DDHCalFibreBundle_h
#define HcalAlgo_DDHCalFibreBundle_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDHCalFibreBundle : public DDAlgorithm {

public:
  //Constructor and Destructor
  DDHCalFibreBundle(); 
  virtual ~DDHCalFibreBundle();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  std::string              idNameSpace; //Namespace of this and ALL sub-parts
  std::string              childPrefix; //Prefix to child name
  std::string              material;    //Name of the material for bundles
  double                   deltaZ;      //Width in  Z  for mother
  double                   deltaPhi;    //Width in phi for mother
  int                      numberPhi;   //Number of sections in phi
  double                   tilt;        //Tilt angle of the readout box
  std::vector<double>      areaSection; //Area of a bundle
  std::vector<double>      rStart;      //Radius at start
  std::vector<double>      rEnd;        //Radius at End
  std::vector<int>         bundle;      //Bundle to be positioned 
};

#endif
