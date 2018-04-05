#ifndef HcalAlgo_DDHCalTBCableAlgo_h
#define HcalAlgo_DDHCalTBCableAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDHCalTBCableAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDHCalTBCableAlgo(); //const std::string & name);
  ~DDHCalTBCableAlgo() override;

  void initialize(const DDNumericArguments & nArgs,
			  const DDVectorArguments & vArgs,
			  const DDMapArguments & mArgs,
			  const DDStringArguments & sArgs,
			  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:

  std::string           genMat;          //General material
  int                   nsectors;        //Number of potenital straight edges
  int                   nsectortot;      //Number of straight edges (actual)
  int                   nhalf;           //Number of half modules
  double                rin;             //(see Figure of hcalbarrel)
  std::vector<double>   theta;           //  .... (in degrees)
  std::vector<double>   rmax;            //  .... 
  std::vector<double>   zoff;            //  ....
  std::string           absMat;          //Absorber material
  double                thick;           //Thickness of absorber
  double                width1, length1; //Width, length of absorber type 1
  double                width2, length2; //Width, length of absorber type 2
  double                gap2;            //Gap between abosrbers of type 2

  std::string           idName;          //Name of the "parent" volume.  
  std::string           idNameSpace;     //Namespace of this and ALL sub-parts
  std::string           rotns;           //Namespace for rotation matrix
};

#endif
