#ifndef DD_PixBarLayerAlgo_h
#define DD_PixBarLayerAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDPixBarLayerAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDPixBarLayerAlgo(); 
  virtual ~DDPixBarLayerAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  std::string              idNameSpace; //Namespace of this and ALL sub-parts
  std::string              genMat;      //Name of general material
  int                      number;      //Number of ladders in phi
  double                   layerDz;     //Length of the layer
  double                   sensorEdge;  //Distance from edge for a half sensor
  double                   coolDz;      //Length of the cooling piece
  double                   coolWidth;   //Width                       
  double                   coolSide;    //Side length
  double                   coolThick;   //Thickness of the shell     
  double                   coolDist;    //Radial distance between centres of 2 
  std::string              coolMat;     //Cooling fluid material name
  std::string              tubeMat;     //Cooling piece material name
  std::vector<std::string> ladder;      //Names     of ladders
  std::vector<double>      ladderWidth; //Widths         ...
  std::vector<double>      ladderThick; //Thickness      ...
};

#endif
