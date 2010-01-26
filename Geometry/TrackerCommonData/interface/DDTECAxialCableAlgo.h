#ifndef DD_TECAxialCableAlgo_h
#define DD_TECAxialCableAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDTECAxialCableAlgo : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDTECAxialCableAlgo(); 
  virtual ~DDTECAxialCableAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  int                 n;          //Number of copies
  double              rangeAngle; //Range in angle
  double              zStart;     //Start z        of the Service volume
  double              zEnd;       //End   z             ........
  double              rMin;       //Minimum radius      ........
  double              rMax;       //Maximum radius      ........
  double              width;      //Angular width
  double              thickR;     //Thickness (along R)
  double              thickZ;     //Thickness (along Z)
  double              dZ;         //Offset along Z from the central point
  std::vector<double> startAngle; //Start angle
  std::vector<double> zPos;       //Starting Z of the cables
  double              delta;      //Increment in phi

  std::string   idNameSpace;      //Namespace of this and ALL sub-parts
  std::string   childName;        //Child name
  std::string   matName;          //Material name
};

#endif
