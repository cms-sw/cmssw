#ifndef DD_TIDAxialCableAlgo_h
#define DD_TIDAxialCableAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDTIDAxialCableAlgo : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDTIDAxialCableAlgo(); 
  ~DDTIDAxialCableAlgo() override;
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:

  double              zBend;      //Start z (at bending)........
  double              zEnd;       //End   z             ........
  double              rMin;       //Minimum radius      ........
  double              rMax;       //Maximum radius      ........
  double              rTop;       //Maximum radius (top)........
  double              width;      //Angular width
  double              thick;      //Thickness 
  std::vector<double> angles;     //Phi Angles
  std::vector<double> zposWheel;  //Z position of wheels
  std::vector<double> zposRing;   //Z position of rings inside wheels

  std::string   idNameSpace;      //Namespace of this and ALL sub-parts
  std::string   childName;        //Child name
  std::string   matIn;            //Material name (for inner parts)
  std::string   matOut;           //Material name (for outer part)
};

#endif
