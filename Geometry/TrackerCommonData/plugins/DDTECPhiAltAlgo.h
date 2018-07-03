#ifndef DD_TECPhiAltAlgo_h
#define DD_TECPhiAltAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDTECPhiAltAlgo : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDTECPhiAltAlgo(); 
  ~DDTECPhiAltAlgo() override;
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:

  double        startAngle;  //Start angle
  double        incrAngle;   //Increment in angle
  double        radius  ;    //Radius
  double        zIn;         //z position for the even ones
  double        zOut;        //z position for the odd  ones
  int           number;      //Number of copies
  int           startCopyNo; //Start copy number
  int           incrCopyNo;  //Increment in copy number

  std::string   idNameSpace; //Namespace of this and ALL sub-parts
  std::string   childName;   //Child name
};

#endif
