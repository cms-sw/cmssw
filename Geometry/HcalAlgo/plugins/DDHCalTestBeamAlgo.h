#ifndef HcalAlgo_DDHCalTestBeamAlgo_h
#define HcalAlgo_DDHCalTestBeamAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDHCalTestBeamAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDHCalTestBeamAlgo(); 
  ~DDHCalTestBeamAlgo() override;
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:

  double        eta;         //Eta at which beam is focussed
  double        phi;         //Phi    ................
  double        theta;       //Corresponding theta value
  double        distance;    //Distance of the centre of rotation
  double        distanceZ;   //Distance along x-axis of the centre of rotation
  double        dist;        //Overall distance
  double        dz;          //Half length along z of the volume to be placed
  int           copyNumber;  //Copy Number

  std::string   idNameSpace; //Namespace of this and ALL sub-parts
  std::string   childName;   //Children name
};

#endif
