#ifndef HGCalCommonData_DDHGCalEEAlgo_h
#define HGCalCommonData_DDHGCalEEAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDHGCalEEAlgo : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDHGCalEEAlgo(); //const std::string & name);
  virtual ~DDHGCalEEAlgo();
  
  struct HGCalEEPar {
    double yh1, bl1, tl1, yh2, bl2, tl2, alp, theta, phi, xpos, ypos, zpos;
    HGCalEEPar(double yh1v=0, double bl1v=0, double tl1v=0, double yh2v=0, 
	       double bl2v=0, double tl2v=0, double alpv=0, double thv=0,
	       double fiv=0, double x=0, double y=0, double z=0) :
    yh1(yh1v), bl1(bl1v), tl1(tl1v), yh2(yh2v), bl2(bl2v), tl2(tl2v),
      alp(alpv), theta(thv), phi(fiv), xpos(x), ypos(y), zpos(z) {}
  }; 
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);
  void execute(DDCompactView& cpv);

protected:

  void constructLayers (DDLogicalPart, DDCompactView& cpv);
  HGCalEEPar parameterLayer(double rinF, double routF, double rinB,
			    double routB, double zi, double zo);
  double     rMax(double z);

private:

  std::vector<std::string> materials;     //Materials
  std::vector<std::string> names;         //Names
  std::string              rotstr;        //Rotation matrix to place in mother
  std::vector<int>         layerType;     //Type of the layer
  std::vector<int>         heightType;    //Height to be evaluated from itself
  std::vector<int>         copyNumber;    //Copy number offsets for a section
  std::vector<double>      thick;         //Thickness of the material
  std::vector<double>      thickBlock;    //Thickness of a block of materials
  double                   zMinBlock;     //Starting z-value of the block
  int                      sectors;       //Sectors   
  double                   slopeB;        //Slope at the lower R
  std::vector<double>      slopeT;        //Slopes at the larger R
  std::vector<double>      zFront;        //Starting Z values for the slopes
  std::vector<double>      rMaxFront;     //Corresponding rMax's
  std::string              idName;        //Name of the "parent" volume.  
  std::string              idNameSpace;   //Namespace of this and ALL sub-parts
};

#endif
