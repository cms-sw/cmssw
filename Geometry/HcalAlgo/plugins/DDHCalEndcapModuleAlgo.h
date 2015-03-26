#ifndef HcalAlgo_DDHCalEndcapModuleAlgo_h
#define HcalAlgo_DDHCalEndcapModuleAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDHCalEndcapModuleAlgo : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDHCalEndcapModuleAlgo(); //const std::string & name);
  virtual ~DDHCalEndcapModuleAlgo();
  
  struct HcalEndcapPar {
    double yh1, bl1, tl1, yh2, bl2, tl2, alp, theta, phi, xpos, ypos, zpos;
    HcalEndcapPar(double yh1v=0, double bl1v=0, double tl1v=0, double yh2v=0, 
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

  void constructInsideModule0(DDLogicalPart module, DDCompactView& cpv);
  void constructInsideModule (DDLogicalPart module, DDCompactView& cpv);
  HcalEndcapPar parameterLayer0(unsigned int iphi);
  HcalEndcapPar parameterLayer(unsigned int iphi, double rinF, double routF, 
			       double rinB, double routB, double zi,double zo);
  void constructScintLayer(DDLogicalPart detector, double dz,
			   DDHCalEndcapModuleAlgo::HcalEndcapPar parm,
			   std::string nm, int id, DDCompactView& cpv);
  double getTrim(unsigned int j) const;
  double getRout(double z) const;

private:

  std::string              genMaterial;   //General material
  std::string              absorberMat;   //Absorber material
  std::string              plasticMat;    //Plastic material cover
  std::string              scintMat;      //Scintillator material
  std::string              rotstr;        //Rotation matrix to place in mother
  int                      sectors;       //Number of potenital straight edges
  double                   zMinBlock;     //Minimum z extent of the block 
  double                   zMaxBlock;     //Maximum
  double                   z1Beam;        //Position of gap end   along z-axis
  double                   ziDip;         //Starting Z of dipped part of body
  double                   dzStep;        //Width in Z of a layer
  double                   moduleThick;   //Thickness of a layer (air/absorber)
  double                   layerThick;    //Thickness of a layer (plastic)
  double                   scintThick;    //Thickness of scinitllator
  double                   rMaxBack;      //Maximum R after  the dip
  double                   rMaxFront;     //Maximum R before the dip
  double                   slopeBot;      //Slope of the bottom edge
  double                   slopeTop;      //Slope of the top edge
  double                   slopeTopF;     //Slope of the top front edge
  double                   trimLeft;      //Trim of the left edge
  double                   trimRight;     //Trim of the right edge
  double                   tolAbs;        //Tolerance for absorber
  int                      modType;       //Type of module
  int                      modNumber;     //Module number
  int                      layerType;     //Layer type
  std::vector<int>         layerNumber;   //layer numbers
  std::vector<std::string> phiName;       //Name of Phi sections
  std::vector<std::string> layerName;     //Layer Names

  std::string              idName;        //Name of the "parent" volume.  
  std::string              idNameSpace;   //Namespace of this and ALL sub-parts
  std::string              modName;       //Module Name
  int                      idOffset;      // Geant4 ID's...    = 4000;
};

#endif
