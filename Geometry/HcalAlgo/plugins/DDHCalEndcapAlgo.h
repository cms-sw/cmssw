#ifndef HcalAlgo_DDHCalEndcapAlgo_h
#define HcalAlgo_DDHCalEndcapAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDHCalEndcapAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDHCalEndcapAlgo(); //const std::string & name);
  ~DDHCalEndcapAlgo() override;
  
  //Get Methods
  std::string getGenMat()                  const {return genMaterial;}
  std::string getRotation()                const {return rotation;}
  int         getNsectors()                const {return nsectors;}
  int         getNsectortot()              const {return nsectortot;}
  int         getEndcaps()                 const {return nEndcap;}
  int         equipModule(unsigned int i)  const {return eModule[i];}
  double      getZShift()                  const {return zShift;}

  double      getZFront()                  const {return zFront;}
  double      getZEnd()                    const {return zEnd;}
  double      getZiNose()                  const {return ziNose;}
  double      getZiL0Nose()                const {return ziL0Nose;}
  double      getZiBody()                  const {return ziBody;}
  double      getZiL0Body()                const {return ziL0Body;}
  double      getZiKink()                  const {return ziKink;}
  double      getZ0Beam()                  const {return z0Beam;}
  double      getZ1Beam()                  const {return z1Beam;}
  double      getZiDip()                   const {return ziDip;}
  double      getDzStep()                  const {return dzStep;}
  double      getDzShift()                 const {return dzShift;}
  double      getZShiftHac2()              const {return zShiftHac2;}

  double      getRout()                    const {return rout;}
  double      getRinKink()                 const {return riKink;}
  double      getRinDip()                  const {return riDip;}
  double      getRoutDip()                 const {return roDip;}
  double      getHeboxDepth()              const {return heboxDepth;}
  double      getDrEnd()                   const {return drEnd;}
  double      getAngTop()                  const {return angTop;}
  double      getAngBot()                  const {return angBot;}
  double      getAngGap()                  const {return angGap;}
  double      getSlope()                   const {return slope;}

  std::string getAbsMat()                  const {return absMat;}
  int         getModules()                 const {return modules;}
  std::string getModName(unsigned int i)   const {return modName[i];}
  std::string getModMat(unsigned int i)    const {return modMat[i];}
  int         getModType(unsigned int i)   const {return modType[i];}
  int         getSectionModule(unsigned i) const {return sectionModule[i];}
  int         getLayerN(unsigned int i)    const {return layerN[i];}
  int         getLayer(unsigned int i, unsigned int j) const;
  double      getThick(unsigned int i)     const {return thick[i];}
  double      getTrim(unsigned int i, unsigned int j) const;
  double      getZminBlock(unsigned i)     const {return zminBlock[i];}
  double      getZmaxBlock(unsigned i)     const {return zmaxBlock[i];}
  double      getRinBlock1(unsigned i)     const {return rinBlock1[i];}
  double      getRinBlock2(unsigned i)     const {return rinBlock2[i];}
  double      getRoutBlock1(unsigned i)    const {return routBlock1[i];}
  double      getRoutBlock2(unsigned i)    const {return routBlock2[i];}

  int         getPhi()                     const {return phiSections;}
  std::string getPhiName(unsigned int i)   const {return phiName[i];}
  int         getLayers()                  const {return layers;}
  std::string getLayerName(unsigned int i) const {return layerName[i];}
  int         getLayerType(unsigned int i) const {return layerType[i];}
  double      getLayerT(unsigned int i)    const {return layerT[i];}
  double      getScintT(unsigned int i)    const {return scintT[i];}
  std::string getPlastMat()                const {return plastMat;}
  std::string getScintMat()                const {return scintMat;}
  std::string getRotMat()                  const {return rotmat;}
  double      getTolPos()                  const {return tolPos;}
  double      getTolAbs()                  const {return tolAbs;}

  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

protected:

  void constructGeneralVolume(DDCompactView& cpv);
  void constructInsideSector(const DDLogicalPart& sector, DDCompactView& cpv);
  void parameterLayer (int iphi, double rinF, double routF, double rinB, 
		       double routB, double zi, double zo, double& yh1, 
		       double& bl1, double& tl1, double& yh2, double& bl2,
		       double& tl2, double& alp, double& theta, double& phi,
		       double& xpos, double& ypos, double& zcpv);
  void parameterLayer0(int mod, int layer, int iphi, double& yh, double& bl, 
		       double& tl, double& alp, double& xpos, double& ypos, 
		       double& zcpv);
  void constructInsideModule0(const DDLogicalPart& module, int mod, DDCompactView& cpv);
  void constructInsideModule (const DDLogicalPart& module, int mod, DDCompactView& cpv);
  void constructScintLayer   (const DDLogicalPart& glog, double pDz, double yh, 
			      double bl, double tl, double alp, 
			      const std::string& name, int id, DDCompactView& cpv);

private:

  std::string              genMaterial;   //General material
  int                      nsectors;      //Number of potenital straight edges
  int                      nsectortot;    //Number of straight edges (actual)
  int                      nEndcap;       //Number of endcaps
  std::vector<int>         eModule;       //Modules to be present in part i (?)
  std::string              rotHalf;       //Rotation matrix for half
  std::string              rotns;         //Name space for rotation
  std::string              rotation;      //Rotation matrix to place in mother
  double                   zShift;        //needed for TB setup (move HE)

  double                   zFront;        //Z of the front section
  double                   zEnd;          //Outer Z of the HE
  double                   ziNose;        //Starting Z of the nose
  double                   ziL0Nose;      //Starting Z of layer 0 at nose
  double                   ziBody;        //Starting Z of the body
  double                   ziL0Body;      //Starting Z of layer 0 at body
  double                   ziKink;        //Position of the kink point
  double                   z0Beam;        //Position of gap front along z-axis
  double                   z1Beam;        //Position of gap end   along z-axis
  double                   ziDip;         //Starting Z of dipped part of body
  double                   dzStep;        //Width in Z of a layer
  double                   dzShift;       //Shift in Z for HE
  double                   zShiftHac2;    //needed for TB (remove part Hac2)

  double                   rout;          //Outer R of the HE
  double                   riKink;        //Inner radius at kink point
  double                   riDip;         //Inner radius at the dip point
  double                   roDip;         //Outer radius at the dip point
  double                   heboxDepth;    //Depth of the HE box
  double                   drEnd;         //Shift in R for the end absorber

  double                   angTop;        //Angle of top end of HE
  double                   angBot;        //Angle of the bottom end of HE
  double                   angGap;        //Gap angle (in degrees)
  double                   slope;         //Slope of the gap on HE side

  std::string              absMat;        //Absorber     material
  int                      modules;       //Number of modules
  std::vector<std::string> modName;       //Name
  std::vector<std::string> modMat;        //Material
  std::vector<int>         modType;       //Type (0/1 for front/standard)
  std::vector<int>         sectionModule; //Number of sections in a module
  std::vector<int>         layerN;        //Number of layers
  std::vector<int>         layerN0;       //Layer numbers in section 0
  std::vector<int>         layerN1;       //Layer numbers in section 1
  std::vector<int>         layerN2;       //Layer numbers in section 2
  std::vector<int>         layerN3;       //Layer numbers in section 3
  std::vector<int>         layerN4;       //Layer numbers in section 4
  std::vector<int>         layerN5;       //Layer numbers in section 5
  std::vector<double>      thick;         //Thickness of absorber/air
  std::vector<double>      trimLeft;      //Trimming of left  layers in module
  std::vector<double>      trimRight;     //Trimming of right layers in module
  std::vector<double>      zminBlock;     //Minimum Z
  std::vector<double>      zmaxBlock;     //Maximum Z
  std::vector<double>      rinBlock1;     //Inner Radius
  std::vector<double>      routBlock1;    //Outer Radius at zmin
  std::vector<double>      rinBlock2;     //Inner Radius
  std::vector<double>      routBlock2;    //Outer Radius at zmax
  
  int                      phiSections;   //Number of phi sections
  std::vector<std::string> phiName;       //Name of Phi sections
  int                      layers;        //Number of layers
  std::vector<std::string> layerName;     //Layer Names
  std::vector<int>         layerType;     //Detector type in each layer
  std::vector<double>      layerT;        //Layer thickness (plastic + scint.)
  std::vector<double>      scintT;        //Scintillator thickness
  std::string              plastMat;      //Plastic      material
  std::string              scintMat;      //Scintillator material
  std::string              rotmat;        //Rotation matrix for positioning

  std::string              idName;        //Name of the "parent" volume.  
  std::string              idNameSpace;   //Namespace of this and ALL sub-parts
  int                      idOffset;      // Geant4 ID's...    = 4000;

  double                   tolPos, tolAbs; //Tolerances
};

#endif
