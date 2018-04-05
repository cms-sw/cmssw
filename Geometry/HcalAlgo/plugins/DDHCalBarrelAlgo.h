#ifndef HcalAlgo_DDHCalBarrelAlgo_h
#define HcalAlgo_DDHCalBarrelAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDHCalBarrelAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDHCalBarrelAlgo(); //const std::string & name);
  ~DDHCalBarrelAlgo() override;
  
  //Get Methods
  std::string getGenMaterial()                const {return genMaterial;}
  int         getNsectors()                   const {return nsectors;}
  int         getNsectortot()                 const {return nsectortot;}
  int         getNhalf()                      const {return nhalf;}
  double      getRin()                        const {return rin;}
  double      getRout()                       const {return rout;}
  int         getRzones()                     const {return rzones;}
  double      getTanTheta(unsigned int i)     const {return ttheta[i];}
  double      getTheta(unsigned int i)        const {return theta[i];}
  double      getRmax(unsigned int i)         const {return rmax[i];}
  double      getZoff(unsigned int i)         const {return zoff[i];}
  
  int         getNLayers()                    const {return nLayers;}
  int         getLayerId(unsigned i)          const {return layerId[i];}
  std::string getLayerLabel(unsigned i)       const {return layerLabel[i];}
  std::string getLayerMaterial(unsigned i)    const {return layerMat[i];}
  double      getLayerWidth(unsigned i)       const {return layerWidth[i];}
  double      getLayerD1(unsigned i)          const {return layerD1[i];}
  double      getLayerD2(unsigned i)          const {return layerD2[i];}
  double      getLayerAlpha(unsigned i)       const {return layerAlpha[i];}
  double      getLayerT1(unsigned i)          const {return layerT1[i];}
  double      getLayerT2(unsigned i)          const {return layerT2[i];}
  int         getLayerAbsorb(unsigned int i)  const {return layerAbsorb[i];}
  double      getLayerGap(unsigned int i)     const {return layerGap[i];}

  std::string getSideMat(unsigned int i)      const {return sideMat[i];}
  double      getSideD(unsigned int i)        const {return sideD[i];}
  double      getSideT(unsigned int i)        const {return sideT[i];}
  int         getSideAbsorber()               const {return nSideAbs;}
  std::string getSideAbsName(unsigned int i)  const {return sideAbsName[i];}
  std::string getSideAbsMat(unsigned int i)   const {return sideAbsMat[i];}
  double      getSideAbsW(unsigned int i)     const {return sideAbsW[i];}

  int         getAbsorberN()                  const {return nAbsorber;}
  std::string getAbsorbName(unsigned int i)   const {return absorbName[i];}
  std::string getAbsorbMat(unsigned int i)    const {return absorbMat[i];}
  double      getAbsorbD(unsigned int i)      const {return absorbD[i];}
  double      getAbsorbT(unsigned int i)      const {return absorbT[i];}
  std::string getMiddleMat()                  const {return middleMat;}
  double      getMiddleD()                    const {return middleD;}
  double      getMiddleW()                    const {return middleW;}
  int         getMidAbsorber()                const {return nMidAbs;}
  std::string getMidAbsName(unsigned int i)   const {return midName[i];}
  std::string getMidAbsMat(unsigned int i)    const {return midMat[i];}
  double      getMidAbsW(unsigned int i)      const {return midW[i];}
  double      getMidAbsT(unsigned int i)      const {return midT[i];}

  std::string getDetMat()                     const {return detMat;}
  std::string getDetMatPl()                   const {return detMatPl;}
  std::string getDetMatSc()                   const {return detMatSc;}
  int         getDetType(unsigned int i)      const {return detType[i];}
  double      getDetdP1(unsigned int i)       const {return detdP1[i];}
  double      getDetdP2(unsigned int i)       const {return detdP2[i];}
  double      getDetT11(unsigned int i)       const {return detT11[i];}
  double      getDetT12(unsigned int i)       const {return detT12[i];}
  double      getDetTsc(unsigned int i)       const {return detTsc[i];}
  double      getDetT21(unsigned int i)       const {return detT21[i];}
  double      getDetT22(unsigned int i)       const {return detT22[i];}
  double      getDetWidth1(unsigned int i)    const {return detWidth1[i];}
  double      getDetWidth2(unsigned int i)    const {return detWidth2[i];}
  int         getDetPosY(unsigned int i)      const {return detPosY[i];}

  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;

  void execute(DDCompactView& cpv) override;

protected:

  void constructGeneralVolume(DDCompactView& cpv);
  void constructInsideSector(const DDLogicalPart& sector, DDCompactView& cpv);
  void constructInsideLayers(const DDLogicalPart& laylog, const std::string& name, int id, 
			     int nAbs, double rin, double d1, double alpha1, 
			     double d2, double alpha2, double t1, double t2, DDCompactView& cpv);
  DDLogicalPart constructSideLayer(const DDLogicalPart& laylog, const std::string& nm,
				   int nAbs, double rin, double alpha, DDCompactView& cpv);
  DDLogicalPart constructMidLayer(const DDLogicalPart& laylog, const std::string& nm,
				  double rin, double alpha, DDCompactView& cpv);
  void constructInsideDetectors(const DDLogicalPart& detector,
				const std::string& name, int id, double dx, 
				double dy, double dz, int type, DDCompactView& cpv);

private:
  //General Volume
  //      <----- Zmax ------>
  // Rout ************************-------
  //      *                      *Rstep2|        Theta angle w.r.t. vertical
  //      *                      *---------------
  //      *                     *               |
  //      *                    *Theta[i]        Rmax[i]
  //      *                   *---------------  |
  //                        *Theta[0] Rmax[0]|  |
  // Rin  *****************----------------------

  std::string  genMaterial;         //General material
  int       nsectors;               //Number of potenital straight edges
  int       nsectortot;             //Number of straight edges (actual)
  int       nhalf;                  //Number of half modules
  double    rin, rout;              //See picture
  int       rzones;                 //  ....
  std::vector<double>   theta;      //  .... (in degrees)
  std::vector<double>   rmax;       //  .... 
  std::vector<double>   zoff;       //  ....
  std::vector<double>   ttheta;     //tan(theta)
  std::string rotHalf;              //Rotation matrix of the second half
  std::string rotns;                //Name space for Rotation matrices

  //Upper layers inside general volume
  //     <---- Zout ---->
  //  |  ****************     |
  //  |  *              *     Wstep
  //  W  *              ***** |
  //  |  *                  *
  //  |  ********************
  //     <------ Zin ------->
  // Middle layers inside general volume
  //     <------ Zout ------>         Zout = Full sector Z at position
  //  |  ********************         Zin  = Full sector Z at position
  //  |  *                 *
  //  W  *                * Angle = Theta sector
  //  |  *               *  )
  //  |  ****************--------
  //     <------ Zin ------->

  // Lower layers
  //     <------ Zout ------>         Zin(i)=Zout(i-1)
  //  |  ********************         Zout(i)=Zin(i)+W(i)/tan(Theta(i))
  //  |  *                 *
  //  W  *                *  Theta
  //  |  *               *
  //  |  ****************--------
  //     <--- Zin ------>

  int                      nLayers;     //Number of layers
  std::vector<int>         layerId;     //Number identification
  std::vector<std::string> layerLabel;  //String identification
  std::vector<std::string> layerMat;    //Material
  std::vector<double>      layerWidth;  //W in picture
  std::vector<double>      layerD1;     //d1 in front picture
  std::vector<double>      layerD2;     //d2 in front picture
  std::vector<double>      layerAlpha;  //Angular width of the middle tiles
  std::vector<double>      layerT1;     //t in front picture (side)
  std::vector<double>      layerT2;     //t in front picture (front)
  std::vector<int>         layerAbsorb; //Absorber flag
  std::vector<double>      layerGap;    //Gap at the edge

  int                      nAbsorber;   //Number of absorber layers in middle
  std::vector<std::string> absorbName;  //Absorber name
  std::vector<std::string> absorbMat;   //Absorber material
  std::vector<double>      absorbD;     //Distance from the bottom surface
  std::vector<double>      absorbT;     //Thickness
  std::string              middleMat;   //Material of the detector layer
  double                   middleD;     //Distance from the bottom surface
  double                   middleW;     //Half width
  int                      nMidAbs;     //Number of absorbers in front part
  std::vector<std::string> midName;     //Absorber names in the front part
  std::vector<std::string> midMat;      //Absorber material
  std::vector<double>      midW;        //Half width
  std::vector<double>      midT;        //Thickness

  std::vector<std::string> sideMat;     //Material for special side layers
  std::vector<double>      sideD;       //Depth from bottom surface
  std::vector<double>      sideT;       //Thickness
  int                      nSideAbs;    //Number of absorbers in special side
  std::vector<std::string> sideAbsName; //Absorber name
  std::vector<std::string> sideAbsMat;  //Absorber material
  std::vector<double>      sideAbsW;    //Half width

  // Detectors. Each volume inside the layer has the shape:
  //
  // ******************************* |
  // *\\\\\\\Plastic\\\\\\\\\\\\\\\* T2
  // ******************************* |        
  // *////Scintillator/////////////* Tsc      
  // ******************************* |        
  // *\\\\\\\Plastic\\\\\\\\\\\\\\\* T1       
  // ******************************* |   |    
  // *         Air                 *     dP1  
  // *******************************     |    
  // 
  std::string         detMat;                 //fill material
  std::string         detRot;                 //Rotation matrix for the 2nd
  std::string         detMatPl;               //Plastic material
  std::string         detMatSc;               //Scintillator material
  std::vector<int>    detType;       
  std::vector<double> detdP1;                 //Air gap (side)
  std::vector<double> detdP2;                 //Air gap (centre)
  std::vector<double> detT11;                 //Back plastic thickness (side)
  std::vector<double> detT12;                 //Back plastic thickness (centre)
  std::vector<double> detTsc;                 //Scintillator
  std::vector<double> detT21;                 //Front plastic thickness (side)
  std::vector<double> detT22;                 //Front plastic thickness (centre)
  std::vector<double> detWidth1;              //Width of phi(1,4) megatiles
  std::vector<double> detWidth2;              //Width of phi(2,3) megatiles
  std::vector<int>    detPosY;                //Positioning of phi(1,4) tiles - 0 centre

  std::string         idName;                 //Name of the "parent" volume.  
  std::string         idNameSpace;            //Namespace of this and ALL sub-parts
  int                 idOffset;               // Geant4 ID's...    = 3000;
};

#endif
