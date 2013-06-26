#ifndef DD_TIBLayerAlgo_h
#define DD_TIBLayerAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDTIBLayerAlgo : public DDAlgorithm {
 public:
  //Constructor and Destructor
  DDTIBLayerAlgo(); 
  virtual ~DDTIBLayerAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  std::string              idNameSpace;    //Namespace of this and ALL parts
  std::string              genMat;         //General material name
  double                   detectorTilt;   //Detector Tilt
  double                   layerL;         //Length of the layer

  double                   radiusLo;       //Radius for detector at lower level
  double                   phioffLo;       //Phi offset             ......
  int                      stringsLo;      //Number of strings      ......
  std::string              detectorLo;     //Detector string name   ......

  double                   radiusUp;       //Radius for detector at upper level
  double                   phioffUp;       //Phi offset             ......
  int                      stringsUp;      //Number of strings      ......
  std::string              detectorUp;     //Detector string name   ......

  double                   cylinderT;      //Cylinder thickness
  double                   cylinderInR;    //Cylinder inner radius
  std::string              cylinderMat;    //Cylinder material
  double                   MFRingInR;      //Inner Manifold Ring Inner Radius 
  double                   MFRingOutR;     //Outer Manifold Ring Outer Radius 
  double                   MFRingT;        //Manifold Ring Thickness
  double                   MFRingDz;       //Manifold Ring Half Lenght
  std::string              MFIntRingMat;      //Manifold Ring Material
  std::string              MFExtRingMat;      //Manifold Ring Material

  double                   supportT;       //Cylinder barrel CF skin thickness

  std::string              centMat;        //Central rings  material
  std::vector<double>      centRing1par;   //Central rings parameters
  std::vector<double>      centRing2par;   //Central rings parameters

  std::string              fillerMat;      //Filler material
  double                   fillerDz;       //Filler Half Length

  std::string              ribMat;         //Rib material
  std::vector<double>      ribW;           //Rib width
  std::vector<double>      ribPhi;         //Rib Phi position

  std::vector<double>      dohmListFW;         //DOHM/AUX positions in #strings FW
  std::vector<double>      dohmListBW;         //DOHM/AUX positions in #strings BW

  double                   dohmtoMF;           //DOHM Distance to MF
  double                   dohmCarrierPhiOff;  //DOHM Carrier Phi offset wrt horizontal
  std::string              dohmPrimName;       //DOHM Primary Logical Volume name
  std::string              dohmAuxName;        //DOHM Auxiliary Logical Volume name

  std::string              dohmCarrierMaterial;//DOHM Carrier Material
  std::string              dohmCableMaterial;  //DOHM Cable Material
  double                   dohmPrimL;          //DOHM PRIMary Length
  std::string              dohmPrimMaterial;   //DOHM PRIMary Material
  double                   dohmAuxL;           //DOHM AUXiliary Length
  std::string              dohmAuxMaterial;    //DOHM AUXiliary Material

  std::string              pillarMaterial;     //Pillar Material

  double                   fwIntPillarDz;      //Internal pillar parameters
  double                   fwIntPillarDPhi;
  std::vector<double>      fwIntPillarZ;
  std::vector<double>      fwIntPillarPhi;
  double                   bwIntPillarDz;
  double                   bwIntPillarDPhi;
  std::vector<double>      bwIntPillarZ;
  std::vector<double>      bwIntPillarPhi;

  double                   fwExtPillarDz;      //External pillar parameters
  double                   fwExtPillarDPhi;
  std::vector<double>      fwExtPillarZ;  
  std::vector<double>      fwExtPillarPhi;
  double                   bwExtPillarDz;
  double                   bwExtPillarDPhi;
  std::vector<double>      bwExtPillarZ;  
  std::vector<double>      bwExtPillarPhi;
};

#endif
