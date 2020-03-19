#ifndef DD_TIBLayerAlgo_MTCC_h
#define DD_TIBLayerAlgo_MTCC_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDTIBLayerAlgo_MTCC : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDTIBLayerAlgo_MTCC();
  ~DDTIBLayerAlgo_MTCC() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  std::string idNameSpace;  //Namespace of this and ALL parts
  std::string genMat;       //General material name
  double detectorTilt;      //Detector Tilt
  double layerL;            //Length of the layer
  double detectorTol;       //Tolerance
  double detectorW;         //Width     of detector layer
  double detectorT;         //Thickness       .........
  double coolTubeW;         //Width     of layer with cable+cool
  double coolTubeT;         //Thickness       .........

  double radiusLo;                   //Radius for detector at lower level
  double phioffLo;                   //Phi offset                 ......
  double phiMinLo;                   //Phi minimum                ......
  double phiMaxLo;                   //Phi maximum                ......
  int stringsLo;                     //Number of strings          ......
  std::vector<double> stringLoList;  //List of strings to fill    ......
  std::string detectorLo;            //Detector string name       ......
  std::string emptyDetectorLo;       //Empty Detector string name ......
  double roffDetLo;                  //Radial offset              ......
  std::string coolCableLo;           //Cable+Cool name            ......
  std::string emptyCoolCableLo;      //Empty Cable+Cool name      ......
  double roffCableLo;                //Radial offset              ......

  double radiusUp;                   //Radius for detector at upper level
  double phioffUp;                   //Phi offset                 ......
  double phiMinUp;                   //Phi minimum                ......
  double phiMaxUp;                   //Phi maximum                ......
  int stringsUp;                     //Number of strings          ......
  std::vector<double> stringUpList;  //List of strings to fill    ......
  std::string detectorUp;            //Detector string name       ......
  std::string emptyDetectorUp;       //Empty Detector string name ......
  double roffDetUp;                  //Radial offset              ......
  std::string coolCableUp;           //Cable+Cool name            ......
  std::string emptyCoolCableUp;      //Empty Cable+Cool name ......
  double roffCableUp;                //Radial offset              ......

  double cylinderT;            //Cylinder thickness
  std::string cylinderMat;     //Cylinder material
  double supportW;             //Support wall width
  double supportT;             //Support wall thickness
  std::string supportMat;      //Support wall  material
  std::string ribMat;          //Rib material
  std::vector<double> ribW;    //Rib width
  std::vector<double> ribPhi;  //Rib Phi position

  int dohmN;                        //Number of phi sectors for DOHMs
  std::vector<double> dohmList;     //List of DOHMs
  double dohmCarrierW;              //DOHM Carrier Width
  double dohmCarrierT;              //DOHM Carrier Thickness
  double dohmCarrierR;              //DOHM Carrier Radial Height
  std::string dohmCarrierMaterial;  //DOHM Carrier Material
  std::string dohmCableMaterial;    //DOHM Cable Material
  double dohmPrimW;                 //DOHM PRIMary Width
  double dohmPrimL;                 //DOHM PRIMary Length
  double dohmPrimT;                 //DOHM PRIMary Thickness
  std::string dohmPrimMaterial;     //DOHM PRIMary Material
  double dohmAuxW;                  //DOHM AUXiliary Width
  double dohmAuxL;                  //DOHM AUXiliary Length
  double dohmAuxT;                  //DOHM AUXiliary Thickness
  std::string dohmAuxMaterial;      //DOHM AUXiliary Material
};

#endif
