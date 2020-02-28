#ifndef DD_TIBRadCableAlgo_MTCC_h
#define DD_TIBRadCableAlgo_MTCC_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDTIBRadCableAlgo_MTCC : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDTIBRadCableAlgo_MTCC();
  ~DDTIBRadCableAlgo_MTCC() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  std::string idNameSpace;  //Namespace of this and ALL sub-parts

  double rMin;                        //Minimum radius
  double rMax;                        //Maximum radius
  std::vector<double> layRin;         //Radii for inner layer
  double deltaR;                      //DeltaR between inner and outer layer
  double cylinderT;                   //Support cylinder thickness
  double supportT;                    //Support disk thickness
  double supportDR;                   //Extra width along R
  std::string supportMat;             //Material for support disk
  double cableT;                      //Cable thickness
  std::vector<std::string> cableMat;  //Materials for cables
  std::vector<std::string> strucMat;  //Materials for open structures
  std::vector<double> layers;         //Layers to be constructed
};

#endif
