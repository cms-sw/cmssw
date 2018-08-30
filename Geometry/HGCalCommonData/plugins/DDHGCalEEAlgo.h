#ifndef HGCalCommonData_DDHGCalEEAlgo_h
#define HGCalCommonData_DDHGCalEEAlgo_h

#include <cmath>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"

class DDHGCalEEAlgo : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDHGCalEEAlgo(); //const std::string & name);
  ~DDHGCalEEAlgo() override;
  
  void initialize(const DDNumericArguments & nArgs,
                  const DDVectorArguments & vArgs,
                  const DDMapArguments & mArgs,
                  const DDStringArguments & sArgs,
                  const DDStringVectorArguments & vsArgs) override;
  void execute(DDCompactView& cpv) override;

protected:

  void          constructLayers (const DDLogicalPart&, DDCompactView& cpv);
  void          positionSensitive(const DDLogicalPart& glog, double rin, 
				  double rout, double zpos, int layertype, 
				  DDCompactView& cpv);

private:

  std::unique_ptr<HGCalWaferType> waferType_;

  std::vector<std::string> wafers_;       //Wafers
  std::vector<std::string> materials_;    //Materials
  std::vector<std::string> names_;        //Names
  std::vector<double>      thick_;        //Thickness of the material
  std::vector<int>         copyNumber_;   //Initial copy numbers
  std::vector<int>         layers_;       //Number of layers in a section
  std::vector<double>      layerThick_;   //Thickness of each section
  std::vector<int>         layerType_;    //Type of the layer
  std::vector<int>         layerSense_;   //Content of a layer (sensitive?)
  int                      firstLayer_;   //Copy # of the first sensitive layer
  double                   zMinBlock_;    //Starting z-value of the block
  std::vector<double>      rad100to200_;  //Parameters for 120-200mum trans.
  std::vector<double>      rad200to300_;  //Parameters for 200-300mum trans.
  double                   zMinRadPar_;   //Minimum z for radius parametriz.
  int                      choiceType_;   //Type of parametrization to be used
  int                      nCutRadPar_;   //Cut off threshold for corners
  double                   fracAreaMin_;  //Minimum fractional conatined area 
  double                   waferSize_;    //Width of the wafer
  double                   waferSepar_;   //Sensor separation
  int                      sectors_;      //Sectors   
  std::vector<double>      slopeB_;       //Slope at the lower R
  std::vector<double>      zFrontB_;      //Starting Z values for the slopes
  std::vector<double>      rMinFront_;    //Corresponding rMin's
  std::vector<double>      slopeT_;       //Slopes at the larger R
  std::vector<double>      zFrontT_;      //Starting Z values for the slopes
  std::vector<double>      rMaxFront_;    //Corresponding rMax's
  std::string              nameSpace_;    //Namespace of this and ALL sub-parts
  std::unordered_set<int>  copies_;       //List of copy #'s
};

#endif
