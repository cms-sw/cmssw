#ifndef HGCalCommonData_DDHGCalHEAlgo_h
#define HGCalCommonData_DDHGCalHEAlgo_h

#include <cmath>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"

class DDHGCalHEAlgo : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDHGCalHEAlgo(); //const std::string & name);
  ~DDHGCalHEAlgo() override;
  
  void initialize(const DDNumericArguments & nArgs,
                  const DDVectorArguments & vArgs,
                  const DDMapArguments & mArgs,
                  const DDStringArguments & sArgs,
                  const DDStringVectorArguments & vsArgs) override;
  void execute(DDCompactView& cpv) override;

protected:

  void          constructLayers (const DDLogicalPart&, DDCompactView& cpv);
  void          positionMix(const DDLogicalPart& glog,const std::string& name,
			    int copy, double thick, const DDMaterial& matter,
			    double rin, double rmid, double routF, double zz,
			    DDCompactView& cpv);
  void          positionSensitive(const DDLogicalPart& glog, double rin, 
				  double rout, double zpos, int layertype,
				  DDCompactView& cpv);

private:

  std::unique_ptr<HGCalWaferType> waferType_;

  std::vector<std::string> wafers_;        //Wafers
  std::vector<std::string> materials_;     //Materials
  std::vector<std::string> names_;         //Names
  std::vector<double>      thick_;         //Thickness of the material
  std::vector<int>         copyNumber_;    //Initial copy numbers
  std::vector<int>         layers_;        //Number of layers in a section
  std::vector<double>      layerThick_;    //Thickness of each section
  std::vector<double>      rMixLayer_;     //Partition between Si/Sci part
  std::vector<int>         layerType_;     //Type of the layer
  std::vector<int>         layerSense_;    //Content of a layer (sensitive?)
  int                      firstLayer_;    //Copy # of the first sensitive layer
  std::vector<std::string> materialsTop_;  //Materials of top layers
  std::vector<std::string> namesTop_;      //Names of top layers
  std::vector<double>      layerThickTop_; //Thickness of the top sections
  std::vector<int>         layerTypeTop_;  //Type of the Top layer
  std::vector<int>         copyNumberTop_; //Initial copy numbers (top section)
  std::vector<std::string> materialsBot_;  //Materials of bottom layers
  std::vector<std::string> namesBot_;      //Names of bottom layers
  std::vector<double>      layerThickBot_; //Thickness of the bottom sections
  std::vector<int>         layerTypeBot_;  //Type of the bottom layers
  std::vector<int>         copyNumberBot_; //Initial copy numbers (bot section)
  std::vector<int>         layerSenseBot_; //Content of bottom layer (sensitive?)

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
