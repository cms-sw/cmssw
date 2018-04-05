#ifndef HGCalCommonData_DDHGCalEEAlgo_h
#define HGCalCommonData_DDHGCalEEAlgo_h

#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include <unordered_set>

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
  double        rMax(double z);
  void          positionSensitive(const DDLogicalPart& glog, double rin, 
				  double rout, int layertype, 
				  DDCompactView& cpv);

private:

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
  double                   rMaxFine_;     //Maximum r-value for fine wafer
  double                   rMinThick_;    //Transition R between 200 & 300 mum
  double                   waferSize_;    //Width of the wafer
  double                   waferSepar_;   //Sensor separation
  int                      sectors_;      //Sectors   
  std::vector<double>      slopeB_;       //Slope at the lower R
  std::vector<double>      slopeT_;       //Slopes at the larger R
  std::vector<double>      zFront_;       //Starting Z values for the slopes
  std::vector<double>      rMaxFront_;    //Corresponding rMax's
  std::string              nameSpace_;    //Namespace of this and ALL sub-parts
  std::unordered_set<int>  copies_;       //List of copy #'s
};

#endif
