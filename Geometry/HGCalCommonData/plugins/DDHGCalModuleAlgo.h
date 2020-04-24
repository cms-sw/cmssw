#ifndef HGCalCommonData_DDHGCalModuleAlgo_h
#define HGCalCommonData_DDHGCalModuleAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include <unordered_set>

class DDHGCalModuleAlgo : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDHGCalModuleAlgo(); //const std::string & name);
  ~DDHGCalModuleAlgo() override;
  
  void initialize(const DDNumericArguments & nArgs,
                  const DDVectorArguments & vArgs,
                  const DDMapArguments & mArgs,
                  const DDStringArguments & sArgs,
                  const DDStringVectorArguments & vsArgs) override;
  void execute(DDCompactView& cpv) override;

protected:

  void          constructLayers (const DDLogicalPart&, DDCompactView& cpv);
  double        rMax(double z);
  void          positionSensitive(DDLogicalPart& glog, double rin, double rout,
				  DDCompactView& cpv);

private:

  std::vector<std::string> wafer;         //Wafers
  std::vector<std::string> materials;     //Materials
  std::vector<std::string> names;         //Names
  std::vector<double>      thick;         //Thickness of the material
  std::vector<int>         copyNumber;    //Initial copy numbers
  std::vector<int>         layers;        //Number of layers in a section
  std::vector<double>      layerThick;    //Thickness of each section
  std::vector<int>         layerType;     //Type of the layer
  std::vector<int>         layerSense;    //COntent of a layer (sensitive?)
  double                   zMinBlock;     //Starting z-value of the block
  double                   rMaxFine;      //Maximum r-value for fine wafer
  double                   waferW;        //Width of the wafer
  double                   waferGap;      //Gap between 2 wafers
  int                      sectors;       //Sectors   
  std::vector<double>      slopeB;        //Slope at the lower R
  std::vector<double>      slopeT;        //Slopes at the larger R
  std::vector<double>      zFront;        //Starting Z values for the slopes
  std::vector<double>      rMaxFront;     //Corresponding rMax's
  std::string              idName;        //Name of the "parent" volume.  
  std::string              idNameSpace;   //Namespace of this and ALL sub-parts
  std::unordered_set<int>  copies;        //List of copy #'s
};

#endif
