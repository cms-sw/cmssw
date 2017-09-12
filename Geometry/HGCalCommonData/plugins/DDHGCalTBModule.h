#ifndef HGCalCommonData_DDHGCalTBModule_h
#define HGCalCommonData_DDHGCalTBModule_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include <unordered_set>

class DDHGCalTBModule : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDHGCalTBModule(); //const std::string & name);
  ~DDHGCalTBModule() override;
  
  void initialize(const DDNumericArguments & nArgs,
                  const DDVectorArguments & vArgs,
                  const DDMapArguments & mArgs,
                  const DDStringArguments & sArgs,
                  const DDStringVectorArguments & vsArgs) override;
  void execute(DDCompactView& cpv) override;

protected:

  void          constructLayers (const DDLogicalPart&, DDCompactView& cpv);
  double        rMax(double z);
  void          positionSensitive(DDLogicalPart& glog, int type, double rin,
				  double rout, DDCompactView& cpv);

private:

  std::vector<std::string> wafer_;         //Wafers
  std::vector<std::string> covers_;        //Insensitive layers of hexagonal size
  std::vector<std::string> materials_;     //Materials
  std::vector<std::string> names_;         //Names
  std::vector<double>      thick_;         //Thickness of the material
  std::vector<int>         copyNumber_;    //Initial copy numbers
  std::vector<int>         layers_;        //Number of layers in a section
  std::vector<double>      layerThick_;    //Thickness of each section
  std::vector<int>         layerType_;     //Type of the layer
  std::vector<int>         layerSense_;    //COntent of a layer (sensitive?)
  double                   zMinBlock_;     //Starting z-value of the block
  double                   rMaxFine_;      //Maximum r-value for fine wafer
  double                   waferW_;        //Width of the wafer
  double                   waferGap_;      //Gap between 2 wafers
  double                   absorbW_;       //Width of the absorber
  double                   absorbH_;       //Height of the absorber
  int                      sectors_;       //Sectors   
  std::vector<double>      slopeB_;        //Slope at the lower R
  std::vector<double>      slopeT_;        //Slopes at the larger R
  std::vector<double>      zFront_;        //Starting Z values for the slopes
  std::vector<double>      rMaxFront_;     //Corresponding rMax's
  std::string              idName_;        //Name of the "parent" volume.  
  std::string              idNameSpace_;   //Namespace of this and ALL sub-parts
  std::unordered_set<int>  copies_;        //List of copy #'s
};

#endif
