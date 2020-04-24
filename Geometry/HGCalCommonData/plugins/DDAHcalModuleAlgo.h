#ifndef HGCalCommonData_DDAHcalModuleAlgo_h
#define HGCalCommonData_DDAHcalModuleAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include <unordered_set>

class DDAHcalModuleAlgo : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDAHcalModuleAlgo(); //const std::string & name);
  ~DDAHcalModuleAlgo() override;
  
  void initialize(const DDNumericArguments & nArgs,
                  const DDVectorArguments & vArgs,
                  const DDMapArguments & mArgs,
                  const DDStringArguments & sArgs,
                  const DDStringVectorArguments & vsArgs) override;
  void execute(DDCompactView& cpv) override;

protected:

  void          constructLayers (const DDLogicalPart&, DDCompactView& cpv);
  void          positionSensitive(DDLogicalPart& glog, DDCompactView& cpv);

private:

  std::string              tile;          //Scintillator tile
  std::vector<std::string> materials;     //Materials
  std::vector<std::string> names;         //Names
  std::vector<double>      thick;         //Thickness of the material
  std::vector<int>         copyNumber;    //Initial copy numbers
  std::vector<int>         layers;        //Number of layers in a section
  std::vector<double>      layerThick;    //Thickness of each section
  std::vector<int>         layerType;     //Type of the layer
  std::vector<int>         layerSense;    //Content of a layer (sensitive?)
  std::vector<double>      widths;        //Width (passive, active)
  std::vector<double>      heights;       //Heights (passive, active)
  std::vector<int>         tileN;         // # of tiles (along x, y)
  std::vector<double>      tileStep;      //Separation between tiles (x, y)
  double                   zMinBlock;     //Starting z-value of the block
  std::string              idName;        //Name of the "parent" volume.  
  std::string              idNameSpace;   //Namespace of this and ALL sub-parts
  std::unordered_set<int>  copies;        //List of copy #'s
};

#endif
