#ifndef HGCalCommonData_DDHGCalTBModuleX_h
#define HGCalCommonData_DDHGCalTBModuleX_h

#include <string>
#include <unordered_set>
#include <vector>
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDTypes.h"

class DDHGCalTBModuleX : public DDAlgorithm {
public:
  // Constructor and Destructor
  DDHGCalTBModuleX();  //
  ~DDHGCalTBModuleX() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;
  void execute(DDCompactView& cpv) override;

protected:
  void constructBlocks(const DDLogicalPart&, DDCompactView& cpv);
  void constructLayers(int block,
                       int layerFront,
                       int layerBack,
                       double zFront,
                       double thick,
                       bool ignore,
                       const DDLogicalPart&,
                       DDCompactView&);
  void positionSensitive(double zpos,
                         int copyIn,
                         int type,
                         double rmax,
                         int ncrMax,
                         bool ignoreCenter,
                         const std::string&,
                         const DDMaterial&,
                         const DDLogicalPart&,
                         DDCompactView& cpv);

private:
  static constexpr double tolerance_ = 0.00001;
  const double factor_, tan30deg_;

  std::vector<std::string> wafer_;      // Wafers
  std::vector<std::string> covers_;     // Insensitive layers of hexagonal size
  std::string genMat_;                  // General material used for blocks
  std::vector<std::string> materials_;  // Material names in each layer
  std::vector<std::string> names_;      // Names of each layer
  std::vector<double> layerThick_;      // Thickness of the material
  std::vector<int> copyNumber_;         // Copy numbers (initiated to 1)
  std::vector<double> blockThick_;      // Thickness of each section
  int inOut_;                           // Number of inner+outer parts
  std::vector<int> layerFrontIn_;       // First layer index (inner) in block
  std::vector<int> layerBackIn_;        // Last layer index (inner) in block
  std::vector<int> layerFrontOut_;      // First layer index (outner) in block
  std::vector<int> layerBackOut_;       // Last layer index (outner) in block
  std::vector<int> layerType_;          // Type of the layer
  std::vector<int> layerSense_;         // Content of a layer
  std::vector<int> maxModule_;          // Maximum # of row/column
  double zMinBlock_;                    // Starting z-value of the block
  double rMaxFine_;                     // Maximum r-value for fine wafer
  double waferW_;                       // Width of the wafer
  double waferGap_;                     // Gap between 2 wafers
  double absorbW_;                      // Width of the absorber
  double absorbH_;                      // Height of the absorber
  double rMax_;                         // Maximum radial extent
  double rMaxB_;                        // Maximum radial extent of a block
  std::string idName_;                  // Name of the "parent" volume.
  std::string idNameSpace_;             // Namespace of this and ALL sub-parts
  std::unordered_set<int> copies_;      // List of copy #'s
};

#endif
