#ifndef DDEcalPreshowerAlgoTB_h
#define DDEcalPreshowerAlgoTB_h

#include <vector>
#include <string>

#include "DetectorDescription/Core/interface/DDAlgorithm.h"

class DDEcalPreshowerAlgoTB : public DDAlgorithm {

public:
  DDEcalPreshowerAlgoTB();
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;
  void execute(DDCompactView& pos) override;

private:
  std::string getMaterial(unsigned int i)   const {return materials_[i];}
  void doLayers(DDCompactView& pos);
  void doWedges(DDCompactView& pos);    
  void doSens(DDCompactView& pos);

  std::vector<double> quadMin_, quadMax_; 
  int nmat_; // number of preshower layers
  double thickness_; // overall thickness of the preshower envelope
  double zlead1_, zlead2_, zfoam1_, zfoam2_;
  std::vector<std::string> materials_; // materials of the presh-layers
  std::vector<double> thickLayers_; 
  std::vector<double> rminVec; 
  std::vector<double> rmaxVec;
  std::vector<double> micromodulesx;
  std::vector<double> micromodulesy;
  std::string dummyMaterial;
  std::string   idNameSpace; //Namespace of this and ALL sub-parts
  double waf_intra_col_sep, waf_inter_col_sep, waf_active, wedge_length, wedge_offset, zwedge_ceramic_diff, ywedge_ceramic_diff, absorbx, absorby, trabsorbx, trabsorby, ScndplaneXshift, ScndplaneYshift, TotSFXshift, TotSFYshift;
  int go;
};

#endif // DDEcalPreshowerAlgoTB_h
