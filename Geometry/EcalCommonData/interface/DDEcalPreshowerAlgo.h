#ifndef DDEcalPreshowerAlgo_h
#define DDEcalPreshowerAlgo_h

#include <vector>
#include <string>

#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"

class DDEcalPreshowerAlgo : public DDAlgorithm {

public:
  
  DDMaterial getMaterial(unsigned int i)   const {return DDMaterial(materials_[i]);}
  DDMaterial getLaddMaterial()  const { return DDMaterial(LaddMaterial_) ; }
  std::string getLayName(unsigned int i)   const {return layName_[i];}
  std::string getLadPrefix(unsigned int i)   const {return ladPfx_[i];}

  DDEcalPreshowerAlgo();
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs) override;
  void execute(DDCompactView& pos) override;

private:

  void doLayers(DDCompactView& pos);
  void doLadders(DDCompactView& pos); 
  void doSens(DDCompactView& pos);
  
  int nmat_;                       // number of preshower layers
  double thickness_;               // overall thickness of the preshower envelope
  std::vector<std::string> materials_;  // materials of the presh-layers
  std::vector<std::string> layName_;    // names of the presh-layers
  std::vector<std::string> ladPfx_ ;    // name prefix for ladders
  std::string LaddMaterial_;            // ladd material - air
  std::vector<double> thickLayers_; 
  std::vector<double> abs1stx;
  std::vector<double> abs1sty;
  std::vector<double> abs2ndx;
  std::vector<double> abs2ndy;
  std::vector<double> asym_ladd_;
  std::vector<double> rminVec; 
  std::vector<double> rmaxVec;
  std::vector<double> noLaddInCol_;
  std::vector<double> startOfFirstLadd_;
  std::vector<std::string> types_l5_;
  std::vector<std::string> types_l4_;
  std::vector<double> ladd_l5_map_;
  std::vector<double> ladd_l4_map_;
  std::vector<std::string> typeOfLaddRow0;
  std::vector<std::string> typeOfLaddRow1;
  std::vector<std::string> typeOfLaddRow2;
  std::vector<std::string> typeOfLaddRow3;

  double zlead1_, zlead2_, zfoam1_, zfoam2_;
  double waf_intra_col_sep, waf_inter_col_sep, waf_active, wedge_length, wedge_offset, zwedge_ceramic_diff, ywedge_ceramic_diff, wedge_angle, box_thick,dee_separation, In_rad_Abs_Al, In_rad_Abs_Pb;
  double ladder_thick, yladder_1stwedge_diff, ladder_width, ladder_length, micromodule_length;
  double absAlX_X_, absAlX_Y_, absAlX_subtr1_Xshift_, absAlX_subtr1_Yshift_, rMax_Abs_Al_;
  double absAlY_X_, absAlY_Y_, absAlY_subtr1_Xshift_, absAlY_subtr1_Yshift_;
  double LdrBck_Length, LdrFrnt_Length,LdrFrnt_Offset,LdrBck_Offset, ceramic_length, wedge_back_thick;
  
};


#endif // DDEcalPreshowerAlgo_h
