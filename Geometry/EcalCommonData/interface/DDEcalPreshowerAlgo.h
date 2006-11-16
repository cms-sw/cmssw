#ifndef DDEcalPreshowerAlgo_h
#define DDEcalPreshowerAlgo_h

#include <vector>
#include <string>

#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDEcalPreshowerAlgo : public DDAlgorithm
{
 public:
    std::string getMaterial(unsigned int i)   const {return materials_[i];}
    DDEcalPreshowerAlgo();
    void initialize(const DDNumericArguments & nArgs,
                    const DDVectorArguments & vArgs,
                    const DDMapArguments & mArgs,
		    const DDStringArguments & sArgs,
		    const DDStringVectorArguments & vsArgs);
    void execute();
 private:
    void doLayers();
    void doWedges();    
    void doSens();

    std::vector<double> quadMin_, quadMax_; 
    int nmat_; // number of preshower layers
    double thickness_; // overall thickness of the preshower envelope
    double zlead1_, zlead2_, zfoam1_, zfoam2_;
    std::vector<std::string> materials_; // materials of the presh-layers
    std::vector<double> thickLayers_; 
    std::vector<double> rminVec; 
    std::vector<double> rmaxVec;
    double waf_intra_col_sep, waf_inter_col_sep, waf_active, wedge_length, wedge_offset, zwedge_ceramic_diff, ywedge_ceramic_diff; 
};


#endif // DDEcalPreshowerAlgo_h
