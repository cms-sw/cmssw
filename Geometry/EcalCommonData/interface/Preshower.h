#ifndef DD_ALGO_PLUGIN_DD_PRESHOWER_H
# define DD_ALGO_PLUGIN_DD_PRESHOWER_H

#include <vector>

#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDD.h"

class Preshower : public DDAlgorithm
{
 public:
    Preshower();
    void initialize(const DDNumericArguments & nArgs,
                    const DDVectorArguments & vArgs,
                    const DDMapArguments & mArgs,
		    const DDStringArguments & sArgs,
		    const DDStringVectorArguments & vsArgs);
    void execute();
 private:
    void doLayers();
    void doWedges();

    std::vector<double> quadMin_, quadMax_; 
    int nmat_; // number of preshower layers
    double thickness_; // overall thickness of the preshower envelope
    double zlead1_, zlead2_, zfoam1_, zfoam2_;
    std::vector<DD::Material> materials_; // materials of the presh-layers
    std::vector<double> thickLayers_;

};


#endif // DD_ALGO_PLUGIN_DD_PRESHOWER_H
