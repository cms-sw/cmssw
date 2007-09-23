//
//
// This class provides the means of reading
// and storing the information of the name of
// a ROC accoriding to the naming document.
//
// A name is on the form
//
// FPix_B{p,m}{L,R}_D{1,2}_BLD{1..24}_PNL{1,2}_PLQ{1,2,3,4}_ROC{1..10} 
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelNameROC.h"
#include <string>


PixelNameROC::PixelNameROC(std::string name){
}

std::ostream& operator<<(std::ostream& s, const PixelNameROC& name){

    s<<FPix_B<<porm_<<LorR_<<"_D"<<(int)disk_<<_BLD"<<(int)blade_
     <<"_PNL"<<(int)panel_<<"_PLQ"<<(int)plaquet_<<"_ROC"<<(int)roc_;
 
    return s;
}

int PixelNameROC::operator==(PixelROC& aroc) const{

  if (roc_!=aroc.roc_) return 0;
  if (plaquet_!=aroc.plaquet_) return 0;
  if (panel_!=aroc.panel_) return 0;
  if (blade_!=aroc.blade_) return 0;
  if (disk_!=aroc.disk_) return 0;
  if (LorR_!=aroc.LorR_) return 0;
  if (porm_!=aroc.porm_) return 0;

}
