#ifndef L1Trigger_TrackFindingTracklet_interface_Util_h
#define L1Trigger_TrackFindingTracklet_interface_Util_h

#include <sstream>
#include <cassert>
#include <cmath>
#include <array>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

namespace trklet {       


  //Converts string in binary to hex (used in writing out memory content)
  inline std::string hexFormat(const std::string& binary) {
    std::string tmp = "";

    unsigned int value = 0;

    for (unsigned int i = 0; i < binary.size(); i++) {
      unsigned int slot = binary.size() - i - 1;
      if (!(binary[slot] == '0' || binary[slot] == '1'))
        continue;
      value = value + (binary[slot] - '0');
    }

    std::stringstream ss;
    ss << "0x" << std::hex << value;

    return ss.str();
  }
 inline std::array<unsigned int, N_LAYER> irmean1_{{851, 1269, 1784, 2347, 2936, 3697}};
 inline std::array<unsigned int, N_DISK> izmean1_{{2239, 2645, 3163, 3782, 4523}};
 inline double zlength1_ = 120.0;
 inline  double rmaxdisk1_ = 120.0;
 inline double rmean1(unsigned int iLayer)  { return irmean1_[iLayer] * rmaxdisk1_ / 4096; }
 inline double zmean1(unsigned int iDisk)  { return izmean1_[iDisk] * zlength1_ / 2048; }


 






 
  //Should be optimized by layer - now first implementation to make sure it works OK
  inline int bendencode(double bend, bool isPS) {
    int ibend = 2.0 * bend;

    assert(std::abs(ibend - 2.0 * bend) < 0.1);

    if (isPS) {
      if (ibend == 0 || ibend == 1)
        return 0;
      if (ibend == 2 || ibend == 3)
        return 1;
      if (ibend == 4 || ibend == 5)
        return 2;
      if (ibend >= 6)
        return 3;
      if (ibend == -1 || ibend == -2)
        return 4;
      if (ibend == -3 || ibend == -4)
        return 5;
      if (ibend == -5 || ibend == -6)
        return 6;
      if (ibend <= -7)
        return 7;

      throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__
                                        << " Unknown bendencode for PS module for bend = " << bend
                                        << " ibend = " << ibend;
    }

    if (ibend == 0 || ibend == 1)
      return 0;
    if (ibend == 2 || ibend == 3)
      return 1;
    if (ibend == 4 || ibend == 5)
      return 2;
    if (ibend == 6 || ibend == 7)
      return 3;
    if (ibend == 8 || ibend == 9)
      return 4;
    if (ibend == 10 || ibend == 11)
      return 5;
    if (ibend == 12 || ibend == 13)
      return 6;
    if (ibend >= 14)
      return 7;
    if (ibend == -1 || ibend == -2)
      return 8;
    if (ibend == -3 || ibend == -4)
      return 9;
    if (ibend == -5 || ibend == -6)
      return 10;
    if (ibend == -7 || ibend == -8)
      return 11;
    if (ibend == -9 || ibend == -10)
      return 12;
    if (ibend == -11 || ibend == -12)
      return 13;
    if (ibend == -13 || ibend == -14)
      return 14;
    if (ibend <= -15)
      return 15;

    throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__
                                      << " Unknown bendencode for 2S module for bend = " << bend
                                      << " ibend = " << ibend;
  }

  //Should be optimized by layer - now first implementation to make sure it works OK
  inline double benddecode(int ibend, bool isPS) {
    if (isPS) {
      if (ibend == 0)
        return 0.25;
      if (ibend == 1)
        return 1.25;
      if (ibend == 2)
        return 2.25;
      if (ibend == 3)
        return 3.25;
      if (ibend == 4)
        return -0.75;
      if (ibend == 5)
        return -1.75;
      if (ibend == 6)
        return -2.75;
      if (ibend == 7)
        return -3.75;

      throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__
                                        << " Unknown benddecode for PS module for ibend = " << ibend;
    }

    if (ibend == 0)
      return 0.25;
    if (ibend == 1)
      return 1.25;
    if (ibend == 2)
      return 2.25;
    if (ibend == 3)
      return 3.25;
    if (ibend == 4)
      return 4.25;
    if (ibend == 5)
      return 5.25;
    if (ibend == 6)
      return 6.25;
    if (ibend == 7)
      return 7.25;
    if (ibend == 8)
      return -0.75;
    if (ibend == 9)
      return -1.75;
    if (ibend == 10)
      return -2.75;
    if (ibend == 11)
      return -3.75;
    if (ibend == 12)
      return -4.75;
    if (ibend == 13)
      return -5.75;
    if (ibend == 14)
      return -6.75;
    if (ibend == 15)
      return -7.75;

    throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__
                                      << " Unknown benddecode for 2S module for ibend = " << ibend;
  }

  inline double cosModuleTilt = 0.886454;
  inline double sinModuleTilt = 0.504148;

  inline double bendDisk_PR(double r, int disk, double rinv, double stripPitch){
    double dr = 0.18;
    double z = zmean1(disk - 1);
    if (((disk ==1 || disk ==2) && r<=Settings::diskSpacingCut[0]) || ((disk==3 || disk ==4) && r<=Settings::diskSpacingCut[1]) || (disk==5 && r<=Settings::diskSpacingCut[2])){
      dr = 0.4;
    }
    double CF = r/z;
    double delta=r*dr*0.5*rinv*CF;
    double bend = -delta/stripPitch;




    return bend;
  }

 inline  double bendDisk_TE(double r, int disk, double rinv, double stripPitch){
    double dr = 0.18;
    double z = zmean1(disk -1);
    if (((disk==1 || disk==2) && r<=Settings::diskSpacingCut[0]) || ((disk==3 || disk==4) && r<=Settings::diskSpacingCut[1]) || (disk==5 && r<=Settings::diskSpacingCut[2])){
      dr = 0.4;
    }
   double CF = r/z;
   double delta=r*dr*0.5*rinv*CF;
   double bend = delta/stripPitch;




   return bend;
  }

  inline double bendBarrel_TE(double z, int layer, double rinv, double stripPitch){
   
    double dr = 0.18;
    double CF =1;
  
    double r=rmean1(layer-1);
 
    if ((layer ==1 && z<=Settings::barrelSpacingCut[3]) || (layer==2 && Settings::barrelSpacingCut[1] <=z && z<=Settings::barrelSpacingCut[4]) || (layer==3  && Settings::barrelSpacingCut[3] <=z && z<=Settings::barrelSpacingCut[5])){
      dr = 0.26;
    }
    else if ((layer==1 && Settings::barrelSpacingCut[2]<=z && z<=Settings::barrelSpacingCut[5]) || (layer==2 && Settings::barrelSpacingCut[4]<=z && z<=Settings::barrelSpacingCut[5])){
      dr = 0.4;
    }
    else if ((layer==2 && z<=Settings::barrelSpacingCut[1]) || (layer==3 && z<=Settings::barrelSpacingCut[3])){
      dr =0.16;
    }
    if ((layer==1 && Settings::barrelSpacingCut[0]<=z && z<=Settings::barrelSpacingCut[5]) || (layer==2 && Settings::barrelSpacingCut[1] <=z && z<=Settings::barrelSpacingCut[5]) || (layer==3 && Settings::barrelSpacingCut[3]<=z && z<=Settings::barrelSpacingCut[5])){
      CF = cosModuleTilt*(z/r) + sinModuleTilt;
    }
    double delta = r*dr*0.5*rinv;
    double bend = delta/(stripPitch*CF);





    return bend;
  
}

  inline double bendBarrel_ME(double z, int layer, double rinv, double stripPitch){
    double dr = 0.18;
    double CF=1;
    double r = rmean1(layer-1);
    if ((layer==1 && z<=Settings::barrelSpacingCut[3]) || (layer==2 && Settings::barrelSpacingCut[1]<=z && z<=Settings::barrelSpacingCut[4]) || (layer==3 && Settings::barrelSpacingCut[3]<=z && z<=Settings::barrelSpacingCut[5])){
      dr = 0.26;
    }
    else if ((layer==1 && Settings::barrelSpacingCut[2]<=z && z<=Settings::barrelSpacingCut[5]) || (layer==2 && Settings::barrelSpacingCut[4]<=z && z<=Settings::barrelSpacingCut[5])){
      dr = 0.4;
    }
    else if ((layer==2 && z<=Settings::barrelSpacingCut[1]) || (layer==3 && z<=Settings::barrelSpacingCut[3])){
      dr = 0.16;
    }
    if ((layer==1 && Settings::barrelSpacingCut[0]<=z && z<=Settings::barrelSpacingCut[5]) || (layer==2 && Settings::barrelSpacingCut[1] <=z && z<=Settings::barrelSpacingCut[5]) || (layer==3 && Settings::barrelSpacingCut[3]<=z && z<=Settings::barrelSpacingCut[5])){
      CF = cosModuleTilt*(z/r) + sinModuleTilt;
    }
    double delta = r*dr*0.5*rinv;
    double bend = -delta/(stripPitch*CF);




    return bend;
  }




 

  inline double rinv(double phi1, double phi2, double r1, double r2) {
    if (r2 <= r1) {  //can not form tracklet
      return 20.0;
    }

    double dphi = phi2 - phi1;
    double dr = r2 - r1;

    return 2.0 * sin(dphi) / dr / sqrt(1.0 + 2 * r1 * r2 * (1.0 - cos(dphi)) / (dr * dr));
  }

};  // namespace trklet
#endif
