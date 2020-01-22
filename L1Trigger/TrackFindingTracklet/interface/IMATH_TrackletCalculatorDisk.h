#ifndef IMATH_TRACKLETCALCULATORDISK_H
#define IMATH_TRACKLETCALCULATORDISK_H

#include "Constants.h"
#include "imath.h"

//
// Constants used:
//   dphisector
//   rmaxL6
//   zmaxD5
//   rmaxdisk
//   kr, kphi1, kz   
//
//   rmean[], zmean[]

class IMATH_TrackletCalculatorDisk {
  
public:
  
  IMATH_TrackletCalculatorDisk(int i1, int i2)
  {

#ifndef CMSSW_GIT_HASH
    printf("=============================================\n");
    printf("IMATH Tracklet Calculator for Disk %i %i",i1,i2);
    printf("dphisector = %f\n",dphisector);
    printf("rmaxL6 = %f, zmaxD5 = %f\n",rmaxL6, zmaxD5);
    printf("      stub Ks: kr, kphi1, kz = %g, %g, %g\n",kr, kphi1, kz);
    printf("  tracklet Ks: krinvpars, kphi0pars, ktpars, kzpars = %g, %g, %g, %g\n",
	   kphi1/kr*pow(2,rinv_shift), kphi1*pow(2,phi0_shift), kz/kr*pow(2,t_shift), kz*pow(2,z0_shift));
    printf("layer proj Ks: kphiproj456, kphider, kzproj, kzder = %g, %g, %g, %g\n",
	   kphi1*pow(2,SS_phiL_shift), kphi1/kr*pow(2,SS_phiderL_shift), kz*pow(2,PS_zL_shift),kz/kr*pow(2,PS_zderL_shift));
    printf(" disk proj Ks: kphiprojdisk, kphiprojderdisk, krprojdisk, krprojderdisk = %g, %g, %g, %g\n",
	   kphi1*pow(2,SS_phiD_shift), kphi1/kr*pow(2,SS_phiderD_shift), kr*pow(2,PS_rD_shift),kr/kz*pow(2,PS_rderD_shift));
    printf("=============================================\n");
#endif 

    z1mean.set_fval(zmean[abs(i1)-1]);
    z2mean.set_fval(zmean[abs(i2)-1]);

    if(i2<0){ //t is negative
      z1mean.set_fval(-zmean[abs(i1)-1]);
      z2mean.set_fval(-zmean[abs(i2)-1]);
      invt.set_mode(var_inv::mode::neg);
      invt.initLUT(0.);
    }
    
    valid_phiL_0.add_cut(&t_layer_cut);
    valid_phiL_1.add_cut(&t_layer_cut);
    valid_phiL_2.add_cut(&t_layer_cut);

    valid_der_phiL.add_cut(&t_layer_cut);

    valid_zL_0.add_cut(&t_layer_cut);
    valid_zL_1.add_cut(&t_layer_cut);
    valid_zL_2.add_cut(&t_layer_cut);

    valid_der_zL.add_cut(&t_layer_cut);

    valid_phiD_0.add_cut(&t_disk_cut_left);
    valid_phiD_1.add_cut(&t_disk_cut_left);
    valid_phiD_2.add_cut(&t_disk_cut_left);

    valid_der_phiD.add_cut(&t_disk_cut_left);

    valid_rD_0.add_cut(&t_disk_cut_left);
    valid_rD_1.add_cut(&t_disk_cut_left);
    valid_rD_2.add_cut(&t_disk_cut_left);

    valid_der_rD.add_cut(&t_disk_cut_left);

    valid_phiD_0.add_cut(&t_disk_cut_right);
    valid_phiD_1.add_cut(&t_disk_cut_right);
    valid_phiD_2.add_cut(&t_disk_cut_right);

    valid_der_phiD.add_cut(&t_disk_cut_right);

    valid_rD_0.add_cut(&t_disk_cut_right);
    valid_rD_1.add_cut(&t_disk_cut_right);
    valid_rD_2.add_cut(&t_disk_cut_right);

    valid_der_rD.add_cut(&t_disk_cut_right);
  }
  
  //max values
  double     dr_max = 20.;
  double delta0_max = 0.005;
  double    a2a_max = 0.1;
  double     x8_max = 1.;
  double    x22_max = 0.3;
  double    x13_max = 300.;
  double deltaZ_max = 8.;
  double der_phiD_max = 0.002;
  
  // constants 
  //
  var_param plus2{"plus2",2.,10};
  var_param plus1{"plus1",1.,10};
  var_param minus1{"minus1",-1,10};
  //
  //
  var_param z1mean{"z1mean","Kz",zmaxD5, kz}; 
  var_param z2mean{"z2mean","Kz",zmaxD5, kz};
    
  
  //inputs
  var_def r1{"r1","Kr",rmaxL6,kr};
  var_def r2{"r2","Kr",rmaxL6,kr};
  var_def z1{"z1","Kz",dzmax,kz};
  var_def z2{"z2","Kz",dzmax,kz};
  
  var_def phi1{"phi1","Kphi",dphisector/0.75, kphi1};
  var_def phi2{"phi2","Kphi",dphisector/0.75, kphi1};

  var_def rproj0{"rproj0","Kr",rmaxL6, kr};
  var_def rproj1{"rproj1","Kr",rmaxL6, kr};
  var_def rproj2{"rproj2","Kr",rmaxL6, kr};

  var_def zproj0{"zproj0","Kz",zmaxD5, kz};
  var_def zproj1{"zproj1","Kz",zmaxD5, kz};
  var_def zproj2{"zproj2","Kz",zmaxD5, kz};

  //calculations

  //tracklet
  var_add z1abs{"z1abs",&z1,&z1mean, zmaxD5};
  var_add z2abs{"z2abs",&z2,&z2mean, zmaxD5};

  var_subtract dr{"dr",&r2,&r1,dr_max};
  
  //R LUT
  var_inv  drinv{"drinv",&dr, 0, 18, 23, 0, var_inv::mode::pos};
  
  var_subtract dphi{"dphi",&phi2,&phi1,dphisector/4.};
  var_subtract dz{"dz",&z2abs,&z1abs, 50.};
  
  var_mult delta0{"delta0",&dphi, &drinv, 4*delta0_max};
  var_mult deltaZ{"deltaZ",&dz,   &drinv, deltaZ_max};
  var_mult delta1{"delta1",&r1, &delta0};
  var_mult delta2{"delta2",&r2, &delta0};
  var_mult a2a{"a2a",&delta1, &delta2, 4*a2a_max};
  var_nounits a2b{"a2b",&a2a};
  var_subtract a2{"a2",&plus2,&a2b,3.};
  var_neg   a2n{"a2n",&a2};
  var_shift a{"a",&a2,1};

  var_add Rabs{"Rabs",&r1,&r2};
  var_timesC R6{"R6",&Rabs,1./6.,12};

  var_mult x4 {"x4",&R6, &delta0};
  var_mult x6a{"x6a",&delta2,&x4, 0.16};
  var_nounits x6b{"x6b",&x6a};
  var_add  x6m{"x6m",&minus1,&x6b, 2.};
  var_mult phi0a{"phi0a",&delta1,&x6m, dphisector};

  var_mult     z0a{"z0a",&r1, &deltaZ, 240.};
  var_mult     z0b{"z0b",&z0a, &x6m,   240.};
  
  var_add  phi0{"phi0",&phi1,&phi0a, 2*dphisector};
  var_mult rinv{"rinv",&a2n, &delta0, 4*maxrinv};
  var_mult t{"t",&a, &deltaZ, 15.8};
  var_add  z0{"z0",&z1abs,&z0b,40.};

  var_adjustK rinv_final{"rinv_final",&rinv, kphi1/kr*pow(2,rinv_shift)};
  var_adjustK phi0_final{"phi0_final",&phi0, kphi1*pow(2,phi0_shift)};
  var_adjustK t_final{"t_final",&t,          kz/kr*pow(2,t_shift)};
  var_adjustK z0_final{"z0_final",&z0,       kz*pow(2,z0_shift)};

  //projection to r
  //
  var_shift  x2{"x2",&delta0,1};

  var_mult   x1_0{"x1_0",&x2,&rproj0};
  var_mult   x1_1{"x1_1",&x2,&rproj1};
  var_mult   x1_2{"x1_2",&x2,&rproj2};

  var_mult   x8_0{"x8_0",&x1_0,&a2n, x8_max};
  var_mult   x8_1{"x8_1",&x1_1,&a2n, x8_max};
  var_mult   x8_2{"x8_2",&x1_2,&a2n, x8_max};

  var_mult    x12_0{"x12_0",&x8_0,&x8_0};
  var_mult    x12_1{"x12_1",&x8_1,&x8_1};
  var_mult    x12_2{"x12_2",&x8_2,&x8_2};

  var_nounits x12A_0{"x12A_0",&x12_0};
  var_nounits x12A_1{"x12A_1",&x12_1};
  var_nounits x12A_2{"x12A_2",&x12_2};

  var_timesC x20_0{"x20_0",&x12A_0,1./6.};
  var_timesC x20_1{"x20_1",&x12A_1,1./6.};
  var_timesC x20_2{"x20_2",&x12A_2,1./6.};

  var_add     x10_0{"x10_0",&plus1,&x20_0};
  var_add     x10_1{"x10_1",&plus1,&x20_1};
  var_add     x10_2{"x10_2",&plus1,&x20_2};

  var_mult    x22_0{"x22_0",&x8_0,&x10_0, 2*x22_max};
  var_mult    x22_1{"x22_1",&x8_1,&x10_1, 2*x22_max};
  var_mult    x22_2{"x22_2",&x8_2,&x10_2, 2*x22_max};

  var_subtract phiL_0{"phiL_0",&phi0_final, &x22_0, -1, phi0_final.get_nbits()+1};
  var_subtract phiL_1{"phiL_1",&phi0_final, &x22_1, -1, phi0_final.get_nbits()+1};
  var_subtract phiL_2{"phiL_2",&phi0_final, &x22_2, -1, phi0_final.get_nbits()+1};

  var_shift x3{"x3",&rinv,1};
  var_neg der_phiL{"der_phiL",&x3};

  var_adjustK phiL_0_final{"phiL_0_final",&phiL_0,               kphi1*pow(2,SS_phiL_shift)};
  var_adjustK phiL_1_final{"phiL_1_final",&phiL_1,               kphi1*pow(2,SS_phiL_shift)};
  var_adjustK phiL_2_final{"phiL_2_final",&phiL_2,               kphi1*pow(2,SS_phiL_shift)};

  var_adjustK der_phiL_final{"der_phiL_final",&der_phiL,   kphi1/kr*pow(2,SS_phiderL_shift)};

  var_mult   x11_0{"x11_0",&rproj0, &t};
  var_mult   x11_1{"x11_1",&rproj1, &t};
  var_mult   x11_2{"x11_2",&rproj2, &t};

  var_mult   x23_0{"x23_0",&x11_0,&x10_0, 800};
  var_mult   x23_1{"x23_1",&x11_1,&x10_1, 800};
  var_mult   x23_2{"x23_2",&x11_2,&x10_2, 800};

  var_add    zL_0{"zL_0",&z0,&x23_0};
  var_add    zL_1{"zL_1",&z0,&x23_1};
  var_add    zL_2{"zL_2",&z0,&x23_2};

  var_adjustK zL_0_final{"zL_0_final",&zL_0,                     kz*pow(2,PS_zL_shift)};
  var_adjustK zL_1_final{"zL_1_final",&zL_1,                     kz*pow(2,PS_zL_shift)};
  var_adjustK zL_2_final{"zL_2_final",&zL_2,                     kz*pow(2,PS_zL_shift)};

  var_adjustK der_zL_final{"der_zL_final",&t_final,        kz/kr*pow(2,PS_zderL_shift)};

  
  //projection to z
  //
  var_inv   invt{"invt",&t_final, 0., 18, 26, 1, var_inv::mode::pos, 13};

  var_mult       x7{"x7",&x2, &a2};
  
  var_subtract   x5_0{"x5_0",&zproj0,&z0};
  var_subtract   x5_1{"x5_1",&zproj1,&z0};
  var_subtract   x5_2{"x5_2",&zproj2,&z0};

  var_mult      x13_0{"x13_0",&x5_0,&invt, x13_max};
  var_mult      x13_1{"x13_1",&x5_1,&invt, x13_max};
  var_mult      x13_2{"x13_2",&x5_2,&invt, x13_max};

  var_mult      x25_0{"x25_0",&x13_0,&x7, dphisector};
  var_mult      x25_1{"x25_1",&x13_1,&x7, dphisector};
  var_mult      x25_2{"x25_2",&x13_2,&x7, dphisector};

  var_add      phiD_0{"phiD_0",&phi0,&x25_0, 2*dphisector};
  var_add      phiD_1{"phiD_1",&phi0,&x25_1, 2*dphisector};
  var_add      phiD_2{"phiD_2",&phi0,&x25_2, 2*dphisector};

  var_adjustK phiD_0_final{"phiD_0_final",&phiD_0,     kphi1*pow(2,SS_phiD_shift)};
  var_adjustK phiD_1_final{"phiD_1_final",&phiD_1,     kphi1*pow(2,SS_phiD_shift)};
  var_adjustK phiD_2_final{"phiD_2_final",&phiD_2,     kphi1*pow(2,SS_phiD_shift)};
  
  var_mult der_phiD{"der_phiD",&x7, &invt, 2*der_phiD_max};

  var_adjustK der_phiD_final{"der_phiD_final",&der_phiD, kphi1/kr*pow(2,SS_phiderD_shift)};

  var_mult      x26_0{"x26_0",&x25_0,&x25_0};
  var_mult      x26_1{"x26_1",&x25_1,&x25_1};
  var_mult      x26_2{"x26_2",&x25_2,&x25_2};

  var_nounits  x26A_0{"x26A_0",&x26_0};
  var_nounits  x26A_1{"x26A_1",&x26_1};
  var_nounits  x26A_2{"x26A_2",&x26_2};

  var_timesC     x9_0{"x9_0",&x26A_0,1./6.};
  var_timesC     x9_1{"x9_1",&x26A_1,1./6.};
  var_timesC     x9_2{"x9_2",&x26A_2,1./6.};

  var_subtract x27_0{"x27_0",&plus1,&x9_0};
  var_subtract x27_1{"x27_1",&plus1,&x9_1};
  var_subtract x27_2{"x27_2",&plus1,&x9_2};

  var_mult       rD_0{"rD_0",&x13_0, &x27_0, rmaxdisk};
  var_mult       rD_1{"rD_1",&x13_1, &x27_1, rmaxdisk};
  var_mult       rD_2{"rD_2",&x13_2, &x27_2, rmaxdisk};

  var_adjustK rD_0_final{"rD_0_final",&rD_0,                      kr*pow(2,PS_rD_shift)};
  var_adjustK rD_1_final{"rD_1_final",&rD_1,                      kr*pow(2,PS_rD_shift)};
  var_adjustK rD_2_final{"rD_2_final",&rD_2,                      kr*pow(2,PS_rD_shift)};

  var_adjustK der_rD_final{"der_rD_final",&invt,            kr/kz*pow(2,PS_rderD_shift)};

  var_cut rinv_final_cut{&rinv_final,-rinvcut,rinvcut};
  var_cut z0_final_cut{&z0_final,-z0cut,z0cut};

  var_cut z1abs_cut{&z1abs,-zmaxD5,zmaxD5};
  var_cut z2abs_cut{&z2abs,-zmaxD5,zmaxD5};
  var_cut dr_cut{&dr,-dr_max,dr_max};
  var_cut dphi_cut{&dphi,-dphisector/4.,dphisector/4.};
  var_cut dz_cut{&dz,-50.,50.};
  var_cut delta0_cut{&delta0,-delta0_max,delta0_max};
  var_cut deltaZ_cut{&deltaZ,-deltaZ_max,deltaZ_max};
  var_cut a2a_cut{&a2a,-a2a_max,a2a_max};
  var_cut a2_cut{&a2,-3.,3.};
  var_cut x6a_cut{&x6a,-0.02,0.02};
  var_cut x6m_cut{&x6m,-2.,2.};
  var_cut phi0a_cut{&phi0a,-dphisector,dphisector};
  var_cut z0a_cut{&z0a,-205,205};
  var_cut phi0_cut{&phi0,-2*dphisector,2*dphisector};
  var_cut rinv_cut{&rinv,-maxrinv,maxrinv};
  var_cut t_cut{&t,-7.9,7.9};
  var_cut z0_cut{&z0,-20.,20.};
  var_cut x8_0_cut{&x8_0,-x8_max,x8_max};
  var_cut x8_1_cut{&x8_1,-x8_max,x8_max};
  var_cut x8_2_cut{&x8_2,-x8_max,x8_max};
  var_cut x22_0_cut{&x22_0,-x22_max,x22_max};
  var_cut x22_1_cut{&x22_1,-x22_max,x22_max};
  var_cut x22_2_cut{&x22_2,-x22_max,x22_max};
  var_cut x23_0_cut{&x23_0,-200,200};
  var_cut x23_1_cut{&x23_1,-200,200};
  var_cut x23_2_cut{&x23_2,-200,200};
  var_cut x13_0_cut{&x13_0,-x13_max,x13_max};
  var_cut x13_1_cut{&x13_1,-x13_max,x13_max};
  var_cut x13_2_cut{&x13_2,-x13_max,x13_max};
  var_cut x25_0_cut{&x25_0,-dphisector,dphisector};
  var_cut x25_1_cut{&x25_1,-dphisector,dphisector};
  var_cut x25_2_cut{&x25_2,-dphisector,dphisector};
  var_cut phiD_0_cut{&phiD_0,-2*dphisector,2*dphisector};
  var_cut phiD_1_cut{&phiD_1,-2*dphisector,2*dphisector};
  var_cut phiD_2_cut{&phiD_2,-2*dphisector,2*dphisector};
  var_cut der_phiD_cut{&der_phiD,-der_phiD_max,der_phiD_max};
  var_cut rD_0_cut{&rD_0,-rmaxdisk,rmaxdisk};
  var_cut rD_1_cut{&rD_1,-rmaxdisk,rmaxdisk};
  var_cut rD_2_cut{&rD_2,-rmaxdisk,rmaxdisk};

  var_cut t_disk_cut_left{&t,-7.9,-1};
  var_cut t_disk_cut_right{&t,1,7.9};
  var_cut t_layer_cut{&t,-2.5,2.5};

  // the following flags are used to apply the cuts in TrackletCalculator
  // and in the output Verilog
  var_flag valid_trackpar{"valid_trackpar",&rinv_final,&phi0_final,&t_final,&z0_final};

  var_flag valid_phiL_0{"valid_phiL_0",&phiL_0_final};
  var_flag valid_phiL_1{"valid_phiL_1",&phiL_1_final};
  var_flag valid_phiL_2{"valid_phiL_2",&phiL_2_final};

  var_flag valid_zL_0{"valid_zL_0",&zL_0_final};
  var_flag valid_zL_1{"valid_zL_1",&zL_1_final};
  var_flag valid_zL_2{"valid_zL_2",&zL_2_final};

  var_flag valid_der_phiL{"valid_der_phiL",&der_phiL_final};
  var_flag valid_der_zL{"valid_der_zL",&der_zL_final};

  var_flag valid_phiD_0{"valid_phiD_0",&phiD_0_final};
  var_flag valid_phiD_1{"valid_phiD_1",&phiD_1_final};
  var_flag valid_phiD_2{"valid_phiD_2",&phiD_2_final};

  var_flag valid_rD_0{"valid_rD_0",&rD_0_final};
  var_flag valid_rD_1{"valid_rD_1",&rD_1_final};
  var_flag valid_rD_2{"valid_rD_2",&rD_2_final};

  var_flag valid_der_phiD{"valid_der_phiD",&der_phiD_final};
  var_flag valid_der_rD{"valid_der_rD",&der_rD_final};
  
};

#endif
