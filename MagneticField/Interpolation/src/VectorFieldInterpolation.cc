// include header for VectorFieldInterpolation
#include "VectorFieldInterpolation.h"


void VectorFieldInterpolation::defineCellPoint000(double X1, double X2, double X3, double F1, double F2, double F3){
  CellPoint000[0] = X1;
  CellPoint000[1] = X2;
  CellPoint000[2] = X3;
  CellPoint000[3] = F1;
  CellPoint000[4] = F2;
  CellPoint000[5] = F3;
  return;
}


void VectorFieldInterpolation::defineCellPoint100(double X1, double X2, double X3, double F1, double F2, double F3){
  CellPoint100[0] = X1;
  CellPoint100[1] = X2;
  CellPoint100[2] = X3;
  CellPoint100[3] = F1;
  CellPoint100[4] = F2;
  CellPoint100[5] = F3;
  return;
}


void VectorFieldInterpolation::defineCellPoint010(double X1, double X2, double X3, double F1, double F2, double F3){
  CellPoint010[0] = X1;
  CellPoint010[1] = X2;
  CellPoint010[2] = X3;
  CellPoint010[3] = F1;
  CellPoint010[4] = F2;
  CellPoint010[5] = F3;
  return;
}


void VectorFieldInterpolation::defineCellPoint110(double X1, double X2, double X3, double F1, double F2, double F3){
  CellPoint110[0] = X1;
  CellPoint110[1] = X2;
  CellPoint110[2] = X3;
  CellPoint110[3] = F1;
  CellPoint110[4] = F2;
  CellPoint110[5] = F3;
  return;
}


void VectorFieldInterpolation::defineCellPoint001(double X1, double X2, double X3, double F1, double F2, double F3){
  CellPoint001[0] = X1;
  CellPoint001[1] = X2;
  CellPoint001[2] = X3;
  CellPoint001[3] = F1;
  CellPoint001[4] = F2;
  CellPoint001[5] = F3;
  return;
}


void VectorFieldInterpolation::defineCellPoint101(double X1, double X2, double X3, double F1, double F2, double F3){
  CellPoint101[0] = X1;
  CellPoint101[1] = X2;
  CellPoint101[2] = X3;
  CellPoint101[3] = F1;
  CellPoint101[4] = F2;
  CellPoint101[5] = F3;
  return;
}


void VectorFieldInterpolation::defineCellPoint011(double X1, double X2, double X3, double F1, double F2, double F3){
  CellPoint011[0] = X1;
  CellPoint011[1] = X2;
  CellPoint011[2] = X3;
  CellPoint011[3] = F1;
  CellPoint011[4] = F2;
  CellPoint011[5] = F3;
  return;
}


void VectorFieldInterpolation::defineCellPoint111(double X1, double X2, double X3, double F1, double F2, double F3){
  CellPoint111[0] = X1;
  CellPoint111[1] = X2;
  CellPoint111[2] = X3;
  CellPoint111[3] = F1;
  CellPoint111[4] = F2;
  CellPoint111[5] = F3;
  return;
}


void VectorFieldInterpolation::putSCoordGetVField(double X1, double X2, double X3, double &F1, double &F2, double &F3){
  SC[0] = X1;
  SC[1] = X2;
  SC[2] = X3;

  // values describing 4 help points after interpolation step of variables X1
  // 5 dimensions: 2 space dimensions + 3 field dimensions
  //                        X2' , X3' , F1' , F2' , F3'
  double HelpPoint00[5]; // {0.0 , 0.0 , 0.0 , 0.0 , 0.0};
  double HelpPoint10[5];
  double HelpPoint01[5];
  double HelpPoint11[5];

  // values describing 2 help points after interpolation step of variables X2'
  // 4 dimensions: 1 space dimensions + 3 field dimensions
  //                       X3" , F1" , F2" , F3"
  double HelpPoint0[4]; // {0.0 , 0.0 , 0.0 , 0.0};
  double HelpPoint1[4];


  // 1. iteration *****
  // prepare interpolation between CellPoint000 and CellPoint100
  double DeltaX100X000 = CellPoint100[0] - CellPoint000[0];
  double DeltaSC0X000overDeltaX100X000 = 0.;
  if (DeltaX100X000 != 0.) DeltaSC0X000overDeltaX100X000 = (SC[0] - CellPoint000[0]) / DeltaX100X000;

  // prepare interpolation between CellPoint010 and CellPoint110
  double DeltaX110X010 = CellPoint110[0] - CellPoint010[0];
  double DeltaSC0X010overDeltaX110X010 = 0.;
  if (DeltaX110X010 != 0.) DeltaSC0X010overDeltaX110X010 = (SC[0] - CellPoint010[0]) / DeltaX110X010;

  // prepare interpolation between CellPoint001 and CellPoint101
  double DeltaX101X001 = CellPoint101[0] - CellPoint001[0];
  double DeltaSC0X001overDeltaX101X001 = 0.;
  if (DeltaX101X001 != 0.) DeltaSC0X001overDeltaX101X001 = (SC[0] - CellPoint001[0]) / DeltaX101X001;

  // prepare interpolation between CellPoint011 and CellPoint111
  double DeltaX111X011 = CellPoint111[0] - CellPoint011[0];
  double DeltaSC0X011overDeltaX111X011 = 0.;
  if (DeltaX111X011 != 0.) DeltaSC0X011overDeltaX111X011 = (SC[0] - CellPoint011[0]) / DeltaX111X011;

  for (int i=0; i<5; ++i){
    int ip = i+1; 

    // interpolate between CellPoint000 and CellPoint100
    HelpPoint00[i] = CellPoint000[ip] + DeltaSC0X000overDeltaX100X000 * (CellPoint100[ip] - CellPoint000[ip]);

    // interpolate between CellPoint010 and CellPoint110
    HelpPoint10[i] = CellPoint010[ip] + DeltaSC0X010overDeltaX110X010 * (CellPoint110[ip] - CellPoint010[ip]);

    // interpolate between CellPoint001 and CellPoint101
    HelpPoint01[i] = CellPoint001[ip] + DeltaSC0X001overDeltaX101X001 * (CellPoint101[ip] - CellPoint001[ip]);

    // interpolate between CellPoint011 and CellPoint111
    HelpPoint11[i] = CellPoint011[ip] + DeltaSC0X011overDeltaX111X011 * (CellPoint111[ip] - CellPoint011[ip]);
  }

  // 2. iteration *****
  // prepare interpolation between HelpPoint00 and HelpPoint10
  double DeltaX10X00 = HelpPoint10[0] - HelpPoint00[0];
  double DeltaSC1X00overDeltaX10X00 = 0.;
  if (DeltaX10X00 != 0.) DeltaSC1X00overDeltaX10X00 = (SC[1] - HelpPoint00[0]) / DeltaX10X00;
  // prepare interpolation between HelpPoint01 and HelpPoint11
  double DeltaX11X01 = HelpPoint11[0] - HelpPoint01[0];
  double DeltaSC1X01overDeltaX11X01 = 0.;
  if (DeltaX11X01 != 0.) DeltaSC1X01overDeltaX11X01 = (SC[1] - HelpPoint01[0]) / DeltaX11X01;

  for (int i=0; i<4; ++i){
    int ip = i+1; 

    // interpolate between HelpPoint00 and HelpPoint10
    HelpPoint0[i] = HelpPoint00[ip] + DeltaSC1X00overDeltaX10X00 * (HelpPoint10[ip] - HelpPoint00[ip]);

    // interpolate between HelpPoint01 and HelpPoint11
    HelpPoint1[i] = HelpPoint01[ip] + DeltaSC1X01overDeltaX11X01 * (HelpPoint11[ip] - HelpPoint01[ip]);
  }

  // 3. iteration *****
  // prepare interpolation between HelpPoint0 and HelpPoint1
  double DeltaX1X0 = HelpPoint1[0] - HelpPoint0[0];
  double DeltaSC2X0overDeltaX1X0 = 0.;
  if (DeltaX1X0 != 0.) DeltaSC2X0overDeltaX1X0 = (SC[2] - HelpPoint0[0]) / DeltaX1X0;

  for (int i=0; i<3; ++i){
    int ip = i+1; 

    // interpolate between HelpPoint0 and HelpPoint1
    VF[i] = HelpPoint0[ip] + DeltaSC2X0overDeltaX1X0 * (HelpPoint1[ip] - HelpPoint0[ip]);
  }

  F1 = VF[0];
  F2 = VF[1];
  F3 = VF[2];

  return;
}
