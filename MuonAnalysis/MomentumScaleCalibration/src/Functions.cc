#include "MuonAnalysis/MomentumScaleCalibration/interface/Functions.h"

scaleFunctionBase<double *> * scaleFunctionService( const int identifier )
{
  switch ( identifier ) {
  case ( 0 ): return ( new scaleFunctionType0<double * > ); break;
  case ( 1 ): return ( new scaleFunctionType1<double * > ); break;
  case ( 2 ): return ( new scaleFunctionType2<double * > ); break;
  case ( 3 ): return ( new scaleFunctionType3<double * > ); break;
  case ( 4 ): return ( new scaleFunctionType4<double * > ); break;
  case ( 5 ): return ( new scaleFunctionType5<double * > ); break;
  case ( 6 ): return ( new scaleFunctionType6<double * > ); break;
  case ( 7 ): return ( new scaleFunctionType7<double * > ); break;
  case ( 8 ): return ( new scaleFunctionType8<double * > ); break;
  case ( 9 ): return ( new scaleFunctionType9<double * > ); break;
  case ( 10 ): return ( new scaleFunctionType10<double * > ); break;
  case ( 11 ): return ( new scaleFunctionType11<double * > ); break;
  case ( 12 ): return ( new scaleFunctionType12<double * > ); break;
  case ( 13 ): return ( new scaleFunctionType13<double * > ); break;
  case ( 14 ): return ( new scaleFunctionType14<double * > ); break;
  case ( 15 ): return ( new scaleFunctionType15<double * > ); break;
  case ( 16 ): return ( new scaleFunctionType16<double * > ); break;
  case ( 17 ): return ( new scaleFunctionType17<double * > ); break;
  case ( 18 ): return ( new scaleFunctionType18<double * > ); break;
  case ( 19 ): return ( new scaleFunctionType19<double * > ); break;
  case ( 20 ): return ( new scaleFunctionType20<double * > ); break;
  case ( 21 ): return ( new scaleFunctionType21<double * > ); break;
  case ( 22 ): return ( new scaleFunctionType22<double * > ); break;
  case ( 23 ): return ( new scaleFunctionType23<double * > ); break;
  case ( 24 ): return ( new scaleFunctionType24<double * > ); break;
  case ( 25 ): return ( new scaleFunctionType25<double * > ); break;
  case ( 26 ): return ( new scaleFunctionType26<double * > ); break;
  case ( 27 ): return ( new scaleFunctionType27<double * > ); break;
  case ( 28 ): return ( new scaleFunctionType28<double * > ); break;
  case ( 29 ): return ( new scaleFunctionType29<double * > ); break;
  case ( 30 ): return ( new scaleFunctionType30<double * > ); break;
  case ( 31 ): return ( new scaleFunctionType31<double * > ); break;
  case ( 32 ): return ( new scaleFunctionType32<double * > ); break;
  case ( 33 ): return ( new scaleFunctionType33<double * > ); break;
  case ( 34 ): return ( new scaleFunctionType34<double * > ); break;
  case ( 35 ): return ( new scaleFunctionType35<double * > ); break;
  case ( 36 ): return ( new scaleFunctionType36<double * > ); break;
  case ( 37 ): return ( new scaleFunctionType37<double * > ); break;
  case ( 38 ): return ( new scaleFunctionType38<double * > ); break;
  case ( 50 ): return ( new scaleFunctionType50<double * > ); break;
  case ( 51 ): return ( new scaleFunctionType51<double * > ); break;
  case ( 52 ): return ( new scaleFunctionType52<double * > ); break;
  default: std::cout << "Error: wrong identifier = " << identifier << std::endl; exit(1);
  }
}

scaleFunctionBase<std::vector<double> > * scaleFunctionVecService( const int identifier )
{
  switch ( identifier ) {
  case ( 0 ): return ( new scaleFunctionType0<std::vector<double> > ); break;
  case ( 1 ): return ( new scaleFunctionType1<std::vector<double> > ); break;
  case ( 2 ): return ( new scaleFunctionType2<std::vector<double> > ); break;
  case ( 3 ): return ( new scaleFunctionType3<std::vector<double> > ); break;
  case ( 4 ): return ( new scaleFunctionType4<std::vector<double> > ); break;
  case ( 5 ): return ( new scaleFunctionType5<std::vector<double> > ); break;
  case ( 6 ): return ( new scaleFunctionType6<std::vector<double> > ); break;
  case ( 7 ): return ( new scaleFunctionType7<std::vector<double> > ); break;
  case ( 8 ): return ( new scaleFunctionType8<std::vector<double> > ); break;
  case ( 9 ): return ( new scaleFunctionType9<std::vector<double> > ); break;
  case ( 10 ): return ( new scaleFunctionType10<std::vector<double> > ); break;
  case ( 11 ): return ( new scaleFunctionType11<std::vector<double> > ); break;
  case ( 12 ): return ( new scaleFunctionType12<std::vector<double> > ); break;
  case ( 13 ): return ( new scaleFunctionType13<std::vector<double> > ); break;
  case ( 14 ): return ( new scaleFunctionType14<std::vector<double> > ); break;
  case ( 15 ): return ( new scaleFunctionType15<std::vector<double> > ); break;
  case ( 16 ): return ( new scaleFunctionType16<std::vector<double> > ); break;
  case ( 17 ): return ( new scaleFunctionType17<std::vector<double> > ); break;
  case ( 18 ): return ( new scaleFunctionType18<std::vector<double> > ); break;
  case ( 19 ): return ( new scaleFunctionType19<std::vector<double> > ); break;
  case ( 20 ): return ( new scaleFunctionType20<std::vector<double> > ); break;
  case ( 21 ): return ( new scaleFunctionType21<std::vector<double> > ); break;
  case ( 22 ): return ( new scaleFunctionType22<std::vector<double> > ); break;
  case ( 23 ): return ( new scaleFunctionType23<std::vector<double> > ); break;
  case ( 24 ): return ( new scaleFunctionType24<std::vector<double> > ); break;
  case ( 25 ): return ( new scaleFunctionType25<std::vector<double> > ); break;
  case ( 26 ): return ( new scaleFunctionType26<std::vector<double> > ); break;
  case ( 27 ): return ( new scaleFunctionType27<std::vector<double> > ); break;
  case ( 28 ): return ( new scaleFunctionType28<std::vector<double> > ); break;
  case ( 29 ): return ( new scaleFunctionType29<std::vector<double> > ); break;  
  case ( 30 ): return ( new scaleFunctionType30<std::vector<double> > ); break;
  case ( 31 ): return ( new scaleFunctionType31<std::vector<double> > ); break;
  case ( 32 ): return ( new scaleFunctionType32<std::vector<double> > ); break;
  case ( 33 ): return ( new scaleFunctionType33<std::vector<double> > ); break;
  case ( 34 ): return ( new scaleFunctionType34<std::vector<double> > ); break;
  case ( 35 ): return ( new scaleFunctionType35<std::vector<double> > ); break;
  case ( 36 ): return ( new scaleFunctionType36<std::vector<double> > ); break;
  case ( 37 ): return ( new scaleFunctionType37<std::vector<double> > ); break;
  case ( 38 ): return ( new scaleFunctionType38<std::vector<double> > ); break;
  case ( 50 ): return ( new scaleFunctionType50<std::vector<double> > ); break;
  case ( 51 ): return ( new scaleFunctionType51<std::vector<double> > ); break;
  case ( 52 ): return ( new scaleFunctionType52<std::vector<double> > ); break;
  default: std::cout << "Error: wrong identifier = " << identifier << std::endl; exit(1);
  }
}

smearFunctionBase * smearFunctionService( const int identifier )
{
  switch ( identifier ) {
  case ( 0 ): return ( new smearFunctionType0 ); break;
  case ( 1 ): return ( new smearFunctionType1 ); break;
  case ( 2 ): return ( new smearFunctionType2 ); break;
  case ( 3 ): return ( new smearFunctionType3 ); break;
  case ( 4 ): return ( new smearFunctionType4 ); break;
  case ( 5 ): return ( new smearFunctionType5 ); break;
  case ( 6 ): return ( new smearFunctionType6 ); break;
  case ( 7 ): return ( new smearFunctionType7 ); break;
  default: std::cout << "Error: undefined smear type = " << identifier << std::endl; exit(1); break;
  }
}

resolutionFunctionBase<double *> * resolutionFunctionService( const int identifier )
{
  switch ( identifier ) {
  case ( 0 ): std::cout << "Error: resolution function type " << identifier << " not defined" << std::endl; exit(1); break;
  case ( 1 ): return ( new resolutionFunctionType1<double *> ); break;
  case ( 2 ): std::cout << "Error: resolution function type " << identifier << " not defined" << std::endl; exit(1); break;
  case ( 3 ): std::cout << "Error: resolution function type " << identifier << " not defined" << std::endl; exit(1); break;
  case ( 4 ): std::cout << "Error: resolution function type " << identifier << " not defined" << std::endl; exit(1); break;
  case ( 5 ): std::cout << "Error: resolution function type " << identifier << " not defined" << std::endl; exit(1); break;
  case ( 6 ): return ( new resolutionFunctionType6<double *> ); break;
  case ( 7 ): return ( new resolutionFunctionType7<double *> ); break;
  case ( 8 ): return ( new resolutionFunctionType8<double *> ); break;
  case ( 9 ): return ( new resolutionFunctionType9<double *> ); break;
  case ( 10 ): return ( new resolutionFunctionType10<double *> ); break;
  case ( 11 ): return ( new resolutionFunctionType11<double *> ); break;
  case ( 12 ): return ( new resolutionFunctionType12<double *> ); break;
  case ( 13 ): return ( new resolutionFunctionType13<double *> ); break;
  case ( 14 ): return ( new resolutionFunctionType14<double *> ); break;
  case ( 15 ): return ( new resolutionFunctionType15<double *> ); break;
  case ( 17 ): return ( new resolutionFunctionType17<double *> ); break;
  case ( 18 ): return ( new resolutionFunctionType18<double *> ); break;
  case ( 19 ): return ( new resolutionFunctionType19<double *> ); break;
  case ( 20 ): return ( new resolutionFunctionType20<double *> ); break;
  case ( 30 ): return ( new resolutionFunctionType30<double *> ); break;
  case ( 31 ): return ( new resolutionFunctionType31<double *> ); break;
  case ( 32 ): return ( new resolutionFunctionType32<double *> ); break;
  case ( 40 ): return ( new resolutionFunctionType40<double *> ); break;
  case ( 41 ): return ( new resolutionFunctionType41<double *> ); break;
  case ( 42 ): return ( new resolutionFunctionType42<double *> ); break;
  case ( 43 ): return ( new resolutionFunctionType43<double *> ); break;
  case ( 44 ): return ( new resolutionFunctionType44<double *> ); break;
  case ( 45 ): return ( new resolutionFunctionType45<double *> ); break;
  case ( 46 ): return ( new resolutionFunctionType46<double *> ); break;
  case ( 47 ): return ( new resolutionFunctionType47<double *> ); break;
  case ( 99 ): return ( new resolutionFunctionType99<double *> ); break;

  default: std::cout << "Error: undefined resolution type = " << identifier << std::endl; exit(1); break;
  }
}

resolutionFunctionBase<std::vector<double> > * resolutionFunctionVecService( const int identifier )
{
  switch ( identifier ) {
  case ( 0 ): std::cout << "Error: resolution function type " << identifier << " not defined" << std::endl; exit(1); break;
  case ( 1 ): return ( new resolutionFunctionType1<std::vector<double> > ); break;
  case ( 2 ): std::cout << "Error: resolution function type " << identifier << " not defined" << std::endl; exit(1); break;
  case ( 3 ): std::cout << "Error: resolution function type " << identifier << " not defined" << std::endl; exit(1); break;
  case ( 4 ): std::cout << "Error: resolution function type " << identifier << " not defined" << std::endl; exit(1); break;
  case ( 5 ): std::cout << "Error: resolution function type " << identifier << " not defined" << std::endl; exit(1); break;
  case ( 6 ): return ( new resolutionFunctionType6<std::vector<double> > ); break;
  case ( 7 ): return ( new resolutionFunctionType7<std::vector<double> > ); break;
  case ( 8 ): return ( new resolutionFunctionType8<std::vector<double> > ); break;
  case ( 9 ): return ( new resolutionFunctionType9<std::vector<double> > ); break;
  case ( 10 ): return ( new resolutionFunctionType10<std::vector<double> > ); break;
  case ( 11 ): return ( new resolutionFunctionType11<std::vector<double> > ); break;
  case ( 12 ): return ( new resolutionFunctionType12<std::vector<double> > ); break;
  case ( 13 ): return ( new resolutionFunctionType13<std::vector<double> > ); break;
  case ( 14 ): return ( new resolutionFunctionType14<std::vector<double> > ); break;
  case ( 15 ): return ( new resolutionFunctionType15<std::vector<double> > ); break;
  case ( 17 ): return ( new resolutionFunctionType17<std::vector<double> > ); break;
  case ( 18 ): return ( new resolutionFunctionType18<std::vector<double> > ); break;
  case ( 19 ): return ( new resolutionFunctionType19<std::vector<double> > ); break;
  case ( 20 ): return ( new resolutionFunctionType20<std::vector<double> > ); break;
  case ( 30 ): return ( new resolutionFunctionType30<std::vector<double> > ); break;
  case ( 31 ): return ( new resolutionFunctionType31<std::vector<double> > ); break;
  case ( 32 ): return ( new resolutionFunctionType32<std::vector<double> > ); break;
  case ( 40 ): return ( new resolutionFunctionType40<std::vector<double> > ); break; 
  case ( 41 ): return ( new resolutionFunctionType41<std::vector<double> > ); break;
  case ( 42 ): return ( new resolutionFunctionType42<std::vector<double> > ); break;
  case ( 43 ): return ( new resolutionFunctionType43<std::vector<double> > ); break;
  case ( 44 ): return ( new resolutionFunctionType44<std::vector<double> > ); break;
  case ( 45 ): return ( new resolutionFunctionType45<std::vector<double> > ); break;
  case ( 46 ): return ( new resolutionFunctionType46<std::vector<double> > ); break;
  case ( 47 ): return ( new resolutionFunctionType47<std::vector<double> > ); break;
  case ( 99 ): return ( new resolutionFunctionType99<std::vector<double> > ); break;
  
  default: std::cout << "Error: undefined resolution type = " << identifier << std::endl; exit(1); break;
    }
 }

backgroundFunctionBase * backgroundFunctionService( const int identifier, const double & lowerLimit, const double & upperLimit )
{
  switch ( identifier ) {
  case ( 0 ):  std::cout << "Error: background function type " << identifier << " not defined" << std::endl; exit(1); break;
  case ( 1 ):  return new backgroundFunctionType1(lowerLimit, upperLimit); break;
  case ( 2 ):  return new backgroundFunctionType2(lowerLimit, upperLimit); break;
  // case ( 3 ):  return new backgroundFunctionType3(lowerLimit, upperLimit); break;
  case ( 4 ):  return new backgroundFunctionType4(lowerLimit, upperLimit); break;
  case ( 5 ):  return new backgroundFunctionType5(lowerLimit, upperLimit); break;
  case ( 6 ):  return new backgroundFunctionType6(lowerLimit, upperLimit); break;
  case ( 7 ):  return new backgroundFunctionType7(lowerLimit, upperLimit); break;
  case ( 8 ):  return new backgroundFunctionType8(lowerLimit, upperLimit); break;
  case ( 9 ):  return new backgroundFunctionType9(lowerLimit, upperLimit); break; //Gul
  case ( 10 ): return new backgroundFunctionType10(lowerLimit, upperLimit); break; //Gul
  case ( 11 ): return new backgroundFunctionType11(lowerLimit, upperLimit); break; // SC
  default: std::cout << "Error: undefined background function type = " << identifier << std::endl; exit(1); break;
  }
}
