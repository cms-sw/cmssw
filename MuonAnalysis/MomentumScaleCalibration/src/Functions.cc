#include "MuonAnalysis/MomentumScaleCalibration/interface/Functions.h"

scaleFunctionBase<double *> * scaleFunctionService( const int identifier )
{
  switch ( identifier ) {
  case (  0 ): return ( new scaleFunctionType0<double * > ); break;
  case ( 50 ): return ( new scaleFunctionType50<double * > ); break;
  case ( 64 ): return ( new scaleFunctionType64<double * > ); break;
  default: std::cout << "Error: wrong identifier = " << identifier << std::endl; exit(1);
  }
}

scaleFunctionBase<std::vector<double> > * scaleFunctionVecService( const int identifier )
{
  switch ( identifier ) {
  case (  0 ): return ( new scaleFunctionType0<std::vector<double> > ); break;
  case ( 50 ): return ( new scaleFunctionType50<std::vector<double> > ); break;
  case ( 64 ): return ( new scaleFunctionType64<std::vector<double> > ); break;
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
  case (  0 ): return ( new resolutionFunctionType0<double *> ); break;
  case ( 45 ): return ( new resolutionFunctionType45<double *> ); break;
  case ( 46 ): return ( new resolutionFunctionType46<double *> ); break;
  case ( 47 ): return ( new resolutionFunctionType47<double *> ); break;

  default: std::cout << "Error: undefined resolution type = " << identifier << std::endl; exit(1); break;
  }
}

resolutionFunctionBase<std::vector<double> > * resolutionFunctionVecService( const int identifier )
{
  switch ( identifier ) {
  case (  0 ): return ( new resolutionFunctionType0<std::vector<double> > ); break;
  case ( 45 ): return ( new resolutionFunctionType45<std::vector<double> > ); break;
  case ( 46 ): return ( new resolutionFunctionType46<std::vector<double> > ); break;
  case ( 47 ): return ( new resolutionFunctionType47<std::vector<double> > ); break;
  
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
