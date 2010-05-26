#include "MuonAnalysis/MomentumScaleCalibration/interface/Functions.h"

scaleFunctionBase<double * > * scaleFunctionService( const int identifier )
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
  default: cout << "Error: wrong identifier = " << identifier << endl; exit(1);
  }
}

scaleFunctionBase<vector<double> > * scaleFunctionVecService( const int identifier )
{
  switch ( identifier ) {
  case ( 0 ): return ( new scaleFunctionType0<vector<double> > ); break;
  case ( 1 ): return ( new scaleFunctionType1<vector<double> > ); break;
  case ( 2 ): return ( new scaleFunctionType2<vector<double> > ); break;
  case ( 3 ): return ( new scaleFunctionType3<vector<double> > ); break;
  case ( 4 ): return ( new scaleFunctionType4<vector<double> > ); break;
  case ( 5 ): return ( new scaleFunctionType5<vector<double> > ); break;
  case ( 6 ): return ( new scaleFunctionType6<vector<double> > ); break;
  case ( 7 ): return ( new scaleFunctionType7<vector<double> > ); break;
  case ( 8 ): return ( new scaleFunctionType8<vector<double> > ); break;
  case ( 9 ): return ( new scaleFunctionType9<vector<double> > ); break;
  case ( 10 ): return ( new scaleFunctionType10<vector<double> > ); break;
  case ( 11 ): return ( new scaleFunctionType11<vector<double> > ); break;
  case ( 12 ): return ( new scaleFunctionType12<vector<double> > ); break;
  case ( 13 ): return ( new scaleFunctionType13<vector<double> > ); break;
  case ( 14 ): return ( new scaleFunctionType14<vector<double> > ); break;
  case ( 15 ): return ( new scaleFunctionType15<vector<double> > ); break;
  case ( 16 ): return ( new scaleFunctionType16<vector<double> > ); break;
  case ( 17 ): return ( new scaleFunctionType17<vector<double> > ); break;
  case ( 18 ): return ( new scaleFunctionType18<vector<double> > ); break;
  case ( 19 ): return ( new scaleFunctionType19<vector<double> > ); break;  
  default: cout << "Error: wrong identifier = " << identifier << endl; exit(1);
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
  default: cout << "Error: undefined smear type = " << identifier << endl; exit(1); break;
  }
}

resolutionFunctionBase<double *> * resolutionFunctionService( const int identifier )
{
  switch ( identifier ) {
  case ( 0 ): cout << "Error: resolution function type " << identifier << " not defined" << endl; exit(1); break;
  case ( 1 ): return ( new resolutionFunctionType1<double *> ); break;
  case ( 2 ): cout << "Error: resolution function type " << identifier << " not defined" << endl; exit(1); break;
  case ( 3 ): cout << "Error: resolution function type " << identifier << " not defined" << endl; exit(1); break;
  case ( 4 ): cout << "Error: resolution function type " << identifier << " not defined" << endl; exit(1); break;
  case ( 5 ): cout << "Error: resolution function type " << identifier << " not defined" << endl; exit(1); break;
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
  default: cout << "Error: undefined resolution type = " << identifier << endl; exit(1); break;
  }
}

resolutionFunctionBase<vector<double> > * resolutionFunctionVecService( const int identifier )
{
  switch ( identifier ) {
  case ( 0 ): cout << "Error: resolution function type " << identifier << " not defined" << endl; exit(1); break;
  case ( 1 ): return ( new resolutionFunctionType1<vector<double> > ); break;
  case ( 2 ): cout << "Error: resolution function type " << identifier << " not defined" << endl; exit(1); break;
  case ( 3 ): cout << "Error: resolution function type " << identifier << " not defined" << endl; exit(1); break;
  case ( 4 ): cout << "Error: resolution function type " << identifier << " not defined" << endl; exit(1); break;
  case ( 5 ): cout << "Error: resolution function type " << identifier << " not defined" << endl; exit(1); break;
  case ( 6 ): return ( new resolutionFunctionType6<vector<double> > ); break;
  case ( 7 ): return ( new resolutionFunctionType7<vector<double> > ); break;
  case ( 8 ): return ( new resolutionFunctionType8<vector<double> > ); break;
  case ( 9 ): return ( new resolutionFunctionType9<vector<double> > ); break;
  case ( 10 ): return ( new resolutionFunctionType10<vector<double> > ); break;
  case ( 11 ): return ( new resolutionFunctionType11<vector<double> > ); break;
  case ( 12 ): return ( new resolutionFunctionType12<vector<double> > ); break;
  case ( 13 ): return ( new resolutionFunctionType13<vector<double> > ); break;
  case ( 14 ): return ( new resolutionFunctionType14<vector<double> > ); break;
  case ( 15 ): return ( new resolutionFunctionType15<vector<double> > ); break;
  case ( 17 ): return ( new resolutionFunctionType17<vector<double> > ); break;
  case ( 18 ): return ( new resolutionFunctionType18<vector<double> > ); break;
  default: cout << "Error: undefined resolution type = " << identifier << endl; exit(1); break;
  }
}

backgroundFunctionBase * backgroundFunctionService( const int identifier )
{
  switch ( identifier ) {
  case ( 0 ): cout << "Error: background function type " << identifier << " not defined" << endl; exit(1); break;
  case ( 1 ): return new backgroundFunctionType1; break;
  case ( 2 ): return new backgroundFunctionType2; break;
  case ( 3 ): return new backgroundFunctionType3; break;
  default: cout << "Error: undefined background function type = " << identifier << endl; exit(1); break;
  }
}
