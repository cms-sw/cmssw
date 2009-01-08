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
  default: cout << "Error: wrong identifier = " << identifier << endl; exit(1);
  }
}
