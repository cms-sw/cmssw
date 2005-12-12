//   COCOA class implementation file
//Id:  ALIUtils.cc
//CAT: ALIUtils
//
//   History: v1.0 
//   Pedro Arce

#include "OpticalAlignment/CocoaUtilities/interface/ALIUtils.h"
#include "OpticalAlignment/CocoaUtilities/interface/GlobalOptionMgr.h"

#include <math.h>
#include <stdlib.h>
#include <iomanip>


ALIint ALIUtils::debug = -1;
ALIint ALIUtils::report = 1;
ALIdouble ALIUtils::_LengthValueDimensionFactor;
ALIdouble ALIUtils::_LengthSigmaDimensionFactor;
ALIdouble ALIUtils::_AngleValueDimensionFactor;
ALIdouble ALIUtils::_AngleSigmaDimensionFactor;
ALIdouble ALIUtils::_OutputLengthValueDimensionFactor;
ALIdouble ALIUtils::_OutputLengthSigmaDimensionFactor;
ALIdouble ALIUtils::_OutputAngleValueDimensionFactor;
ALIdouble ALIUtils::_OutputAngleSigmaDimensionFactor;
time_t ALIUtils::_time_now;
ALIdouble ALIUtils::deg = 0.017453293;
ALIbool ALIUtils::firstTime;
ALIdouble ALIUtils::maximum_deviation_derivative = 1.E-6;

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ CHECKS THAT EVERY CHARACTER IN A STRING IS NUMBER, ELSE GIVES AN ERROR
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
int ALIUtils::IsNumber( const ALIstring& str)
{
  int isnum = 1;
  int numE = 0;
  for(uint ii=0; ii<str.length(); ii++){
    if(!isdigit(str[ii]) && str[ii]!='.' && str[ii]!='-' && str[ii]!='+') {
      //--- check for E(xponential)
      if(str[ii] == 'E' || str[ii] == 'e' ) {
        if(numE != 0 || ii == str.length()-1)  {
	  isnum = 0;
	  break;
	}
	numE++;
      } else {
	isnum = 0; 
	break;
      }
    }
  }
 
  return isnum;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Dump a Hep3DVector with the chosen precision
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#include "CLHEP/Units/SystemOfUnits.h"
void ALIUtils::dump3v( const Hep3Vector& vec, const std::string& msg) 
{
  //  double phicyl = atan( vec.y()/vec.x() );
  std::cout << msg << std::setprecision(8) << vec;
  std::cout << std::endl;
  //  std::cout << " " << vec.theta()/deg << " " << vec.phi()/deg << " " << vec.perp() << " " << phicyl/deg << std::endl; 
  //  setw(10);
  //  std::cout << msg << " x=" << std::setprecision(8) << vec.x() << " y=" << setprecision(8) <<vec.y() << " z=" << std::setprecision(8) << vec.z() << std::endl;
  // std::setprecision(8);

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void ALIUtils::dumprm( const HepRotation& rm, const std::string& msg, std::ostream& out) 
{

  out << msg << " xx=" << rm.xx() << " xy=" << rm.xy() << " xz=" << rm.xz() << std::endl;
  out << msg << " yx=" << rm.yx() << " yy=" << rm.yy() << " yz=" << rm.yz() << std::endl;
  out << msg << " zx=" << rm.zx() << " zy=" << rm.zy() << " zz=" << rm.zz() << std::endl;

}
 

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Set the dimension factor to convert input length values and errors to
//@@  the dimension of errors
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void ALIUtils::SetLengthDimensionFactors()
{
//---------------------------------------- if it doesn exist, GlobalOptions is 0
  //---------- Calculate factors to convert to meters
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  ALIint ad = ALIint(gomgr->getGlobalOption("length_value_dimension"));

  _LengthValueDimensionFactor = CalculateLengthDimensionFactorFromInt( ad );
  ad =  ALIint(gomgr->GlobalOptions()[ ALIstring("length_error_dimension"
) ]);
  _LengthSigmaDimensionFactor = CalculateLengthDimensionFactorFromInt( ad );

  //---------- Change factor to convert to error dimensions
  //  _LengthValueDimensionFactor /= _LengthSigmaDimensionFactor;
  //_LengthSigmaDimensionFactor = 1;

  if(ALIUtils::debug >= 6) std::cout <<  _LengthValueDimensionFactor << " Set Length DimensionFactors " << _LengthSigmaDimensionFactor << std::endl; 
   
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Set the dimension factor to convert input angle values and errors to 
//@@  the dimension of errors
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void ALIUtils::SetAngleDimensionFactors()
{
//--------------------- if it doesn exist, GlobalOptions is 0
  //---------- Calculate factors to convert to radians
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  ALIint ad =  ALIint(gomgr->GlobalOptions()[ ALIstring("angle_value_dimension") ]);
  _AngleValueDimensionFactor = CalculateAngleDimensionFactorFromInt(ad);

  ad =  ALIint(gomgr->GlobalOptions()[ ALIstring("angle_error_dimension"
) ]);
  _AngleSigmaDimensionFactor = CalculateAngleDimensionFactorFromInt(ad);

  //---------- Change factor to convert to error dimensions
  //  _AngleValueDimensionFactor /= _AngleSigmaDimensionFactor;
  //_AngleSigmaDimensionFactor = 1;

  if(ALIUtils::debug >= 6) std::cout <<  _AngleValueDimensionFactor <<  "Set Angle DimensionFactors" << _AngleSigmaDimensionFactor << std::endl; 
   
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Set the dimension factor to convert input length values and errors to
//@@  the dimension of errors
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void ALIUtils::SetOutputLengthDimensionFactors()
{
//---------------------------------------- if it doesn exist, GlobalOptions is 0
  //---------- Calculate factors to convert to meters
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  ALIint ad =  ALIint(gomgr->GlobalOptions()[ ALIstring("output_length_value_dimension") ]);
  if( ad == 0 ) ad =  ALIint(gomgr->GlobalOptions()[ ALIstring("length_value_dimension") ]);
  _OutputLengthValueDimensionFactor = CalculateLengthDimensionFactorFromInt( ad );

  ad =  ALIint(gomgr->GlobalOptions()[ ALIstring("output_length_error_dimension"
) ]);
  if( ad == 0 ) ad =  ALIint(gomgr->GlobalOptions()[ ALIstring("length_error_dimension"
) ]);
  _OutputLengthSigmaDimensionFactor = CalculateLengthDimensionFactorFromInt( ad );

  //---------- Change factor to convert to error dimensions
  //  _LengthValueDimensionFactor /= _LengthSigmaDimensionFactor;
  //_LengthSigmaDimensionFactor = 1;

  if(ALIUtils::debug >= 6) std::cout <<  _OutputLengthValueDimensionFactor << "Output Length Dimension Factors" << _OutputLengthSigmaDimensionFactor << std::endl; 
   
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Set the dimension factor to convert input angle values and errors to 
//@@  the dimension of errors
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void ALIUtils::SetOutputAngleDimensionFactors()
{
//--------------------- if it doesn exist, GlobalOptions is 0
  //---------- Calculate factors to convert to radians
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  ALIint ad =  ALIint(gomgr->GlobalOptions()[ ALIstring("output_angle_value_dimension") ]);
  if( ad == 0 ) ad =  ALIint(gomgr->GlobalOptions()[ ALIstring("angle_value_dimension") ]);
  _OutputAngleValueDimensionFactor = CalculateAngleDimensionFactorFromInt(ad);


  ad =  ALIint(gomgr->GlobalOptions()[ ALIstring("output_angle_error_dimension") ]);
  if( ad == 0) ad =  ALIint(gomgr->GlobalOptions()[ ALIstring("angle_error_dimension") ]);
  _OutputAngleSigmaDimensionFactor = CalculateAngleDimensionFactorFromInt(ad);

  //---------- Change factor to convert to error dimensions
  //  _AngleValueDimensionFactor /= _AngleSigmaDimensionFactor;
  //_AngleSigmaDimensionFactor = 1;

  if(ALIUtils::debug >= 9) std::cout <<  _OutputAngleValueDimensionFactor <<  "Output Angle Dimension Factors" << _OutputAngleSigmaDimensionFactor << std::endl; 
   
}



//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Calculate dimension factor to convert any length values and errors to meters
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIdouble ALIUtils::CalculateLengthDimensionFactorFromString( ALIstring dimstr )
{
  ALIdouble valsig = 1.;
  ALIstring internalDim = "m";
  if(internalDim == "m" ){
    if( dimstr == "m" ) {
      valsig = 1.;
    }else if( dimstr == "mm" ) {
      valsig = 1.E-3;
    }else if( dimstr == "mum" ) {
      valsig = 1.E-6;
    }else if( dimstr == "cm" ) {
      valsig = 1.E-2;
    }else {
      std::cerr << "!!! UNKNOWN DIMENSION SCALING " << dimstr << std::endl <<
	"VALUE MUST BE BETWEEN 0 AND 3 " << std::endl;
      exit(1);
    }
  }else if(internalDim == "mm" ){
    if( dimstr == "m" ) {
      valsig = 1.E3;
    }else if( dimstr == "mm" ) {
      valsig = 1.;
    }else if( dimstr == "mum" ) {
      valsig = 1.E-3;
    }else if( dimstr == "cm" ) {
      valsig = 1.E+1;
    }else {
      std::cerr << "!!! UNKNOWN DIMENSION SCALING: " << dimstr << std::endl <<
	"VALUE MUST BE A LENGTH DIMENSION " << std::endl;
      exit(1);
    }
  }

  return valsig;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Calculate dimension factor to convert any angle values and errors to radians
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIdouble ALIUtils::CalculateAngleDimensionFactorFromString( ALIstring dimstr )
{
  ALIdouble valsig;
  if( dimstr == "rad" ) {
    valsig = 1.;
  }else 
  if( dimstr == "mrad" ) {
    valsig = 1.E-3;
  }else 
  if( dimstr == "murad" ) {
    valsig = 1.E-6;
  }else 
  if( dimstr == "deg" ) {
    valsig = M_PI/180.;
  }else 
  if( dimstr == "grad" ) {
    valsig = M_PI/200.;
  }else {
    std::cerr << "!!! UNKNOWN DIMENSION SCALING: " << dimstr << std::endl <<
      "VALUE MUST BE AN ANGLE DIMENSION " << std::endl;
    exit(1);
  }

  return valsig;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Calculate dimension factor to convert any length values and errors to meters
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIdouble ALIUtils::CalculateLengthDimensionFactorFromInt( ALIint ad )
{
  ALIdouble valsig;
  switch ( ad ) {
    case 0:                  //----- metres
      valsig = CalculateLengthDimensionFactorFromString( "m" );
      break;
    case 1:                  //----- milimetres
      valsig = CalculateLengthDimensionFactorFromString( "mm" );
      break;
    case 2:                  //----- micrometres
      valsig = CalculateLengthDimensionFactorFromString( "mum" );
      break;
    case 3:                  //----- centimetres
      valsig = CalculateLengthDimensionFactorFromString( "cm" );
      break;
    default:
      std::cerr << "!!! UNKNOWN DIMENSION SCALING " << ad << std::endl <<
              "VALUE MUST BE BETWEEN 0 AND 3 " << std::endl;
      exit(1);
  }

  // use microradinas instead of radians
  //-  valsig *= 1000000.;

  return valsig;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Calculate dimension factor to convert any angle values and errors to radians
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIdouble ALIUtils::CalculateAngleDimensionFactorFromInt( ALIint ad )
{
  ALIdouble valsig;
  switch ( ad ) {
    case 0:                  //----- radians
      valsig = CalculateAngleDimensionFactorFromString( "rad" );
      break;
    case 1:                  //----- miliradians
      valsig = CalculateAngleDimensionFactorFromString( "mrad" );
      break;
    case 2:                  //----- microradians
      valsig = CalculateAngleDimensionFactorFromString( "murad" );
      break;
    case 3:                  //----- degrees
      valsig = CalculateAngleDimensionFactorFromString( "deg" );
      break;
    case 4:                  //----- grads
      valsig = CalculateAngleDimensionFactorFromString( "grad" );
      break;
    default:
      std::cerr << "!!! UNKNOWN DIMENSION SCALING " << ad << std::endl <<
              "VALUE MUST BE BETWEEN 0 AND 3 " << std::endl;
      exit(1);
  }

  // use microradinas instead of radians
  //-  valsig *= 1000000.;

  return valsig;
}

/*template<class T>
 ALIuint FindItemInVector( const T* item, const std::vector<T*> std::vector )
{

}
*/

void ALIUtils::dumpDimensions( std::ofstream& fout ) 
{
  fout << "DIMENSIONS: lengths = ";
  ALIstring internalDim = "m";
  if(_OutputLengthValueDimensionFactor == 1. ) { 
    fout << "m";
  }else if(_OutputLengthValueDimensionFactor == 1.E-3 ) { 
    fout << "mm";
  }else if(_OutputLengthValueDimensionFactor == 1.E-6 ) { 
    fout << "mum";
  }else if(_OutputLengthValueDimensionFactor == 1.E-2 ) { 
    fout << "cm";
  } else {
    std::cerr << " !! unknown OutputLengthValueDimensionFactor " << _OutputLengthValueDimensionFactor << std::endl;
    exit(1);
  }

  fout << " +- ";
  if(_OutputLengthSigmaDimensionFactor == 1. ) { 
    fout << "m";
  }else if(_OutputLengthSigmaDimensionFactor == 1.E-3 ) { 
    fout << "mm";
  }else if(_OutputLengthSigmaDimensionFactor == 1.E-6 ) { 
    fout << "mum";
  }else if(_OutputLengthSigmaDimensionFactor == 1.E-2 ) { 
    fout << "cm";
  } else {
    std::cerr << " !! unknown OutputLengthSigmaDimensionFactor " << _OutputLengthSigmaDimensionFactor << std::endl;
    exit(1);
  }
    
  fout << "  angles = ";
  if(_OutputAngleValueDimensionFactor == 1. ) { 
    fout << "rad";
  }else if(_OutputAngleValueDimensionFactor == 1.E-3 ) { 
    fout << "mrad";
  }else if(_OutputAngleValueDimensionFactor == 1.E-6 ) { 
    fout << "murad";
  }else if(_OutputAngleValueDimensionFactor == M_PI/180. ) { 
    fout << "deg";
  }else if(_OutputAngleValueDimensionFactor == M_PI/200. ) { 
    fout << "grad";
  } else {
    std::cerr << " !! unknown OutputAngleValueDimensionFactor " << _OutputAngleValueDimensionFactor << std::endl;
    exit(1);
  }

  fout << " +- ";
  if(_OutputAngleSigmaDimensionFactor == 1. ) { 
    fout << "rad";
  }else if(_OutputAngleSigmaDimensionFactor == 1.E-3 ) { 
    fout << "mrad";
  }else if(_OutputAngleSigmaDimensionFactor == 1.E-6 ) { 
    fout << "murad";
  }else if(_OutputAngleSigmaDimensionFactor == M_PI/180. ) { 
    fout << "deg";
  }else if(_OutputAngleSigmaDimensionFactor == M_PI/200. ) { 
    fout << "grad";
  } else {
    std::cerr << " !! unknown OutputAngleSigmaDimensionFactor " << _OutputAngleSigmaDimensionFactor << std::endl;
    exit(1);
  }
  fout << std::endl;

}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
double ALIUtils::getFloat( const ALIstring& str ) 
{
  //----------- first check that it is a number
  if(!IsNumber(str) ) {
    std::cerr << "!!!! EXITING: trying to get the float from a string that is not a number " << str << std::endl;
    exit(1);
  }

  return atof( str.c_str() );
}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int ALIUtils::getInt( const ALIstring& str ) 
{
  //----------- first check that it is an integer
  if(!IsNumber(str) ) {
    //----- Check that it is a number 
    std::cerr << "!!!! EXITING: trying to get the integer from a string that is not a number " << str << std::endl;
    exit(1);
  } else {
    //----- Check that it is not a float, no decimal or E-n
    bool isFloat = 0;
    int ch = str.find('.');
    uint ii = 0;
    if(ch != -1 ) {
      for( ii = ch+1; ii < str.size(); ii++) {
	if( str[ii] != '0' ) isFloat = 1;
      }
    }

    ch = str.find('E');
    if(ch != -1 ) ch = str.find('e');
    if(ch != -1 ) {
      if(str[ch+1] == '-') isFloat = 1;
    }

    if(isFloat) {
      std::cerr << "!!!! EXITING: trying to get the integer from a string that is a float: " << str << std::endl;
      std::cerr << ii << " ii "  << ch <<std::endl;
      exit(1);
    }
  }
  return int( atof( str.c_str() ) );
}



//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bool ALIUtils::getBool( const ALIstring& str ) 
{
   bool val;
  
 //t str = upper( str );
  //----------- first check that it is a not number
  if( str == "ON" || str == "TRUE"  ) {
    val = true;
  } else if( str == "OFF" || str == "FALSE" ) {
    val = false;
  } else {
    std::cerr << "!!!! EXITING: trying to get the float from a string that is not 'ON'/'OFF'/'TRUE'/'FALSE' " << str << std::endl;
    exit(1);
  }

  return val;
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ALIstring ALIUtils::subQuotes( const ALIstring& str ) 
{
  
  //---------- Take out leading and trailing '"'
  if( str.find('"') != 0 || str.rfind('"') != str.length()-1 ) { 
    std::cerr << "!!!EXITING trying to substract quotes from a word that has no quotes " << str << std::endl;
    exit(1);
  }

  //  str = str.strip(ALIstring::both, '\"');
  //---------- Take out leading and trallling '"'
  ALIstring strt = str.substr(1,str.size()-2);

  //-  std::cout << " subquotes " << str << std::endl;
  //---------- Look for leading spaces
  while( strt[0] == ' ' ) {
   strt = strt.substr(1,strt.size()-1);
  }

  //---------- Look for trailing spaces
  while( strt[strt.size()-1] == ' ' ) {
   strt = strt.substr(0,strt.size()-1);
  }

  return strt;

}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void ALIUtils::dumpVS( const std::vector<ALIstring>& wl , const std::string& msg, std::ostream& outs ) 
{
  outs << msg << std::endl;
  uint siz = wl.size();
  for( uint ii=0; ii< siz; ii++ ){
    outs << wl[ii] << " ";
    /*  ostream_iterator<ALIstring> os(outs," ");
	copy(wl.begin(), wl.end(), os);*/
  }
  outs << std::endl;
}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ALIdouble ALIUtils::getDimensionValue( const ALIstring& dim, const ALIstring& dimType )
{
  ALIdouble value;
  if( dimType == "Length" ) {
    if( dim == "mm" ) {
      value = 1.E-3;
    }else if( dim == "cm" ) {
      value = 1.E-2;
    }else if( dim == "m" ) {
      value = 1.;
    }else if( dim == "mum" ) {
      value = 1.E-6;
    }else if( dim == "dm" ) {
      value = 1.E-1;
    }else if( dim == "nm" ) {
      value = 1.E-9;
    }else {
      std::cerr << "!!!!FATAL ERROR:  ALIUtils::GetDimensionValue. " << dim << " is a dimension not supported for dimension type " << dimType << std::endl;
      abort();
    }
  } else if( dimType == "Angle" ) {
    if( dim == "rad" ) {
      value = 1.;
    }else if( dim == "mrad" ) {
      value = 1.E-3;
    }else if( dim == "murad" ) {
      value = 1.E-6;
    }else if( dim == "deg" ) {
      value = M_PI/180.;
    }else if( dim == "grad" ) {
      value = M_PI/200.;
    }else {
      std::cerr << "!!!!FATAL ERROR:  ALIUtils::GetDimensionValue. " << dim << " is a dimension not supported for dimension type " << dimType << std::endl;
      abort();
    }
  }else {
      std::cerr << "!!!!FATAL ERROR:  ALIUtils::GetDimensionValue. " << dimType << " is a dimension type not supported " << std::endl;
      abort();
  }

  return value;

}


/*
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
std::ostream& operator << (std::ostream& os, const HepRotation& rm)
{

  return os << " xx=" << rm.xx() << " xy=" << rm.xy() << " xz=" << rm.xz() << std::endl
	    << " yx=" << rm.yx() << " yy=" << rm.yy() << " yz=" << rm.yz() << std::endl
	    << " zx=" << rm.zx() << " zy=" << rm.zy() << " zz=" << rm.zz() << std::endl;

}
*/

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
std::string ALIUtils::changeName( const std::string& oldName, const std::string& subsstr1,  const std::string& subsstr2 )
{

  std::string newName = oldName;
  int il = oldName.find( subsstr1, il );
  //  std::cout << " il " << il << " oldname " << oldName << " " << subsstr1 << std::endl;
  while( il >= 0 ) {
    newName = newName.substr( 0, il ) + subsstr2 +  newName.substr( il+subsstr1.length(), newName.length() );
    //    std::cout << " dnewName " << newName << " " << newName.substr( 0, il ) << " " << subsstr2 << " " << newName.substr( il+subsstr1.length(), newName.length() ) << std::endl;
    il = oldName.find( subsstr1, il+1 );  
  }

  return newName;
}
