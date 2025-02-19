
#ifndef __LASMODULEPROFILE_C
#define __LASMODULEPROFILE_C


#include "Alignment/LaserAlignment/interface/LASModuleProfile.h"


LASModuleProfile::LASModuleProfile() {
  ///
  /// def constructor
  ///

  Init();

}




LASModuleProfile::LASModuleProfile( double theData[512] ) {
  ///
  /// construct and fill the data storage
  ///

  Init();
  for( unsigned int i = 0; i < 512; ++i ) data[i] = theData[i];

}





LASModuleProfile::LASModuleProfile( int theData[512] ) {
  ///
  /// construct and fill the data storage
  ///

  Init();
  for( unsigned int i = 0; i < 512; ++i ) data[i] = theData[i];

}





void LASModuleProfile::SetData( double theData[512] ) {
  ///
  /// fill the data storage
  ///

  for( unsigned int i = 0; i < 512; ++i ) data[i] = theData[i];

}





void LASModuleProfile::SetData( int theData[512] ) {
  ///
  /// fill the data storage
  /// 
  /// temporary workaround
  /// as long as data is provided in int arrays
  ///

  for( unsigned int i = 0; i < 512; ++i ) data[i] = theData[i];

}




// MADE INLINE
// double LASModuleProfile::GetValue( unsigned int theStripNumber ) const {
//   ///
//   /// return an element of the data
//   ///

//   return( data[theStripNumber] );

// }



// MADE INLINE
// void LASModuleProfile::SetValue( unsigned int theStripNumber, const double& theValue ) {
//   ///
//   ///
//   ///

//   data.at( theStripNumber ) = theValue;

// }





void LASModuleProfile::SetAllValuesTo( const double& theValue ) {
  ///
  ///
  ///

  for( unsigned int i = 0; i < data.size(); ++i ) data.at( i ) = theValue;

}





void LASModuleProfile::DumpToArray( double array[512] ) {
  ///
  /// fill array
  ///

  for( unsigned int i = 0; i < 512; ++i ) {
    array[i] = data.at( i );
  }

}





void LASModuleProfile::Init( void ) {
  ///
  /// everything needed for initialization
  ///

  data.resize( 512 );

}





LASModuleProfile& LASModuleProfile::operator=( const LASModuleProfile& anotherProfile ) {
  ///
  ///
  ///

  // check for self-assignment
  if( this != &anotherProfile ) {

    for( unsigned int i = 0; i < 512; ++i ) {
      data.at( i ) = anotherProfile.GetValue( i );
    }

  }
  
  return *this;
  
}





LASModuleProfile LASModuleProfile::operator+( const LASModuleProfile& anotherProfile ) {
  ///
  ///
  ///

  double theArray[512];
  for( unsigned int i = 0; i < 512; ++i ) {
    theArray[i] = this->GetValue( i ) + anotherProfile.GetValue( i );
  }
  
  return( LASModuleProfile( theArray ) );
  
}





LASModuleProfile LASModuleProfile::operator-( const LASModuleProfile& anotherProfile ) {
  ///
  ///
  ///

  double theArray[512];
  for( unsigned int i = 0; i < 512; ++i ) {
    theArray[i] = this->GetValue( i ) - anotherProfile.GetValue( i );
  }
  
  return( LASModuleProfile( theArray ) );

}





LASModuleProfile LASModuleProfile::operator+( const double b[512] ) {
  ///
  /// add a double[512]
  ///

  double theArray[512];
  for( unsigned int i = 0; i < 512; ++i ) {
    theArray[i] = this->GetValue( i ) + b[i];
  }
  
  return( LASModuleProfile( theArray ) );

}





LASModuleProfile LASModuleProfile::operator-( const double b[512] ) {
  ///
  /// subtract a double[512]
  ///

  double theArray[512];
  for( unsigned int i = 0; i < 512; ++i ) {
    theArray[i] = this->GetValue( i ) - b[i];
  }
  
  return( LASModuleProfile( theArray ) );

}





LASModuleProfile& LASModuleProfile::operator+=( const LASModuleProfile& anotherProfile ) {
  ///
  ///
  ///

  for( unsigned int i = 0; i < 512; ++i ) {
    data.at( i ) += anotherProfile.GetValue( i );
  }
  
  return *this;
  
}





LASModuleProfile& LASModuleProfile::operator-=( const LASModuleProfile& anotherProfile ) {
  ///
  ///
  ///

  for( unsigned int i = 0; i < 512; ++i ) {
    data.at( i ) -= anotherProfile.GetValue( i );
  }
  
  return *this;

}





LASModuleProfile& LASModuleProfile::operator+=( const double b[512] ) {
  ///
  ///
  ///

  for( unsigned int i = 0; i < 512; ++i ) {
    data.at( i ) += b[i];
  }
  
  return *this;
  
}





LASModuleProfile& LASModuleProfile::operator-=( const double b[512] ) {
  ///
  ///
  ///

  for( unsigned int i = 0; i < 512; ++i ) {
    data.at( i ) -= b[i];
  }
  
  return *this;

}




LASModuleProfile& LASModuleProfile::operator+=( const int b[512] ) {
  ///
  /// temporary workaround
  /// as long as data is provided in int arrays
  ///

  for( unsigned int i = 0; i < 512; ++i ) {
    data.at( i ) += b[i];
  }
  
  return *this;
  
}





LASModuleProfile& LASModuleProfile::operator-=( const int b[512] ) {
  ///
  /// temporary workaround
  /// as long as data is provided in int arrays
  ///

  for( unsigned int i = 0; i < 512; ++i ) {
    data.at( i ) -= b[i];
  }
  
  return *this;

}





LASModuleProfile& LASModuleProfile::operator/=( const double divisor ) {
  ///
  /// handle with care!!
  ///
  
  for( unsigned int i = 0; i < 512; ++i ) {
    data.at( i ) /= divisor;
  }
  
  return *this;

}




#endif
