

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTTraco/interface/Lut.h"

//----------------
// Constructors --
//----------------


Lut::Lut( int station, int board, int traco ): nStat( station ), nBoard( board ), nTraco( traco ) {

 switch( nStat ){
  case 1:
  tracoPos = new float[50];
  tracoPos[ 0] = -120.19;
  tracoPos[ 1] = -103.39;
  tracoPos[ 2] =  -86.59;
  tracoPos[ 3] =  -69.80;
  tracoPos[10] =  -52.99;
  tracoPos[11] =  -36.19;
  tracoPos[12] =  -19.39;
  tracoPos[13] =   -2.59;
  tracoPos[20] =   14.20;
  tracoPos[21] =   31.00;
  tracoPos[22] =   47.80;
  tracoPos[23] =   64.60;
  tracoPos[30] =   81.40;

  SL_DIFF = 11.8;
  CELL_H     = 1.3;
  CELL_PITCH = 4.2;
  ANGRESOL = 512;
  POSRESOL = 4096;

  m_d  = 431.175;
  m_ST = 31.0;
  m_Xc = tracoPos[ ( nBoard * 10 ) + nTraco ];
  m_Xn = +39.0;
  m_shift = 18.0;
  m_stsize = CELL_PITCH / m_ST;
  m_distp2 = 0.5 + ( 2.0 * CELL_H * m_ST / CELL_PITCH );
  
  break;

  case 3:
  tracoPos = new float[50];
  tracoPos[ 0] = -165.45;
  tracoPos[ 1] = -148.65;
  tracoPos[ 2] = -131.85;
  tracoPos[ 3] = -115.05;
  tracoPos[10] =  -98.25;
  tracoPos[11] =  -81.45;
  tracoPos[12] =  -64.65;
  tracoPos[13] =  -47.85;
  tracoPos[20] =  -31.05;
  tracoPos[21] =  -14.25;
  tracoPos[22] =    2.54;
  tracoPos[23] =   19.34;
  tracoPos[30] =   36.14;
  tracoPos[31] =   52.94;
  tracoPos[32] =   69.74;
  tracoPos[33] =   86.54;
  tracoPos[40] =  103.34;
  tracoPos[41] =  120.14;
  tracoPos[42] =  136.94;
  tracoPos[43] =  153.74;

  SL_DIFF = 11.8;
  CELL_H     = 1.3;
  CELL_PITCH = 4.2;
  ANGRESOL = 512;
  POSRESOL = 4096;

  m_d  = 512.47;
  m_ST = 31.0;
  m_Xc = tracoPos[ ( nBoard * 10 ) + nTraco ];
  m_Xn = -21.0;
  m_shift = 18.0;
  m_stsize = CELL_PITCH / m_ST;
  m_distp2 = 0.5 + ( 2.0 * CELL_H * m_ST / CELL_PITCH );

  break;

 }//end switch
}

Lut::~Lut() {
}

int Lut::get_k( int addr ) {
  int i;
  float x;
  i = addr - 511;
  x = 2.0 * i / ( m_shift * m_distp2 );
  x = atanf(x);
  x = x * ANGRESOL;
  return (int)x;
}

int Lut::get_x( int addr ) {
  int i;
  float d,x,Xl;
  if ( addr <= 511 ) {
    i = addr;
    d = m_d + SL_DIFF;
  }
  else
  if ( addr <= 1023 ) {
    i = addr - 512;
    d = m_d - SL_DIFF;
  }
  else
  {
    i = addr - 1024;
    d = m_d;
  }

  Xl = m_Xc + m_stsize * i;
  x = ( m_Xn - Xl ) / d;
  x = atanf(x);
  x = x * POSRESOL;
  return (int)x;
}

char exaDigit( int i ) {
  if ( i < 10 ) return (   i        + '0' );
  else          return ( ( i - 10 ) + 'A' );
}

std::string lutFmt( int i ) {
  char* buf = new char[6];
  buf[2] = ' ';
  buf[5] = '\0';
  int j4 = i % 16;
  i /= 16;
  int j3 = i % 16;
  i /= 16;
  int j2 = i % 16;
  i /= 16;
  int j1 = i % 16;
  buf[0] = exaDigit( j1 );
  buf[1] = exaDigit( j2 );
  buf[3] = exaDigit( j3 );
  buf[4] = exaDigit( j4 );
  std::string s( buf );
  return s;
}


/*  this for dumping luts in minicrate input format - for MB1  -- Testbeam 2004

int main( int argn, char** argv ) {

//  while ( 1 ) {
//    int k;
//    cin >> k;
//    cout << lutFmt( k ) << endl;
//  }

//  cout << argn << endl;
  if ( argn != 3 ) return 1;
//  cout << *argv[1] << endl;
//  cout << *argv[2] << endl;
  if ( *argv[1] < '0' ) return 2;
  if ( *argv[1] > '9' ) return 2;
  if ( *argv[2] < '0' ) return 3;
  if ( *argv[2] > '9' ) return 3;
  int board = *argv[1] - '0';
  int traco = *argv[2] - '0';
  Lut lut( board, traco );
  int i;
  for ( i = 1; i <= 1536; i++ ) {
    cout << i << " " << lut.get( i ) << endl;
  }

  char* stri = "l1_  ";
  char* name = new char[10];
  char* s = stri;
  char* d = name;
  while ( *d++ = *s++ );
  int board;
  int traco;
  char winNewLine;
  winNewLine = 13;
  ofstream full( "l1_full" );
  for ( board = 0; board < 4; board++ ) {
    for ( traco = 0; traco < 4; traco++ ) {
      if ( ( board == 3 ) && ( traco != 0 ) ) continue;
      name[3] = '0' + board;
      name[4] = '0' + traco;
      cout << board << " " << traco << " " << name << endl;
      ofstream file( name );
      Lut lut( board, traco );
      cout << "loop" << endl;
      int i;
      int nfirst;
      int nwrite;
      nfirst = 0;
      nwrite = 0;
      for ( i = 0; i <= 1535; i++ ) {
//        if ( i < 512 )
//        if ( ( i > 512 ) && ( i < 1024 ) )
        int y = lut.get_x( i );
        int z = y;
        if ( z < 0 ) z += 65536;
        cout << board << " " << traco << " "
             << i << " " << y << endl;
        if ( nwrite == 0 ) {
          file << "4D " << board << " " << traco << " 0 "
               << lutFmt( nfirst );
          full << "4D " << board << " " << traco << " 0 "
               << lutFmt( nfirst );
//               << nfirst << " ";
	}
//        file   << lut.get( i ) << " ";
        file   << " " << lutFmt( z );
        full   << " " << lutFmt( z );
        nwrite++;
        if ( nwrite == 64 ) {
          file << winNewLine << endl;
          full << winNewLine << endl;
          nfirst += nwrite;
          nwrite = 0;
        }
      }
      nfirst = 0;
      nwrite = 0;
      for ( i = 0; i <= 1023; i++ ) {
//        if ( i < 512 )
//        if ( ( i > 512 ) && ( i < 1024 ) )
        int y = lut.get_k( i );
        int z = y;
        if ( z < 0 ) z += 65536;
        cout << board << " " << traco << " "
             << i << " " << y << endl;
        if ( nwrite == 0 ) {
          file << "4D " << board << " " << traco << " 1 "
               << lutFmt( nfirst );
          full << "4D " << board << " " << traco << " 1 "
               << lutFmt( nfirst );
//               << nfirst << " ";
	}
//        file   << lut.get( i ) << " ";
        file   << " " << lutFmt( z );
        full   << " " << lutFmt( z );
        nwrite++;
        if ( nwrite == 64 ) {
          file << winNewLine << endl;
          full << winNewLine << endl;
          nfirst += nwrite;
          nwrite = 0;
        }
      }
      file << "4E " << board << " " << traco << winNewLine << endl;
      full << "4E " << board << " " << traco << winNewLine << endl;
    }
  }

  return 0;

}


*** and for MB3  -- Testbeam 2004

int main( int argn, char** argv ) {

//  while ( 1 ) {
//    int k;
//    cin >> k;
//    cout << lutFmt( k ) << endl;
//  }

//  cout << argn << endl;
  if ( argn != 3 ) return 1;
//  cout << *argv[1] << endl;
//  cout << *argv[2] << endl;
  if ( *argv[1] < '0' ) return 2;
  if ( *argv[1] > '9' ) return 2;
  if ( *argv[2] < '0' ) return 3;
  if ( *argv[2] > '9' ) return 3;
  int board = *argv[1] - '0';
  int traco = *argv[2] - '0';
  Lut lut( board, traco );
  int i;
  for ( i = 1; i <= 1536; i++ ) {
    cout << i << " " << lut.get( i ) << endl;
  }

  char* stri = "l3_  ";
  char* name = new char[10];
  char* s = stri;
  char* d = name;
  while ( *d++ = *s++ );
  int board;
  int traco;
  char winNewLine;
  winNewLine = 13;
  ofstream full( "l3_full" );
  for ( board = 0; board < 5; board++ ) {
    for ( traco = 0; traco < 4; traco++ ) {
      name[3] = '0' + board;
      name[4] = '0' + traco;
      cout << board << " " << traco << " " << name << endl;
      ofstream file( name );
      Lut lut( board, traco );
      cout << "loop" << endl;
      int i;
      int nfirst;
      int nwrite;
      nfirst = 0;
      nwrite = 0;
      for ( i = 0; i <= 1535; i++ ) {
//        if ( i < 512 )
//        if ( ( i > 512 ) && ( i < 1024 ) )
        int y = lut.get_x( i );
        int z = y;
        if ( z < 0 ) z += 65536;
        cout << board << " " << traco << " "
             << i << " " << y << endl;
        if ( nwrite == 0 ) {
          file << "4D " << board << " " << traco << " 0 "
               << lutFmt( nfirst );
          full << "4D " << board << " " << traco << " 0 "
               << lutFmt( nfirst );
//               << nfirst << " ";
        }
//        file   << lut.get( i ) << " ";
        file   << " " << lutFmt( z );
        full   << " " << lutFmt( z );
        nwrite++;
        if ( nwrite == 64 ) {
          file << winNewLine << endl;
          full << winNewLine << endl;
          nfirst += nwrite;
          nwrite = 0;
        }
      }
      nfirst = 0;
      nwrite = 0;
      for ( i = 0; i <= 1023; i++ ) {
//        if ( i < 512 )
//        if ( ( i > 512 ) && ( i < 1024 ) )
        int y = lut.get_k( i );
        int z = y;
        if ( z < 0 ) z += 65536;
        cout << board << " " << traco << " "
             << i << " " << y << endl;
        if ( nwrite == 0 ) {
          file << "4D " << board << " " << traco << " 1 "
               << lutFmt( nfirst );
          full << "4D " << board << " " << traco << " 1 "
               << lutFmt( nfirst );
//               << nfirst << " ";
        }
//        file   << lut.get( i ) << " ";
        file   << " " << lutFmt( z );
        full   << " " << lutFmt( z );
        nwrite++;
        if ( nwrite == 64 ) {
          file << winNewLine << endl;
          full << winNewLine << endl;
          nfirst += nwrite;
          nwrite = 0;
        }
      }
      file << "4E " << board << " " << traco << winNewLine << endl;
      full << "4E " << board << " " << traco << winNewLine << endl;
    }
  }

  return 0;

}

*/


