//
//   110208 SV TRACO hardware bug included
//
//
//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTTraco/interface/Lut.h"

//----------------
// Constructors --
//----------------

Lut::Lut(DTConfigLUTs* conf, int ntc, float SL_shift): _conf_luts(conf) {

  // set parameters from configuration
  m_d		= _conf_luts->D(); 
  m_ST		= _conf_luts->BTIC();
  m_wheel 	= _conf_luts->Wheel();

  // 110208 SV TRACO hardware bug included: Xcn must be corrected because Xcn parameter has been
  // inserted in hardware strings with the shift included, because TRACO doesn't apply shift in positive cases
  float Xcn_corr    = _conf_luts->Xcn();
  if(SL_shift > 0){
    if(Xcn_corr > 0)
      Xcn_corr = Xcn_corr - SL_shift;
    if(Xcn_corr < 0)
      Xcn_corr = Xcn_corr + SL_shift;
  }
	
  m_Xcn 	= Xcn_corr - (CELL_PITCH * 4.0 * (float)(ntc-1) * (float) m_wheel);
  m_pitch_d_ST	= CELL_PITCH / m_ST;


  //std::cout<< "Lut::Lut  ->  m_d " << m_d << " m_ST " << m_ST << " m_wheel " << m_wheel << " m_Xcn " << m_Xcn << " ntc " << ntc << std::endl;
  return;
}
 
Lut::~Lut() {}

void Lut::setForTestBeam( int station, int board, int traco ) {
 // set parameters from fixed values for MB1 nd MB3 (used in Testbeams 2003 2004) 
  int nStat = station;
  int nBoard = board;
  int nTraco = traco;
  float tracoPos[50];

  if( nStat ==1 ){

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

    m_d  = 431.175;
    m_ST = 31;
    float m_Xc = tracoPos[ ( nBoard * 10 ) + nTraco ];
    float m_Xn = +39.0;
    m_Xcn =  m_Xn - m_Xc; 
    //m_stsize = m_CELL_PITCH / m_ST;
    //m_distp2 = 0.5 + ( 2.0 * m_CELL_H * m_ST / m_CELL_PITCH );
  }

  if( nStat ==3){
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

    m_d  = 512.47;
    m_ST = 31;
    float m_Xc = tracoPos[ ( nBoard * 10 ) + nTraco ];
    float m_Xn = -21.0;
    m_Xcn =  m_Xn - m_Xc; 
    //m_stsize = m_CELL_PITCH / m_ST;
    //m_distp2 = 0.5 + ( 2.0 * m_CELL_H * m_ST / m_CELL_PITCH );
  
  }

  return;
}


int Lut::get_k( int addr ) {
//FIX attenzione controlla addr - 511 o -512???
  int i;
  float x;
  i = addr - 512;
  x = (float)i * CELL_PITCH / ( SL_D * m_ST );
  x = atanf(x);
  x = x * ANGRESOL;
  if(m_wheel<0)
    x = -x;

  return (int)x;
}

int Lut::get_x( int addr ) {
  int i;
  float a,b,x;

  if(addr<=511) //LUT outer
  {
   	i=addr;
	b=m_d+SL_DIFF;
  }
  else if(addr<=1023) //LUT inner
  {
	i=addr-512;
	b=m_d-SL_DIFF;		
  }
  else	//LUT correlati
  {
 	i=addr-1024;
 	b=m_d;
  }
  a = m_Xcn - (m_pitch_d_ST * (float)i * (float)m_wheel);
  x = a/b;

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



void DSPtoIEEE32(short DSPmantissa, short DSPexp, float *f)
{
  DSPexp -= 15;
  *f = DSPmantissa * (float)pow( 2.0, DSPexp );
  
  return;
}


void IEEE32toDSP(float f, short *DSPmantissa, short *DSPexp)
{
  //long *pl, lm;
  uint32_t pl;
  uint32_t lm;

  //101104 SV convert float to int in safe way
  union { float f; uint32_t i; } u;
  u.f = f;
  pl = u.i;

  bool sign=false;
  if( f==0.0 )
  {
    *DSPexp = 0;
    *DSPmantissa = 0;
  }
  else
  {
    //pl = reinterpret_cast<uint32_t*> (&f);
    //pl = (long*) (&f);
    if((pl & 0x80000000)!=0)
      sign=true;
    lm =( 0x800000 | (pl & 0x7FFFFF)); // [1][23bit mantissa]
    lm >>= 9; //reduce to 15bits
    lm &= 0x7FFF;
    *DSPexp = ((pl>>23)&0xFF)-126;
    *DSPmantissa = (short)lm;
    if(sign)
      *DSPmantissa = - *DSPmantissa; // convert negative value in 2.s
    // complement
  }
  return;
}


