#ifndef mytestdata_H
#define mytestdata_H

#include <iostream>
#include <string>

class MyTestData {
public:

  MyTestData():
    a( 0 ),
    b( 0. ),
    s(""){
    for( size_t i=0;i<2;i++)
      for( size_t j=0;j<2;j++){
	d[i][j]=0;
	f[i][j]=0;
      }
  }
  MyTestData( int seed ):
    a( seed ),
    b( seed + 1.1 ),
    s( "Bla bla" ){
    for( size_t i=0;i<2;i++)
      for( size_t j=0;j<2;j++){
	d[i][j]=0;
	f[i][j]=0;
      }
    d[0][0]=1;
    d[0][1]=2;
    d[1][0]=3;
    d[1][1]=4;
    f[0][0]=5;
    f[0][1]=6;
    f[1][0]=7;
    f[1][1]=8;
  }
  void print(){
    std::cout <<"MyTestData: a="<<a<<" b="<<b<<" s="<<s<<std::endl;
    for( size_t i=0;i<2;i++)
      for( size_t j=0;j<2;j++){
	std::cout <<"d["<<i<<"]["<<j<<"]="<<d[i][j]<<std::endl;
      }
    for( size_t i=0;i<2;i++)
      for( size_t j=0;j<2;j++){
	std::cout <<"f["<<i<<"]["<<j<<"]="<<f[i][j]<<std::endl;
      }
  }

  bool operator==( const MyTestData& rhs ) const {
    if( a != rhs.a ) return false;
    if( b != rhs.b ) return false;
    for( size_t i=0;i<2;i++)
      for( size_t j=0;j<2;j++){
	if(d[i][j]!=rhs.d[i][j]) return false;
	if(f[i][j]!=rhs.f[i][j]) return false;
      }
    if( s != rhs.s ) return false;
    return true;
  }
  bool operator!=( const MyTestData& rhs ) const {
    return !operator==( rhs );
  }
private:
  int a;
  float b;
  std::string s;
  double d[2][2];
  int f[2][2];
};
#endif
