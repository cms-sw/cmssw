#include "binary_ofstream.h"

#include <cstdio>
#include <iostream>

struct binary_ofstream_error {};

binary_ofstream::binary_ofstream( const char* name) : file_(0)
{
    init (name);
}

binary_ofstream::binary_ofstream( const std::string& name) : file_(0)
{
    init (name.c_str());
}

void binary_ofstream::init( const char* name)
{
    file_ = fopen( name, "wb");
    if (file_ == 0) {
	std::cout << "file " << name << " cannot be opened for writing"
		  << std::endl;
	throw binary_ofstream_error();
    }
}

binary_ofstream::~binary_ofstream()
{
    close();
}
void binary_ofstream::close()
{
    if (file_ != 0) fclose( file_);
    file_ = 0;
}

binary_ofstream& binary_ofstream::operator<<( char n) {
    fputc( n, file_); return *this;
}
binary_ofstream& binary_ofstream::operator<<( unsigned char n) {
    fputc( n, file_); return *this;
}

binary_ofstream& binary_ofstream::operator<<( short n) {
    fwrite( &n, sizeof(n), 1, file_); return *this;}
binary_ofstream& binary_ofstream::operator<<( unsigned short n) {
    fwrite( &n, sizeof(n), 1, file_); return *this;}
binary_ofstream& binary_ofstream::operator<<( int n) {
    fwrite( &n, sizeof(n), 1, file_); return *this;}
binary_ofstream& binary_ofstream::operator<<( unsigned int n) {
    fwrite( &n, sizeof(n), 1, file_); return *this;}
binary_ofstream& binary_ofstream::operator<<( long n) {
    fwrite( &n, sizeof(n), 1, file_); return *this;}
binary_ofstream& binary_ofstream::operator<<( unsigned long n) {
    fwrite( &n, sizeof(n), 1, file_); return *this;}
binary_ofstream& binary_ofstream::operator<<( float n) {
    fwrite( &n, sizeof(n), 1, file_); return *this;}
binary_ofstream& binary_ofstream::operator<<( double n) {
    fwrite( &n, sizeof(n), 1, file_); return *this;}

binary_ofstream& binary_ofstream::operator<<( bool n) {
  return operator<<( static_cast<char>(n));
}

binary_ofstream& binary_ofstream::operator<<( const std::string& s) {
  (*this) << (uint32_t) s.size(); // Use uint32 for backward compatibilty of binary files that were generated on 32-bit machines.
  fwrite( s.data(), 1, s.size(), file_);
  return *this;
}
