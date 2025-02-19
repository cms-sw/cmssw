#ifndef binary_ofstream_H
#define binary_ofstream_H

#include <string>
#include <cstdio>

#include "FWCore/Utilities/interface/Visibility.h"
class binary_ofstream {
public:

    explicit binary_ofstream( const char* name);
    explicit binary_ofstream( const std::string& name);

    ~binary_ofstream();

    binary_ofstream& operator<<( char n);
    binary_ofstream& operator<<( unsigned char n);

    binary_ofstream& operator<<( short n);
    binary_ofstream& operator<<( unsigned short n);

    binary_ofstream& operator<<( int n);
    binary_ofstream& operator<<( unsigned int n);

    binary_ofstream& operator<<( long n);
    binary_ofstream& operator<<( unsigned long n);

    binary_ofstream& operator<<( float n);
    binary_ofstream& operator<<( double n);

    binary_ofstream& operator<<( bool n);
    binary_ofstream& operator<<( const std::string& n);

    void close();

private:

    FILE* file_;

    void init( const char* name);

};

#endif
