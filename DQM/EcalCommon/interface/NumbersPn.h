#ifndef NumbersPn_H
#define NumbersPn_H

#include <string>
#include <stdexcept>
#include <vector>

#include "FWCore/Framework/interface/EventSetup.h"

class NumbersPn {

 public:

  static std::vector<int> getPNs( const int ism, const int ix, const int iy );

  static int ipnEE( const int ism, const int ipnid );

 private:

  static int iLM( const int ism, const int ix, const int iy ) throw( std::runtime_error );

  static std::vector<int> getPNs( const int ilm ) throw( std::runtime_error );

};

#endif // NumbersPn_H
