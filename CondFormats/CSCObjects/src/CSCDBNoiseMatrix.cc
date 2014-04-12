#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include <iostream>

std::ostream & operator<<(std::ostream & os, const CSCDBNoiseMatrix & cscdb)
{
  for ( size_t i = 0; i < cscdb.matrix.size(); ++i )
  {
    os <<  "elem: " << i << " noise matrix: "   <<
    cscdb.matrix[i].elem33 << " " <<
    cscdb.matrix[i].elem34 << " " <<
    cscdb.matrix[i].elem35 << " " <<
    cscdb.matrix[i].elem44 << " " <<
    cscdb.matrix[i].elem45 << " " <<
    cscdb.matrix[i].elem46 << " " <<
    cscdb.matrix[i].elem55 << " " <<
    cscdb.matrix[i].elem56 << " " <<
    cscdb.matrix[i].elem57 << " " <<
    cscdb.matrix[i].elem66 << " " <<
    cscdb.matrix[i].elem67 << " " <<
    cscdb.matrix[i].elem77 << " " << "\n";
  }
  return os;
}
