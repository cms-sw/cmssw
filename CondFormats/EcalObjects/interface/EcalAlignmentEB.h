#ifndef CondFormats_EcalObjects_EcalAlignmentEB_H
#define CondFormats_EcalObjects_EcalAlignmentEB_H
/**
 * Author: Paolo Meridiani
 * Created: 14 Nov 2006
 * $Id: EcalAlignment.h,v 1.1 2010/06/16 10:48:37 fay Exp $
 **/


#include <iostream>
#include <boost/cstdint.hpp>
#include "CondFormats/Alignment/interface/AlignTransform.h"

class EcalAlignmentEB {
//struct EcalAlignment_EB {
  public:
  std::vector<AlignTransform> m_align;
  EcalAlignmentEB(){}
  EcalAlignmentEB(std::vector<AlignTransform> x_align){
    std::cout << " vector size " << x_align.size();
  for (std::vector<AlignTransform>::const_iterator i = x_align.begin(); i != x_align.end(); ++i){	
    m_align.push_back(*i);
  }
 }
  ~EcalAlignmentEB(){}
};

#endif
