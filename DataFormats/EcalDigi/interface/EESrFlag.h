// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-
//$Id: EESrFlag.h,v 1.2 2007/03/27 09:55:00 meridian Exp $

#ifndef EESRFLAG_H
#define EESRFLAG_H

#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDigi/interface/EcalSrFlag.h"
#include "FWCore/Utilities/interface/Exception.h"

/** This class holds a Selective Readout Flag (SRF) associated to an
 * ECAL endcap supercrystal.
 */
class EESrFlag: public EcalSrFlag {
public:

public:
  typedef EcalScDetId key_type; //key for edm::SortedCollection

public:
  /** Default constructor.
   */
  EESrFlag() {}; 
  
  /** Constructor
   * @param sc supercrystal det id
   * @param flag the srp flag, an integer in [0,7]. See constants SRF_xxx in EcalSrFlags class.
   */
  EESrFlag(const EcalScDetId& sc, const int& flag): scId_(sc){
    //SRP flag is coded on 3 bits:
    if(flag<0  || flag>0x7) throw cms::Exception("InvalidValue", "srp flag greater than 0x7 or negative.");
    flag_ = (unsigned char) flag;
  }
  
  /** For edm::SortedCollection.
   * @return det id of the trigger tower the flag is assigned to.
   */
  const EcalScDetId& id() const { return scId_;}
  
private:
  /** trigger tower id
   */
  EcalScDetId scId_;  
};


std::ostream& operator<<(std::ostream& s, const EESrFlag& digi);

#endif //EESRFLAG_H not defined
