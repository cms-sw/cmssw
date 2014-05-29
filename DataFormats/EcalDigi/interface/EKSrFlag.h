// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-
//$Id: EKSrFlag.h,v 1.2 2014/04/02 09:55:00 shervin Exp $

#ifndef EKSRFLAG_H
#define EKSRFLAG_H

#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDigi/interface/EcalSrFlag.h"
#include "FWCore/Utilities/interface/Exception.h"

/** This class holds a Selective Readout Flag (SRF) associated to an
 * ECAL Shashlik supercrystal.
 */
class EKSrFlag: public EcalSrFlag {
public:

public:
  typedef EcalScDetId key_type; //key for edm::SortedCollection

public:
  /** Default constructor.
   */
  EKSrFlag() {}; 
  
  /** Constructor
   * @param sc supercrystal det id
   * @param flag the srp flag, an integer in [0,7]. See constants SRF_xxx in EcalSrFlags class.
   */
  EKSrFlag(const EcalScDetId& sc, const int& flag): scId_(sc){
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


std::ostream& operator<<(std::ostream& s, const EKSrFlag& digi);

#endif //EKSRFLAG_H not defined
