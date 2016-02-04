// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-
//$Id: EBSrFlag.h,v 1.2 2007/03/27 09:55:00 meridian Exp $

#ifndef EBSRFLAG_H
#define EBSRFLAG_H

#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDigi/interface/EcalSrFlag.h"
#include "FWCore/Utilities/interface/Exception.h"


/** This class holds a Selective Readout Flag (SRF) associated to an
 * ECAL barrel trigger tower.
 */
class EBSrFlag: public EcalSrFlag {
public:
  typedef EcalTrigTowerDetId key_type; //key for edm::SortedCollection

public:
  /** Default constructor.
   */
  EBSrFlag() {}; 
  
  /** Constructor
   * @param tt trigger tower det id.
   * @param flag the srp flag, an integer in [0,7]. See constants SRF_xxx in EcalSrFlags class.
   */
  EBSrFlag(const EcalTrigTowerDetId& tt, const int& flag): ttId_(tt){
    //SRP flag is coded on 3 bits:
    if(flag<0  || flag>0x7) throw cms::Exception("InvalidValue", "srp flag greater than 0x7 or negative.");
    flag_ = (unsigned char) flag;
  }
  
  /** For edm::SortedCollection.
   * @return det id of the trigger tower the flag is assigned to.
   */
  const EcalTrigTowerDetId& id() const { return ttId_;}
  
private:
  /** trigger tower id
   */
  EcalTrigTowerDetId ttId_;  
};


std::ostream& operator<<(std::ostream& s, const EBSrFlag& digi);

#endif //EBSRFLAG_H not defined
