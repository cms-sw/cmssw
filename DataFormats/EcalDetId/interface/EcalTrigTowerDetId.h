#ifndef ECALDETID_ECALTRIGTOWERDETID_H
#define ECALDETID_ECALTRIGTOWERDETID_H

#include <ostream>
#include <boost/cstdint.hpp>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


/** \class EcalTrigTowerDetId
    
   Cell id for an Calo Trigger tower

   $Id: EcalTrigTowerDetId.h,v 1.1 2005/07/25 15:30:14 meridian Exp $
*/
namespace cms
{

  class EcalTrigTowerDetId : public DetId {
  public:
    /** Constructor of a null id */
    EcalTrigTowerDetId();
    /** Constructor from a raw value */
    EcalTrigTowerDetId(uint32_t rawid);  
    /** \brief Constructor from signed ieta, iphi
     */
    EcalTrigTowerDetId(int ieta, int iphi);

    /** Constructor from a generic cell id */
    EcalTrigTowerDetId(const DetId& id);
    /** Assignment from a generic cell id */
    EcalTrigTowerDetId& operator=(const DetId& id);


    /// get the z-side of the tower (1/-1)
    int zside() const { return (id_&0x2000)?(1):(-1); }
    /// get the absolute value of the tower ieta
    int ietaAbs() const { return (id_>>7)&0x3f; }
    /// get the tower ieta
    int ieta() const { return zside()*ietaAbs(); }
    /// get the tower iphi
    int iphi() const { return id_&0x7F; }
    /// get a compact index for arrays [TODO: NEEDS WORK]
    int hashedIndex() const;

    static const int MIN_IETA = 1;
    static const int MIN_IPHI = 1;
    static const int MAX_IETA = 32;
    static const int MAX_IPHI = 72;
  };

  std::ostream& operator<<(std::ostream&,const EcalTrigTowerDetId& id);

}

#endif
