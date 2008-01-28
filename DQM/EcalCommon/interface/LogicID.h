// $Id: LogicID.h,v 1.5 2008/01/16 13:32:51 dellaric Exp $

/*!
  \file LogicID.h
  \brief Cache logicID vector from database
  \author B. Gobbo 
  \version $Revision: 1.5 $
  \date $Date: 2008/01/16 13:32:51 $
*/

#ifndef LogicID_H
#define LogicID_H

#include <vector>
#include <string>
#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

class EcalCondDBInterface;

class LogicID {

 public:

  static void init( EcalCondDBInterface* eConn, EcalSubdetector subdet );


  static EcalLogicID getEcalLogicID( std::string name, int id1=0, int id2=0, int id3=0 ) throw( std::runtime_error );

 private:

  static bool                                              init_;
  static std::map< std::string, std::vector<EcalLogicID> > IDmap_;

};

#endif // LogicID_H
