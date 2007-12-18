// $Id: LogicID.h,v 1.2 2007/11/13 14:05:33 dellaric Exp $

/*!
  \file LogicID.h
  \brief Cache logicID vector from database
  \author B. Gobbo 
  \version $Revision: 1.2 $
  \date $Date: 2007/11/13 14:05:33 $
*/

#ifndef LogicID_H
#define LogicID_H

#include <vector>
#include <string>
#include <map>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class EcalCondDBInterface;

class LogicID {

 public:

  static void init( EcalCondDBInterface* eConn ) throw( std::runtime_error );


  static EcalLogicID getEcalLogicID( std::string name, int id1=0, int id2=0 );

 private:

  static bool                                              init_;
  static std::map< std::string, std::vector<EcalLogicID> > IDmap_;

};

#endif // LogicID_H
