// $Id: $

/*!
  \file LogicID.h
  \brief Cache logicID vector from database
  \author B. Gobbo 
  \version $Revision: $
  \date $Date: $
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


  static EcalLogicID getEcalLogicID( std::string name, int id1=0, int id2=0 ) throw( std::runtime_error );

 private:

  static bool                                              init_;
  static std::map< std::string, std::vector<EcalLogicID> > IDmap_;

};

#endif // LogicID_H
