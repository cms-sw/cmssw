// $Id: LogicID.h,v 1.3 2007/12/18 13:13:49 dellaric Exp $

/*!
  \file LogicID.h
  \brief Cache logicID vector from database
  \author B. Gobbo 
  \version $Revision: 1.3 $
  \date $Date: 2007/12/18 13:13:49 $
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

  static void init( EcalCondDBInterface* eConn );


  static EcalLogicID getEcalLogicID( std::string name, int id1=0, int id2=0 ) throw( std::runtime_error );

 private:

  static bool                                              init_;
  static std::map< std::string, std::vector<EcalLogicID> > IDmap_;

};

#endif // LogicID_H
