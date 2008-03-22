// $Id: LogicID.h,v 1.6 2008/01/28 15:41:14 dellaric Exp $

/*!
  \file LogicID.h
  \brief Cache logicID vector from database
  \author B. Gobbo 
  \version $Revision: 1.6 $
  \date $Date: 2008/01/28 15:41:14 $
*/

#ifndef LogicID_H
#define LogicID_H

#include <vector>
#include <string>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LogicID {

public:

static EcalLogicID getEcalLogicID( std::string name,
                                   int id1 = EcalLogicID::NULLID,
                                   int id2 = EcalLogicID::NULLID,
                                   int id3 = EcalLogicID::NULLID ) throw( std::runtime_error );

};

#endif // LogicID_H
