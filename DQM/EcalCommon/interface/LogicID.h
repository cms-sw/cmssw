// $Id: LogicID.h,v 1.8 2008/03/14 14:38:56 dellaric Exp $

/*!
  \file LogicID.h
  \brief Cache logicID vector from database
  \author B. Gobbo 
  \version $Revision: 1.8 $
  \date $Date: 2008/03/14 14:38:56 $
*/

#ifndef LogicID_H
#define LogicID_H

#include <vector>
#include <string>
#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class LogicID {

public:

static EcalLogicID getEcalLogicID( const char* name,
                                   const int id1 = EcalLogicID::NULLID,
                                   const int id2 = EcalLogicID::NULLID,
                                   const int id3 = EcalLogicID::NULLID ) throw( std::runtime_error );

};

#endif // LogicID_H
