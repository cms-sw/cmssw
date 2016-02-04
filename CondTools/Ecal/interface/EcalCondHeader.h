#ifndef _CondTools_Ecal_EcalCondHeader_h_
#define _CondTools_Ecal_EcalCondHeader_h_


/** Header to be associated with a set of ecal conditions
 *  stored in XML format
 *
 *  \author S. Argiro
 *  \todo Strong types instead of strings
 *
 * $Id: EcalCondHeader.h,v 1.1 2008/11/14 15:46:05 argiro Exp $
 */   

#include "CondCore/DBCommon/interface/Time.h"
#include <string>

struct EcalCondHeader {

  
  std::string  method_;
  std::string  version_;
  std::string  datasource_;
  cond::Time_t since_;
  std::string  tag_;
  std::string  date_;

  EcalCondHeader():method_(""),
		 version_(""),
		 datasource_(""),
		 since_(0),
		 tag_(""),
		 date_(""){}


  void reset(){method_=version_=datasource_=tag_=date_="";since_=0;}
  
};

#endif
