// ----------------------------------------------------------------------
//
// MessageLoggerDefaults.cc
//
// All the mechanics of hardwired defaults, but without the values, which are
// coded in HardwiredDefaults.cc
//
// Changes:
//
// 11/02/07 mf	Corrected sev_limit, sev_reportEvery, and sev_timespan
//  		changing if (c != def_destin.category.end()) to
//		if (c != def_destin.sev.end()) if 4 places in each.
//		This fixes the skipped framework job report message 
//		problem.  The bug also could have been causing other 
//		messages to be skipped.  
//
// ----------------------------------------------------------------------



#include "FWCore/MessageService/interface/MessageLoggerDefaults.h"

namespace edm {
namespace service {

std::string 
MessageLoggerDefaults::
threshold(std::string const & dest)
{
  std::string thr = "";
  std::map<std::string,Destination>::iterator d = destination.find(dest);  
  if (d != destination.end()) {
    Destination & destin = d->second;
    thr = destin.threshold;
  }
  std::map<std::string,Destination>::iterator dd = destination.find("default");
  if ( thr == "" ) { 
    if (dd != destination.end()) {
      Destination & def_destin = dd->second;
      thr = def_destin.threshold;
    }
  }
  return thr;   
} // threshold

std::string 
MessageLoggerDefaults::
output(std::string const & dest)
{
  std::string otpt = "";
  std::map<std::string,Destination>::iterator d = destination.find(dest);  
  if (d != destination.end()) {
    Destination & destin = d->second;
    otpt = destin.output;
  }
  // There is no default output; so if we did not find the dest, then return ""
  return otpt;   
} // output

int 
MessageLoggerDefaults::
limit(std::string const & dest, std::string const & cat)
{
  int lim = NO_VALUE_SET;
  std::map<std::string,Destination>::iterator d = destination.find(dest);  
  if (d != destination.end()) {
    Destination & destin = d->second;
    std::map<std::string,Category>::iterator c = destin.category.find(cat);
    if (c != destin.category.end()) {
      lim = c->second.limit;
    } 
  }
  std::map<std::string,Destination>::iterator dd = destination.find("default");
  if ( lim == NO_VALUE_SET ) { 
    if (dd != destination.end()) {
      Destination & def_destin = dd->second;
      std::map<std::string,Category>::iterator 
		      c = def_destin.category.find(cat);
      if (c != def_destin.category.end()) {
        lim = c->second.limit;
      } 
    }
  }
  if ( lim == NO_VALUE_SET ) { 
    if (d != destination.end()) {
      Destination & destin = d->second;
      std::map<std::string,Category>::iterator 
		      cd = destin.category.find("default");
      if (cd != destin.category.end()) {
        lim = cd->second.limit;
      } 
    }
  }     
  if ( lim == NO_VALUE_SET ) { 
    if (dd != destination.end()) {
      Destination & def_destin = dd->second;
      std::map<std::string,Category>::iterator 
		      cdd = def_destin.category.find("default");
      if (cdd != def_destin.category.end()) {
        lim = cdd->second.limit;
      } 
    }
  }     
  return lim;   
} // limit

int 
MessageLoggerDefaults::
reportEvery(std::string const & dest, std::string const & cat)
{
  int re = NO_VALUE_SET;
  std::map<std::string,Destination>::iterator d = destination.find(dest);  
  if (d != destination.end()) {
    Destination & destin = d->second;
    std::map<std::string,Category>::iterator c = destin.category.find(cat);
    if (c != destin.category.end()) {
      re = c->second.reportEvery;
    } 
  }
  std::map<std::string,Destination>::iterator dd = destination.find("default");
  if ( re == NO_VALUE_SET ) { 
    if (dd != destination.end()) {
      Destination & def_destin = dd->second;
      std::map<std::string,Category>::iterator 
		      c = def_destin.category.find(cat);
      if (c != def_destin.category.end()) {
        re = c->second.reportEvery;
      } 
    }
  }
  if ( re == NO_VALUE_SET ) { 
    if (d != destination.end()) {
      Destination & destin = d->second;
      std::map<std::string,Category>::iterator 
		      cd = destin.category.find("default");
      if (cd != destin.category.end()) {
        re = cd->second.reportEvery;
      } 
    }
  }     
  if ( re == NO_VALUE_SET ) { 
    if (dd != destination.end()) {
      Destination & def_destin = dd->second;
      std::map<std::string,Category>::iterator 
		      cdd = def_destin.category.find("default");
      if (cdd != def_destin.category.end()) {
        re = cdd->second.reportEvery;
      } 
    }
  }     
  return re;   
} // reportEvery

int 
MessageLoggerDefaults::
timespan(std::string const & dest, std::string const & cat)
{
  int tim = NO_VALUE_SET;
  std::map<std::string,Destination>::iterator d = destination.find(dest);  
  if (d != destination.end()) {
    Destination & destin = d->second;
    std::map<std::string,Category>::iterator c = destin.category.find(cat);
    if (c != destin.category.end()) {
      tim = c->second.timespan;
    } 
  }
  std::map<std::string,Destination>::iterator dd = destination.find("default");
  if ( tim == NO_VALUE_SET ) { 
    if (dd != destination.end()) {
      Destination & def_destin = dd->second;
      std::map<std::string,Category>::iterator 
		      c = def_destin.category.find(cat);
      if (c != def_destin.category.end()) {
        tim = c->second.timespan;
      } 
    }
  }
  if ( tim == NO_VALUE_SET ) { 
    if (d != destination.end()) {
      Destination & destin = d->second;
      std::map<std::string,Category>::iterator 
		      cd = destin.category.find("default");
      if (cd != destin.category.end()) {
        tim = cd->second.timespan;
      } 
    }
  }     
  if ( tim == NO_VALUE_SET ) { 
    if (dd != destination.end()) {
      Destination & def_destin = dd->second;
      std::map<std::string,Category>::iterator 
		      cdd = def_destin.category.find("default");
      if (cdd != def_destin.category.end()) {
        tim = cdd->second.timespan;
      } 
    }
  }     
  return tim;   
} // timespan

int 
MessageLoggerDefaults::
sev_limit(std::string const & dest, std::string const & cat)
{
  int lim = NO_VALUE_SET;
  std::map<std::string,Destination>::iterator d = destination.find(dest);  
  if (d != destination.end()) {
    Destination & destin = d->second;
    std::map<std::string,Category>::iterator c = destin.sev.find(cat);
    if (c != destin.sev.end()) {
      lim = c->second.limit;
    } 
  }
  std::map<std::string,Destination>::iterator dd = destination.find("default");
  if ( lim == NO_VALUE_SET ) { 
    if (dd != destination.end()) {
      Destination & def_destin = dd->second;
      std::map<std::string,Category>::iterator 
		      c = def_destin.sev.find(cat);
      if (c != def_destin.sev.end()) {
        lim = c->second.limit;
      } 
    }
  }
  if ( lim == NO_VALUE_SET ) { 
    if (d != destination.end()) {
      Destination & destin = d->second;
      std::map<std::string,Category>::iterator 
		      cd = destin.sev.find("default");
      if (cd != destin.sev.end()) {
        lim = cd->second.limit;
      } 
    }
  }     
  if ( lim == NO_VALUE_SET ) { 
    if (dd != destination.end()) {
      Destination & def_destin = dd->second;
      std::map<std::string,Category>::iterator 
		      cdd = def_destin.sev.find("default");
      if (cdd != def_destin.sev.end()) {
        lim = cdd->second.limit;
      } 
    }
  }     
  return lim;   
} // sev_limit

int 
MessageLoggerDefaults::
sev_reportEvery(std::string const & dest, std::string const & cat)
{
  int re = NO_VALUE_SET;
  std::map<std::string,Destination>::iterator d = destination.find(dest);  
  if (d != destination.end()) {
    Destination & destin = d->second;
    std::map<std::string,Category>::iterator c = destin.sev.find(cat);
    if (c != destin.sev.end()) {
      re = c->second.reportEvery;
    } 
  }
  std::map<std::string,Destination>::iterator dd = destination.find("default");
  if ( re == NO_VALUE_SET ) { 
    if (dd != destination.end()) {
      Destination & def_destin = dd->second;
      std::map<std::string,Category>::iterator 
		      c = def_destin.sev.find(cat);
      if (c != def_destin.sev.end()) {
        re = c->second.reportEvery;
      } 
    }
  }
  if ( re == NO_VALUE_SET ) { 
    if (d != destination.end()) {
      Destination & destin = d->second;
      std::map<std::string,Category>::iterator 
		      cd = destin.sev.find("default");
      if (cd != destin.sev.end()) {
        re = cd->second.reportEvery;
      } 
    }
  }     
  if ( re == NO_VALUE_SET ) { 
    if (dd != destination.end()) {
      Destination & def_destin = dd->second;
      std::map<std::string,Category>::iterator 
		      cdd = def_destin.sev.find("default");
      if (cdd != def_destin.sev.end()) {
        re = cdd->second.reportEvery;
      } 
    }
  }     
  return re;   
} // sev_reportEvery

int 
MessageLoggerDefaults::
sev_timespan(std::string const & dest, std::string const & cat)
{
  int tim = NO_VALUE_SET;
  std::map<std::string,Destination>::iterator d = destination.find(dest);  
  if (d != destination.end()) {
    Destination & destin = d->second;
    std::map<std::string,Category>::iterator c = destin.sev.find(cat);
    if (c != destin.sev.end()) {
      tim = c->second.timespan;
    } 
  }
  std::map<std::string,Destination>::iterator dd = destination.find("default");
  if ( tim == NO_VALUE_SET ) { 
    if (dd != destination.end()) {
      Destination & def_destin = dd->second;
      std::map<std::string,Category>::iterator 
		      c = def_destin.sev.find(cat);
      if (c != def_destin.sev.end()) {
        tim = c->second.timespan;
      } 
    }
  }
  if ( tim == NO_VALUE_SET ) { 
    if (d != destination.end()) {
      Destination & destin = d->second;
      std::map<std::string,Category>::iterator 
		      cd = destin.sev.find("default");
      if (cd != destin.sev.end()) {
        tim = cd->second.timespan;
      } 
    }
  }     
  if ( tim == NO_VALUE_SET ) { 
    if (dd != destination.end()) {
      Destination & def_destin = dd->second;
      std::map<std::string,Category>::iterator 
		      cdd = def_destin.sev.find("default");
      if (cdd != def_destin.sev.end()) {
        tim = cdd->second.timespan;
      } 
    }
  }     
  return tim;   
} // sev_timespan




} // end of namespace service  
} // end of namespace edm  

