#include "EventFilter/Utilities/interface/ServiceWebRegistry.h"
#include "EventFilter/Utilities/interface/ServiceWeb.h"

#include <iostream>

namespace evf{
ServiceWebRegistry::ServiceWebRegistry(const edm::ParameterSet &ps){
  std::cout << "Service registry constructor at " << std::hex << (unsigned long) this << std::dec << std::endl;
}



std::vector<ServiceWeb*> ServiceWebRegistry::getWebs()
{
  std::cout << " service web registry " << std::hex << (unsigned long) this << std::dec 
	    << " has " << clm_.size() << " services registered " << std::endl;
  std::vector<ServiceWeb*> retval;
  for(idct i = clm_.begin(); i != clm_.end(); i++)
    {
      std::cout << "service " << (*i).first << " has web " << std::endl;
      retval.push_back((*i).second);
    }
  return retval;
}


void ServiceWebRegistry::invoke(xgi::Input *in, xgi::Output *out, const std::string &name)
{
  idct i = clm_.find(name);
  if(i != clm_.end())
    {
      try{
	(*i).second->defaultWebPage(in,out);
      }
      catch(...)
	{
	  std::cout << "exception caught when calling serviceweb page for " << name << std::endl;
	}
    }
}


void ServiceWebRegistry::publish(xdata::InfoSpace *is)
{
    idct i = clm_.begin();
    while (i != clm_.end())
      {
	(*i).second->publish(is);
	i++;
      }

}

void ServiceWebRegistry::clear()
{
  std::cout << "Service registry clear for " << std::hex << (unsigned long) this << std::dec << std::endl; 
  clm_.clear();
}


} //end namespace evf
