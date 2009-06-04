#include "DQM/TrackerCommon/interface/MessageDispatcher.h"
#include <iostream>

std::string Message::getType() 
{ 
  if (type == info) { return "INFO"; } 
  else if (type == warning) { return "WARNING"; } 
  else { return "ERROR"; } 
}

void MessageDispatcher::dispatchMessages(xgi::Output *out)
{
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  
  *out << "<?xml version=\"1.0\" ?>" << std::endl;
   
  *out << "<Messages>" << std::endl;
  
  int i = 1;
  std::vector<Message *>::iterator it;
  for (it = undispatched.begin(); it != undispatched.end(); it++)
    {
      *out << "<Message"  << i << ">" << std::endl;
      
      *out << "<Type>"  << (*it)->getType()  << "</Type>"  << std::endl;
      *out << "<Title>" << (*it)->getTitle() << "</Title>" << std::endl;
      *out << "<Text>"  << (*it)->getText()  << "</Text>"  << std::endl;
      
      *out << "</Message" << i << ">" << std::endl;

      i++;
    }
  
  *out << "</Messages>" << std::endl;

  undispatched.clear();
}
