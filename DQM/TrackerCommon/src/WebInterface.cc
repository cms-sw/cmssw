#include "DQM/TrackerCommon/interface/WebInterface.h"
#include "DQM/TrackerCommon/interface/CgiReader.h"
#include "DQM/TrackerCommon/interface/CgiWriter.h"
#include "DQM/TrackerCommon/interface/ContentReader.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"


WebInterface::WebInterface(std::string _exeURL, std::string _appURL) 
{
  exeURL = _exeURL;
  appURL = _appURL;
  
  std::cout << "created a WebInterface for the DQMClient, the url = " << appURL << std::endl;
  std::cout << "within context = " << exeURL << std::endl;
 
  dqmStore_ = edm::Service<DQMStore>().operator->();
}


std::string WebInterface::get_from_multimap(std::multimap<std::string, std::string> &mymap, std::string key)
{
  std::multimap<std::string, std::string>::iterator it;
  it = mymap.find(key);
  if (it != mymap.end())
    {
      return (it->second);
    }
  return "";
}

void WebInterface::handleRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  handleStandardRequest(in, out);
  handleCustomRequest(in, out);

  // dispatch any messages:
  if (msg_dispatcher.hasAnyMessages()) msg_dispatcher.dispatchMessages(out);
}

void WebInterface::Default(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  CgiWriter writer(out, getContextURL());
  writer.output_preamble();
  writer.output_head();
  page_p->printHTML(out);
  writer.output_finish();
}

void WebInterface::add(std::string name, WebElement * element_p)
{
  page_p->add(name, element_p);
}

void WebInterface::handleStandardRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  std::multimap<std::string, std::string> request_multimap;
  CgiReader reader(in);
  reader.read_form(request_multimap);

  // get the string that identifies the request:
  std::string requestID = get_from_multimap(request_multimap, "RequestID");

  if (requestID == "Configure")    Configure(in, out);
  if (requestID == "Open")         Open(in, out);
//  if (requestID == "Subscribe")    Subscribe(in, out);
//  if (requestID == "Unsubscribe")  Unsubscribe(in, out);
  if (requestID == "ContentsOpen") ContentsOpen(in, out);
  if (requestID == "Draw")         DrawGif(in, out);
}

void WebInterface::Configure(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception)
{
  CgiReader reader(in);
  reader.read_form(conf_map);

  std::string host = get_from_multimap(conf_map, "Hostname");
  std::string port = get_from_multimap(conf_map, "Port");
  std::string clientname = get_from_multimap(conf_map, "Clientname");

}

void WebInterface::Open(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception)
{
  std::multimap<std::string, std::string> nav_map;

  CgiReader reader(in);
  reader.read_form(nav_map);

  std::string to_open = get_from_multimap(nav_map, "Open");
  std::string current = get_from_multimap(nav_map, "Current");
  
  // if the user does not want to go to the top directory...
  if (to_open != "top")
    {
      // ...and if the current directory is not the top...
      if (current != "top")
	{
	  // ...and if we don't want to go one directory up...
	  if (to_open != "..")
	    {
	      // ...then we need to add the current directory at the beginning of the to_open string
	      to_open = current + "/" + to_open;
	    }	  
	}
    }

  
  std::cout << "will try to open " << to_open << std::endl;
  if (dqmStore_)
    {
      if (to_open == "top") 
	{
	  dqmStore_->setCurrentFolder("/");
	}
      else if (to_open == "..")
	{
	  dqmStore_->goUp();
	}
      else 
	{
	  dqmStore_->setCurrentFolder(to_open); 
	}
      printNavigatorXML(dqmStore_->pwd(), out); 
    }
  else 
    {
      std::cout << "no DQMStore object, subscription to " << to_open << " failed!" << std::endl;
    }

}

void WebInterface::printNavigatorXML(std::string current, xgi::Output * out)
{
  if (!dqmStore_) 
    {
     std::cout << "NO GUI!!!" << std::endl;
      return;
    }
  
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  
  *out << "<?xml version=\"1.0\" ?>" << std::endl;
   
  *out << "<navigator>" << std::endl;
  
  *out << "<current>" << current << "</current>" << std::endl;
  
  ContentReader reader(dqmStore_);
  
  std::list<std::string>::iterator it;
  
  std::list<std::string> open_l;
  reader.give_subdirs(current, open_l, "SuperUser");
  for (it = open_l.begin(); it != open_l.end(); it++)
    *out << "<open>" << *it << "</open>" << std::endl;
  
  std::list<std::string> subscribe_l;
  reader.give_files(current, subscribe_l, false);
  for (it = subscribe_l.begin(); it != subscribe_l.end(); it++)
    *out << "<subscribe>" << *it << "</subscribe>" << std::endl;
  
  std::list<std::string> unsubscribe_l;
  reader.give_files(current, unsubscribe_l, true);
  for (it = unsubscribe_l.begin(); it != unsubscribe_l.end(); it++)
    *out << "<unsubscribe>" << *it << "</unsubscribe>" << std::endl;
  
  *out << "</navigator>" << std::endl;
  

    std::cout << "<?xml version=\"1.0\" ?>" << std::endl;
    
    std::cout << "<navigator>" << std::endl;
    
    std::cout << "<current>" << current << "</current>" << std::endl;
    
    for (it = open_l.begin(); it != open_l.end(); it++)
    std::cout << "<open>" << *it << "</open>" << std::endl;
    
    for (it = subscribe_l.begin(); it != subscribe_l.end(); it++)
    std::cout << "<subscribe>" << *it << "</subscribe>" << std::endl;
    
    for (it = unsubscribe_l.begin(); it != unsubscribe_l.end(); it++)
    std::cout << "<unsubscribe>" << *it << "</unsubscribe>" << std::endl;
    
    std::cout << "</navigator>" << std::endl;

}

void WebInterface::ContentsOpen(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception)
{
  std::multimap<std::string, std::string> nav_map;

  CgiReader reader(in);
  reader.read_form(nav_map);

  std::string to_open = get_from_multimap(nav_map, "Open");
  std::string current = get_from_multimap(nav_map, "Current");
  
  // if the user does not want to go to the top directory...
  if (to_open != "top")
    {
      // ...and if the current directory is not the top...
      if (current != "top")
	{
	  // ...and if we don't want to go one directory up...
	  if (to_open != "..")
	    {
	      // ...then we need to add the current directory at the beginning of the to_open string
	      to_open = current + "/" + to_open;
	    }	  
	}
    }

  
  std::cout << "will try to open " << to_open << std::endl;
  if (dqmStore_)
    {
      if (to_open == "top") 
	{
	  dqmStore_->cd();
	}
      else if (to_open == "..")
	{
	  dqmStore_->goUp();
	}
      else 
	{
	  dqmStore_->setCurrentFolder(to_open); 
	}
      printContentViewerXML(dqmStore_->pwd(), out); 
    }
  else 
    {
      std::cout << "no DQMStore object, subscription to " << to_open << " failed!" << std::endl;
    }
}

void WebInterface::printContentViewerXML(std::string current, xgi::Output * out)
{
  if (!(dqmStore_)) 
    {
     std::cout << "NO GUI!!!" << std::endl;
      return;
    }
  
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  
  *out << "<?xml version=\"1.0\" ?>" << std::endl;
   
  *out << "<contentViewer>" << std::endl;
  
  *out << "<current>" << current << "</current>" << std::endl;
  
  ContentReader reader(dqmStore_);
  
  std::list<std::string>::iterator it;
  
  std::list<std::string> open_l;
  reader.give_subdirs(current, open_l, "SuperUser");
  for (it = open_l.begin(); it != open_l.end(); it++)
    *out << "<open>" << *it << "</open>" << std::endl;
  
  std::list<std::string> view_l;
  reader.give_files(current, view_l, true);
  for (it = view_l.begin(); it != view_l.end(); it++)
    *out << "<view>" << *it << "</view>" << std::endl;
  
  *out << "</contentViewer>" << std::endl;
}


void WebInterface::DrawGif(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception)
{
  std::multimap<std::string, std::string> view_multimap;
  ME_MAP view_map;

  CgiReader cgi_reader(in);
  cgi_reader.read_form(view_multimap);

  std::string current = get_from_multimap(view_multimap, "Current");
  

  if (dqmStore_) 
    {
      std::cout << "The DQMStore pointer is empty!" << std::endl;
      return;
    }

  ContentReader con_reader(dqmStore_);  
  std::multimap<std::string,std::string>::iterator lower = view_multimap.lower_bound("View");
  std::multimap<std::string,std::string>::iterator upper = view_multimap.upper_bound("View");
  std::multimap<std::string,std::string>::iterator it;
  for (it = lower; it != upper; it++)
    {
      std::string name = it->second;
      MonitorElement *pointer = con_reader.give_ME(name);
      if (pointer != 0) 
	{
	  view_map.add(name, pointer);
	  std::cout << "ADDING " << name << " TO view_map!!!" << std::endl;
	}
    }

  // Print the ME_map into a file
  std::string id = get_from_multimap(view_multimap, "DisplayFrameName");
  std::cout << "will try to print " << id << std::endl;
  
  // And return the URL of the file:
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  *out << "<?xml version=\"1.0\" ?>" << std::endl;
  *out << "<fileURL>" << std::endl;
  *out << getContextURL() + "/temporary/" + id + ".gif" << std::endl;
  *out << "</fileURL>" << std::endl;

}

void WebInterface::printMap(ME_MAP view_map, std::string id)
{
  view_map.print(id);
}

void WebInterface::sendMessage(std::string title, std::string text, MessageType type)
{
  Message * msg = new Message(title, text, type);
  msg_dispatcher.add(msg);
}
