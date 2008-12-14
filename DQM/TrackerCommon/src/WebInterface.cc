#include "DQM/TrackerCommon/interface/WebInterface.h"
#include "DQM/TrackerCommon/interface/CgiReader.h"
#include "DQM/TrackerCommon/interface/CgiWriter.h"
#include "DQM/TrackerCommon/interface/ContentReader.h"
#include "DQMServices/Core/interface/DQMOldReceiver.h"

using namespace std;

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

//  std::cout << "will try to connect to host : " << host << std::endl;
//  std::cout << "listening to port : " << port << std::endl;
//  std::cout << "using name : " << clientname << std::endl;
  /*
    if (*mui_p == 0)
    {
    *mui_p = new DQMOldReceiver(host, atoi(port.c_str()), clientname);
    }
    else
    {
    if (*mui_p->isConnected())
    {
    *mui_p->disconnect();
    }
    *mui_p->connect(host, atoi(port.c_str()));
    }
  */
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

  DQMStore * myBei = (*mui_p)->getBEInterface();
  
  std::cout << "will try to open " << to_open << std::endl;
  if (*mui_p)
    {
      if (to_open == "top") 
	{
	  myBei->setCurrentFolder("/");
	}
      else if (to_open == "..")
	{
	  myBei->goUp();
	}
      else 
	{
	  myBei->setCurrentFolder(to_open); 
	}
    }
  else 
    {
      std::cout << "no MUI object, subscription to " << to_open << " failed!" << std::endl;
    }
  printNavigatorXML(myBei->pwd(), out); 
}
/*void WebInterface::Subscribe(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception)
{
  std::multimap<std::string, std::string> nav_map;

  CgiReader reader(in);
  reader.read_form(nav_map);

  std::string to_subscribe = get_from_multimap(nav_map, "SubscribeTo");

  if ((*mui_p))
    {
      std::cout << "will try to subscribe to " << to_subscribe << std::endl;
      (*mui_p)->subscribe(to_subscribe); 
      log4cplus::helpers::sleep(2); // delay for the subscription to be made
    }
  else 
    {
      std::cout << "no MUI object, subscription to " << to_subscribe << " failed!" << std::endl;
    }

  std::string current = get_from_multimap(nav_map, "Current");
  if (current == "")
    {
      current = "top";
    }
  printNavigatorXML(current, out);
}

void WebInterface::Unsubscribe(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception)
{
  std::multimap<std::string, std::string> nav_map;

  CgiReader reader(in);
  reader.read_form(nav_map);

  std::string to_unsubscribe = get_from_multimap(nav_map, "UnsubscribeFrom");
  
  if ((*mui_p))
    {
      std::cout << "will try to unsubscribe from " << to_unsubscribe << std::endl;
      (*mui_p)->unsubscribe(to_unsubscribe); 
      log4cplus::helpers::sleep(2); // delay for the unsubscription to be made
    }
  else 
    {
      std::cout << "no MUI object, unsubscription from" << to_unsubscribe << " failed!" << std::endl;
    }

  std::string current = get_from_multimap(nav_map, "Current");
  if (current == "")
    {
      current = "top";
    }
  printNavigatorXML(current, out);
}*/

void WebInterface::printNavigatorXML(std::string current, xgi::Output * out)
{
  if (!(*mui_p)) 
    {
      cout << "NO GUI!!!" << endl;
      return;
    }
  
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  
  *out << "<?xml version=\"1.0\" ?>" << endl;
   
  *out << "<navigator>" << endl;
  
  *out << "<current>" << current << "</current>" << endl;
  
  DQMStore * myBei = (*mui_p)->getBEInterface();
  ContentReader reader(myBei);
  
  std::list<std::string>::iterator it;
  
  std::list<std::string> open_l;
  reader.give_subdirs(current, open_l, "SuperUser");
  for (it = open_l.begin(); it != open_l.end(); it++)
    *out << "<open>" << *it << "</open>" << endl;
  
  std::list<std::string> subscribe_l;
  reader.give_files(current, subscribe_l, false);
  for (it = subscribe_l.begin(); it != subscribe_l.end(); it++)
    *out << "<subscribe>" << *it << "</subscribe>" << endl;
  
  std::list<std::string> unsubscribe_l;
  reader.give_files(current, unsubscribe_l, true);
  for (it = unsubscribe_l.begin(); it != unsubscribe_l.end(); it++)
    *out << "<unsubscribe>" << *it << "</unsubscribe>" << endl;
  
  *out << "</navigator>" << endl;
  

    std::cout << "<?xml version=\"1.0\" ?>" << endl;
    
    std::cout << "<navigator>" << endl;
    
    std::cout << "<current>" << current << "</current>" << endl;
    
    for (it = open_l.begin(); it != open_l.end(); it++)
    std::cout << "<open>" << *it << "</open>" << endl;
    
    for (it = subscribe_l.begin(); it != subscribe_l.end(); it++)
    std::cout << "<subscribe>" << *it << "</subscribe>" << endl;
    
    for (it = unsubscribe_l.begin(); it != unsubscribe_l.end(); it++)
    std::cout << "<unsubscribe>" << *it << "</unsubscribe>" << endl;
    
    std::cout << "</navigator>" << endl;

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

  DQMStore * myBei = (*mui_p)->getBEInterface();
  
  std::cout << "will try to open " << to_open << std::endl;
  if (*mui_p)
    {
      if (to_open == "top") 
	{
	  myBei->cd();
	}
      else if (to_open == "..")
	{
	  myBei->goUp();
	}
      else 
	{
	  myBei->setCurrentFolder(to_open); 
	}
    }
  else 
    {
      std::cout << "no MUI object, subscription to " << to_open << " failed!" << std::endl;
    }
  printContentViewerXML(myBei->pwd(), out); 
}

void WebInterface::printContentViewerXML(std::string current, xgi::Output * out)
{
  if (!(*mui_p)) 
    {
      cout << "NO GUI!!!" << endl;
      return;
    }
  
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  
  *out << "<?xml version=\"1.0\" ?>" << endl;
   
  *out << "<contentViewer>" << endl;
  
  *out << "<current>" << current << "</current>" << endl;
  
  DQMStore * myBei = (*mui_p)->getBEInterface();
  ContentReader reader(myBei);
  
  std::list<std::string>::iterator it;
  
  std::list<std::string> open_l;
  reader.give_subdirs(current, open_l, "SuperUser");
  for (it = open_l.begin(); it != open_l.end(); it++)
    *out << "<open>" << *it << "</open>" << endl;
  
  std::list<std::string> view_l;
  reader.give_files(current, view_l, true);
  for (it = view_l.begin(); it != view_l.end(); it++)
    *out << "<view>" << *it << "</view>" << endl;
  
  *out << "</contentViewer>" << endl;
}


void WebInterface::DrawGif(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception)
{
  std::multimap<std::string, std::string> view_multimap;
  ME_MAP view_map;

  CgiReader cgi_reader(in);
  cgi_reader.read_form(view_multimap);

  std::string current = get_from_multimap(view_multimap, "Current");
  

  if (!(*mui_p)) 
    {
      std::cout << "The mui pointer is empty!" << std::endl;
      return;
    }

  DQMStore * myBei = (*mui_p)->getBEInterface();

  ContentReader con_reader(myBei);  
  multimap<string,string>::iterator lower = view_multimap.lower_bound("View");
  multimap<string,string>::iterator upper = view_multimap.upper_bound("View");
  multimap<string,string>::iterator it;
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
