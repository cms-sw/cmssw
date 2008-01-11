#include "DQMServices/Core/interface/NodeBase.h"
#include "DQMServices/Core/interface/MonitorData.h"
#include "DQMServices/Core/interface/DaqMonitorROOTBackEnd.h"
#include "DQMServices/Core/interface/SocketUtils.h"
#include "DQMServices/Core/interface/DQMMessage.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <errno.h>

using namespace dqm::monitor_data;

using std::cout; using std::endl; using std::cerr;
using std::string; using std::vector;

NodeBase::NodeBase(string name) 
  : tName(name)
{
  string_reader = 0; bei = 0;
}

NodeBase::~NodeBase(void)
{
  if(string_reader)delete string_reader;
}

// name says it all!
void NodeBase::introduceYourself(int socket, const string & prefix) const
{
  if (socket == -1)
    {
      nullError("socket");
      return;
    }
  
  string name = prefix + tName;
  if(bei->getVerbose() > 0)
    cout << " Sending node name: <" << name << ">" << endl;
  SocketUtils::sendString (name.c_str (), socket);
}

// check object description: expect size=2 (<dir path>,<# of objects>)
// return success flag
bool NodeBase::checkObjDesc(const vector<string> & in)
{
  if(in.size() != 2)
    {
      cerr << " *** Error! Expected pathname and # of objects! " << endl;
      for(vector<string>::const_iterator it = in.begin(); it != in.end(); 
	  ++it)
	cout << " input = " << *it << endl;
      return false;
    }
  return true;
}

// true if node has closed connection
// if message is string, unpack (put_here)
bool NodeBase::conClosed(DQMMessage * mess, int msize, string & put_here)
{
  if(msize == -5 || msize == 0)
    {
      cout << " *** Lost connection with node " << endl;
      return true;
    }

  if (!mess->what ())
    // different kind of error (msize==-1), or EWOULDBLOCK (msize==-4)
    return false; 
  
  if(mess->what() == kMESS_STRING)
    {
      if(!string_reader)string_reader = new StringReader();

      if (string_reader->getSize () < mess->buffer ()->BufferSize ())
	string_reader->makeNew (mess->buffer ()->BufferSize ());

      mess->buffer ()->ReadString (string_reader->get_c_str (), mess->buffer ()->BufferSize ());
      // save string message
      put_here = string_reader->get_c_str();
      // ------------------------------------------
      // Source/Client has finished; will close connection
      // ------------------------------------------
      if(strncmp(string_reader->get_c_str(), 
		 Quitting.c_str(), Quitting.size())==0)
	{
	  cout << " Node has closed the connection " << endl;
	  return true;
	}
    }

  // if here, node is up
  return false;
}


// come here to connect after host & port number have been determined;
// for RootMonitorThread: prefix = "nameSourcePrefix"
// for Client: prefix = ""
// return fd of the connected socket
int NodeBase::connect(const string & prefix) const
{
  int fd;
  hostent *h = gethostbyname (host_.c_str ());
  sockaddr_in *s_address = new sockaddr_in ();
  s_address->sin_family = AF_INET;
  memcpy (&s_address->sin_addr, h->h_addr_list[0], h->h_length);
  s_address->sin_port = htons (port_);
  
  fd = socket (AF_INET, SOCK_STREAM, IPPROTO_TCP);
  int status = ::connect (fd, (sockaddr*) s_address, sizeof (sockaddr_in));      
  
  // check if connection has been established
  if (status >= 0)                                  // == 0 is enough
    {
      cout << " Started connection with "<< host_ << " at "<< port_ 
	   << endl;
      introduceYourself (fd, prefix);
    }
  else 
    fd = -1;
  return fd;
}



// return: 
//   1, if event_count <= 10, or
//  10, if event_count <= 100, or
// 100, if event_count <= 1000, etc
unsigned NodeBase::printout_period(unsigned event_count) const
{
  unsigned upper = 10;
  while(event_count > upper)upper *= 10;
  return upper/10;
}

// get list of directory contents and pack them into <dir>
// return success flag
bool NodeBase::getFullContents(MonitorElementRootFolder * folder, DirFormat & dir)
{
  if(!folder)
    {
      cerr << " *** Cannot get contents for non-existing directory " 
	   << dir.dir_path << endl;
      return false;
    }
  
  // need to list only non-null MEs
  bool skipNull = true;
  string name = dir.dir_path + ":" + folder->getChildren(skipNull);
  if(!unpackDirFormat(name, dir))
    return false;

  // at this point, <dir> has all monitoring elements the directory includes
  return true;
}

// printout info about sending <message> to <node_name>;
// <type> indicates whether this is monitorable/subscription request/etc
void NodeBase::showMessage(const string & type, const string & node_name, 
			   const string & message) const
{
  cout << type << node_name;
  unsigned size = message.size();
  if(size < 200) // must make this a configuration parameter
    cout << ": <" << message << " > " << endl;
  else
    cout << " (size: " << size << " characters) " << endl;
  
}
