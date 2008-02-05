#ifndef _NODE_BASE_H_
#define _NODE_BASE_H_

class DQMMessage;
class DaqMonitorBEInterface;
class MonitorElement;
template <class T> class MonitorElementT;
class MonitorElementRootFolder;

// DQM classes
#include "DQMServices/Core/interface/StringUtil.h"
#include "DQMServices/Core/interface/ProfileTimer.h"
#include "DQMServices/Core/interface/MonitorElementRootT.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

// ROOT classes
#include <TNamed.h>

// std classes
#include <string>
#include <vector>
#include <iostream>


class NodeBase : public StringUtil
{
 public:
  NodeBase(std::string name);
  virtual ~NodeBase(void);
  
 private:
  /// use this class to read strings 
  /// (cannot use C++ strings with TMessage::ReadString)
  class StringReader 
  {
  public:
    void makeNew(unsigned n)
    {
      if(n)
	{
	  if(put_here)delete [] put_here;
	  size = n;
	  put_here = new char[n];
	}
    }
    char * get_c_str(void){return put_here;}
    int getSize(void) const{return size;}
    StringReader() {
      put_here = 0; size = 0;
      makeNew(1024);
    }
    ~StringReader()
    {
      if(put_here)delete [] put_here;     
    }
  private:
    char * put_here;
    int size; // string size
  };

  StringReader * string_reader;

 protected:
  /// server name
  std::string host_; 
  /// server port
  unsigned port_; 
  /// node name
  std::string tName; 

  /// upon a collector crash, sources and clients will automatically attempt
  /// to reconnect with a time delay; this is the parameter in secs
  /// (if < 0, the node will NOT attempt to reconnect...)
  int reconnectDelay_;
  /// set reconnect delay parameter (in secs);
  /// use delay < 0 for no reconnection attempts
  void setReconnectDelay(int delay){reconnectDelay_ = delay;}
  /// true if node should attempt to reconnect upon a connection problem
  bool shouldReconnect(void){return reconnectDelay_ > 0;}
  /// use to get hold of structure with monitoring elements that class owns
  DaqMonitorBEInterface *bei;
  ///
  void introduceYourself(int socket, const std::string & prefix="")const;
  /// come here to connect passing host & port number;
  /// use noBlock = kTRUE for RootMonitorThread, kFALSE for Client;
  /// returns socket descriptor if connect is successfull, -1 otherwise
  int connect(const std::string & prefix="") const;
  /// check object description: expect size=2 (<dir path>,<# of objects>)
  /// return success flag
  bool checkObjDesc(const std::vector<std::string> & in);
  /// get list of directory contents and pack them into <dir>
  /// return success flag
  bool getFullContents(MonitorElementRootFolder * folder, DirFormat & dir);

  /// convert "MonitorElement*" to "THX *", by using "MERX *" as indermediate step
  /// (eg. go from "MonitorElement *" to "TH1F *" via "MonitorElementRootH1 *", etc)
  template <class THX, class MERX> 
    THX * convertObject(MonitorElement * me)
  {
    MERX * temp = dynamic_cast<MERX *> (me);
    if(!temp)
      std::cerr<<" *** Failed to obtain MonitorElementROOTX * with dynamic_cast" 
	       << " called with object " << me->getName() << std::endl; 
    THX * ret = (THX *) (temp->operator->());
    if(!ret)
      std::cerr << 
	" *** Failed to obtain THX * with dynamic_cast called with object "
		<< me->getName() << std::endl;
    return ret;
  }

  /// return: 
  ///   1, if event_count <= 10, or
  ///  10, if event_count <= 100, or
  /// 100, if event_count <= 1000, etc
  unsigned printout_period(unsigned event_count) const;
  /// true if node has closed connection
  /// if message is string, unpack (put_here)
  bool conClosed(DQMMessage * mess, int msize, std::string & put_here);
  /// to be called by inheriting class only
  virtual bool isConnected(void) const
  {
    std::cerr << " *** Error! virtual NodeBase::isConnected should not be called!"
	      << std::endl;
    return false;
  }
  /// to be called by inheriting class only
  virtual void connect(void)
  {
    std::cerr << " *** Error! virtual NodeBase::connect should not be called!"
	      << std::endl;
  }

  void setBackEndInterface(DaqMonitorBEInterface * getThis)
  {bei = getThis; assert(bei != 0);}

  // printout info about sending <message> to <node_name>;
  // <type> indicates whether this is monitorable/subscription request/etc
  void showMessage(const std::string & type, const std::string & node_name, 
		   const std::string & message) const;

  
};


#endif // _NODE_BASE_H_
