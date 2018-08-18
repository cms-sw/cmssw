#ifndef DaqSource_DTSpyHelper_h
#define DaqSource_DTSpyHelper_h


#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <netinet/in.h>
#include <cstdio>
#include <netdb.h>

class DTtcpExcp
{
     int errornumber;
  public:
   DTtcpExcp(int err):errornumber(err){};
};

class DTCtcp
{
  protected:

    int port;
    int sock;

    int connected;
    struct sockaddr_in clientAddr;
    struct sockaddr_in myaddr;


  public:

    DTCtcp();
    DTCtcp(int port);
    DTCtcp(int sock,int opt);
    DTCtcp(DTCtcp *);
    ~DTCtcp();

    DTCtcp * Accept();
    void Connect(const char *hostaddr,int port);
    void Connect(unsigned long hostaddr,int port);
    int Disconnect();
   
    short Id();
    unsigned long addr();
    int Send(char * buffer,int size); 
    int Receive(char *buffer,int size);

};

#endif
