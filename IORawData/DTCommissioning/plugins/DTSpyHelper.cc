#include "DTSpyHelper.h"
#include <cerrno>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstring>

#ifdef __wasAPPLE__
typedef int socklen_t;
#endif

DTCtcp::DTCtcp()
{
   DTCtcp(0);
}

DTCtcp::DTCtcp(int localport)
{
//  struct sockaddr_in myaddr;
 
    connected=false;

     printf("zeroing...\n");
    bzero ((char *) &myaddr, sizeof(myaddr));
     printf("zeroing done..\n");


   sock = socket (AF_INET, SOCK_STREAM, 0);
     printf("create socket..\n");

   if (sock < 0)
   {
     printf("no socket...\n");
      throw DTtcpExcp(errno);
   }
  

    myaddr.sin_family       = AF_INET;
    myaddr.sin_port         = htons (localport);
 
    //int blen = 65536;
    int blen = 65536*8;

//     printf("setting socket opts buf...\n");
//    if(setsockopt(sock,SOL_SOCKET,SO_SNDBUF,(char *)&blen,sizeof(blen))<0)
//      throw DTtcpExcp(errno);
//    if(setsockopt(sock,SOL_SOCKET,SO_RCVBUF,(char *)&blen,sizeof(blen))<0)
//      throw DTtcpExcp(errno);
     printf("setting socket opts reuse...\n");
    if(setsockopt(sock,SOL_SOCKET,SO_REUSEADDR,(char *)&blen,sizeof(blen))<0)
      throw DTtcpExcp(errno);
//     printf("setting socket opts nodelay...\n");
//    if(setsockopt(sock,SOL_SOCKET,TCP_NODELAY,(char *)&blen,sizeof(blen))<0)
//      throw;
     printf("binding...\n");

    port = localport;

    if (port)
        if(bind(sock,(struct sockaddr *)&myaddr,sizeof (myaddr)) < 0)
        { perror ("bind failed");
          throw DTtcpExcp(errno);
        }

          
}

DTCtcp::DTCtcp(int snew, int opt)
{

   connected = true;
   port =0;

   sock = snew;
}

DTCtcp::DTCtcp(DTCtcp* aconn)
{

   connected = aconn->connected;
   port = aconn->port;

   sock = aconn->sock;
}

DTCtcp::~DTCtcp()
{
  printf("deleting DTCtcp\n");
  //if (connected) shutdown(sock,2);
  shutdown(sock,SHUT_RDWR);
  close (sock);
}

short 
DTCtcp::Id()
{
    long maddr = clientAddr.sin_addr.s_addr;
    maddr = htonl (maddr);
    return maddr&0xff;
}

unsigned long
DTCtcp::addr()
{
    unsigned long maddr = clientAddr.sin_addr.s_addr;
    maddr = htonl (maddr);
    return maddr;
}

int
DTCtcp::Disconnect()
{
  connected = false;
  return shutdown(sock,SHUT_RDWR);
}

DTCtcp *
DTCtcp::Accept()
{

 //   struct sockaddr_in  clientAddr; /* client's address */    

    bzero ((char *) &clientAddr, sizeof (clientAddr));

    if (listen (sock, 2) < 0)
        {
        perror ("listen failed");
        throw DTtcpExcp(errno);
        }

    int len = sizeof (clientAddr);

    int snew = accept (sock, (struct sockaddr *) &clientAddr,(socklen_t *) &len);
    if (snew <=0) 
    {
        perror ("accept failed");
        throw DTtcpExcp(errno);
    }
 
    return new DTCtcp(snew,0);
}

void 
DTCtcp::Connect(unsigned long in,int toport)
{
    clientAddr.sin_family      = AF_INET;
    clientAddr.sin_addr.s_addr = htonl (in);
    clientAddr.sin_port        = htons (toport);
  
 if (connect (sock, (struct sockaddr  *)&clientAddr, sizeof (clientAddr)) < 0)
{
        perror ("connect failed");
        throw DTtcpExcp(errno);
}
  connected = true;
}

void 
DTCtcp::Connect(const char *host,int toport)
{
    clientAddr.sin_family      = AF_INET;
    clientAddr.sin_addr.s_addr = inet_addr (host);
    clientAddr.sin_port        = htons (toport);
  
 if (connect (sock, (struct sockaddr  *)&clientAddr, sizeof (clientAddr)) < 0)
{
        perror ("connect failed");
        throw DTtcpExcp(errno);
}
  connected = true;
}

int
DTCtcp::Receive(char *buffer,int size)
{
//    return  read (sock, buffer,size) ;
 
    int howmany = 0;
    int toberead = size;
    do
    {
      //int readnow = recv (sock, &buffer[howmany], toberead,MSG_WAITALL) ;
      int readnow = recv (sock, &buffer[howmany], toberead,0) ;
      //if (readnow < 0 ) {printf("some rrorrs...%d\n",errno); return -1;}
      if (readnow <= 0 ) 
           {printf("some rrorrs...%d\n",errno); throw DTtcpExcp(errno);}
      else { 
        howmany+=readnow; toberead-=readnow;}
    } while (toberead>0);
    return howmany;
}

int
DTCtcp::Send(char *buffer,int size)
{
  if (connected==false) throw DTtcpExcp(EPIPE);
    int myret =  write (sock, buffer, size) ;
    if (myret<0) throw DTtcpExcp(errno);
    return myret;
}
