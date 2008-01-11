#include "DQMServices/Core/interface/SocketUtils.h"
#include "DQMServices/Core/interface/DQMMessage.h"
#include <sstream>
#include <iostream>
#include <string>
#include <TClass.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <errno.h>


int
checkedSocketRead (int s, void *buffer, int nbytes)
{
  int n = 0;
  int dataSize = 0;

  while (nbytes != dataSize) {
	n = recv (s, (char*) buffer + dataSize, nbytes - dataSize, 0);
	if (n <= 0)                                                   
	  {
	    std::cout << "Error no. " << errno << " whilie checking socket"
		      << " for reading " << std::endl;
	    int e = errno;
	    char ee[256];
	    puts(strerror_r(e,ee,256));
	    return 0;
	  }
	dataSize += n;
  }    
  return dataSize;
}

int
checkedSocketWrite (int s, void *buffer, int nbytes)
{
  int n = 0;
  int dataSize = 0;

  while (nbytes != dataSize) {
    n = send (s, (char *) buffer + dataSize, nbytes - dataSize, 0);
    if (n <= 0)                                                      //In last version it was returning 0 also if (n = 0)
      {
	std::cout << "Error no. " << errno << " whilie checking socket"
		      << " for writing " << std::endl;
	int e = errno;
	char ee[256];
	puts(strerror_r(e,ee,256));
	return 0;
      }
    dataSize += n;
  }    
  return dataSize;
}

int SocketUtils::sendMessage (DQMMessage *mess, int sock) 
{
      int whatSize = sizeof (int);
      int messageSize = sizeof (int);
      int messageLength = host2net (mess->length ());
      int what = host2net (mess->what ());

      char * message = new char [whatSize + messageSize + mess->length ()];
      memcpy (message, &messageLength, messageSize); 
      memcpy (message + sizeof (int), &what, whatSize); 
      if (mess->buffer () && mess->length () > 0) 
	memcpy (message + 2 * sizeof (int), mess->buffer ()->Buffer (), mess->length ());
      
      int dsent = checkedSocketWrite (sock, message, whatSize+messageSize+mess->length ());

      delete [] message;
      return dsent;	 
}


int SocketUtils::sendString (const char *mess, int sock) 
{
  int stringSize = strlen (mess) + 1;
  int lenSize = sizeof (int);
  int whatSize = sizeof (int);
    
  int what = host2net (kMESS_STRING);
  int len = host2net (stringSize);
    
  char *buf =  new char[stringSize + whatSize + lenSize];
  
  memcpy(buf, &len, lenSize);
  memcpy(buf+lenSize, &what, whatSize);
  memcpy(buf+lenSize+whatSize, mess, stringSize);

  int dsent = checkedSocketWrite (sock, buf, lenSize+whatSize+stringSize);
  delete[] buf;
  
  return (dsent - sizeof (int));    //was -2*sizeof(....)
}

int SocketUtils::readMessage (DQMMessage *& mess, int sock)
{
  if(mess) delete mess;
  mess = new DQMMessage;
		
  int objectLength;
  int messageWhat;
	
  int n = checkedSocketRead (sock, &objectLength, sizeof (objectLength));
	
  if (n <= 0)
   {
     mess = 0;
     return n;
   }


  n = checkedSocketRead (sock, &messageWhat, sizeof (messageWhat));
  if (n <= 0)
    {
      mess = 0;  
      return n;
    }    
  objectLength = net2host (objectLength);
  messageWhat = net2host (messageWhat);
  mess->setWhat (messageWhat);
    
  int dataReceived = 0;
  if (objectLength > 0)
    {
      char *realBuffer = new char [objectLength];

      int dataReceived = checkedSocketRead (sock, realBuffer, objectLength);
      if (dataReceived <= 0)
	{
	  mess = 0;
	  return dataReceived;
	}
      DQMRootBuffer *buffer = new DQMRootBuffer (TBuffer::kRead, objectLength, realBuffer, true);
      mess->setBuffer (buffer, objectLength);

      if (mess->what () != kMESS_STRING)
	{
	  mess->buffer ()->InitMap ();
	  mess->buffer ()->Reset ();
	}
    }
  return (dataReceived + sizeof (int));
}
