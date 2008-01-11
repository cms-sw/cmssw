#include <TMessage.h>

class DQMMessage;

class SocketUtils 
{
public:
    
    //    static bool checkedSocketRead (seal::InetSocket *, seal::IOSelector *, void *, seal::IOSize);
    static int sendMessage (DQMMessage *mess , int sock_des);
    static int sendString (const char *mess, int sock_des);
    static int readMessage (DQMMessage*& mess, int sock_des);
};
