#ifndef EVENTFILTER_UTILITIES_QUEUE_DEFS_H
#define EVENTFILTER_UTILITIES_QUEUE_DEFS_H

/*
 * queue_defs.h - common macros and definitions for the public message
 *                queue
 */

#define QUEUE_ID 137      /* base ID of queue to generate.M->S = odd, S->M = even */
#define MAX_MSG_SIZE 0x8000  /* size (in bytes) of largest message we'll ever send. 
				This is the system max  */

#define MSGQ_MESSAGE_TYPE_RANGE 0xfff

#define MSQM_MESSAGE_TYPE_NOP  0x000
#define MSQM_MESSAGE_TYPE_STOP 0x002
#define MSQM_MESSAGE_TYPE_FSTOP 0x003
#define MSQM_MESSAGE_TYPE_MCS  0x004
#define MSQM_MESSAGE_TYPE_PRG  0x008
#define MSQM_MESSAGE_TYPE_WEB  0x00a
#define MSQM_MESSAGE_TYPE_TRP  0x00c
#define MSQM_VULTURE_TYPE_STA  0x00e
#define MSQM_VULTURE_TYPE_STP  0x010


#define MSQS_MESSAGE_TYPE_NOP  0x000
#define MSQS_MESSAGE_TYPE_SLA  0x200
#define MSQS_MESSAGE_TYPE_MCR  0x202
#define MSQS_MESSAGE_TYPE_STOP 0x204
#define MSQS_MESSAGE_TYPE_PRR  0x208
#define MSQS_MESSAGE_TYPE_WEB  0x20a
#define MSQS_MESSAGE_TYPE_TRR  0x20c
#define MSQS_VULTURE_TYPE_ACK  0x20e
#define MSQS_VULTURE_TYPE_DON  0x210

#define NUMERIC_MESSAGE_SIZE 32

#define PIPE_READ 0
#define PIPE_WRITE 1
#define MAX_PIPE_BUFFER_SIZE 0x1000

#include <sys/msg.h> 
#ifdef __APPLE__
// Unsupported on macosx. We define a dummy msgbuf struct just to make sure it
// compiles fine.
struct msgbuf
  {
    unsigned long int mtype;    /* type of received/sent message */
    char mtext[1];           /* message payload */
  };
#endif

namespace evf{
  struct prg{
    prg():ls(0),eols(0),ps(0),nbp(0),nba(0),Ms(0),ms(0),dqm(0),trp(0){}
    unsigned int ls;
    unsigned int eols;
    unsigned int ps;
    unsigned int nbp;
    unsigned int nba;
    unsigned int Ms;
    unsigned int ms;
    unsigned int dqm;
    unsigned int trp;
  };
  

}

#endif /* QUEUE_DEFS_H */
