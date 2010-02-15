#ifndef QUEUE_DEFS_H
# define QUEUE_DEFS_H

/*
 * queue_defs.h - common macros and definitions for the public message
 *                queue example.
 */

#define QUEUE_ID 137      /* base ID of queue to generate.M->S = odd, S->M = even                         */
#define MAX_MSG_SIZE 0x2000  /* size (in bytes) of largest message we'll send. This is the system max  */

#define MSGQ_MESSAGE_TYPE_RANGE 0xfff

#define MSQM_MESSAGE_TYPE_NOP  0x000
#define MSQM_MESSAGE_TYPE_STOP 0x002
#define MSQM_MESSAGE_TYPE_MCS  0x004
#define MSQM_MESSAGE_TYPE_PRG  0x008
#define MSQM_MESSAGE_TYPE_WEB  0x00a
#define MSQM_MESSAGE_TYPE_TRP  0x00c

#define MSQS_MESSAGE_TYPE_NOP  0x000
#define MSQS_MESSAGE_TYPE_SLA  0x200
#define MSQS_MESSAGE_TYPE_MCR  0x202
#define MSQS_MESSAGE_TYPE_STOP 0x204
#define MSQS_MESSAGE_TYPE_PRR  0x208
#define MSQS_MESSAGE_TYPE_WEB  0x20a
#define MSQS_MESSAGE_TYPE_TRR  0x20c

#define NUMERIC_MESSAGE_SIZE 32

#define PIPE_READ 0
#define PIPE_WRITE 1
#define MAX_PIPE_BUFFER_SIZE 0x1000

#include <sys/msg.h> 
//struct msgbuf
//  {
//    unsigned long int mtype;    /* type of received/sent message */
//    char mpayload[1];           /* message payload */
//  };

namespace evf{
  struct prg{
    prg():ls(0),ps(0),nbp(0),nba(0),Ms(0),ms(0){}
    unsigned int ls;
    unsigned int ps;
    unsigned int nbp;
    unsigned int nba;
    unsigned int Ms;
    unsigned int ms;
    unsigned int dqm;
  };
  

}

#endif /* QUEUE_DEFS_H */
