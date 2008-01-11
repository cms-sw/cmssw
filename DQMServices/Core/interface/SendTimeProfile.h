#ifndef _SEND_TIME_PROFILE_H_
#define _SEND_TIME_PROFILE_H_

#include <TObject.h>

/// maximum # of messages per monitoring cycle (see below)
const int N_SEND_MESS_MAX = 200;
/// maxumum # of "wait" attempts per monitoring cycle (see below)
const int N_SEND_WAIT_MAX = 2*N_SEND_MESS_MAX + 1;

/// class containing timing for sending operations
class SendTimeProfile : public TObject {
  /**
    every monitoring cycle contains the following steps:
    1. wait until receiver is ready
    2. send directory pathname, # of updated monitoring elements in directory
    3. wait until receiver is ready
    4. send updated monitoring elements in directory
    5. repeat 1-4 for all directories
    6. wait until receiver is ready
    7. send "DONE"
    Each one of 2, 4, 7 is packaged into a TMessage; 
    If we have N directories with updated monitoring elements, we should have
    - N messages of type: 2 (text)
    - N messages of type: 4 (objects)
    - 1 message of type 7 (text)
    - 2*N+1 "wait" operations
  */


  /// **** ALL TIMES ARE IN SECONDS *****
  
 public:
  /// total waiting time (in monitoring cycle)
  Float_t tot_wait;
  /// total shipping time for "description" (type: 2)
  Float_t tot_desc;
  /// total shipping time for "objects" (type: 4)
  Float_t tot_obj;
  /// shipping time for "DONE" (type: 7)
  Float_t tot_done;
  /// total shipping time = tot_desc + tot_obj + tot_done
  Float_t tot_ship;
  /// # of messages (same as "N" above)
  Int_t N_mess;
  /// = N_mess * 2 + 1
  Int_t N_wait;
  /// shipping time for description (per message)
  Float_t * t_desc; // [N_mess]
  /// shipping time for objects (per message)
  Float_t * t_obj; // [N_mess]
  /// # of objects (per message)
  Int_t * N_obj; // [N_mess]
  /// waiting time (per transaction)
  Float_t * t_wait; // [N_wait]

  void reset(void);
  SendTimeProfile(void);
  ~SendTimeProfile(void);

  ClassDef(SendTimeProfile, 1)
};


#endif // #ifndef _SEND_TIME_PROFILE_H_
