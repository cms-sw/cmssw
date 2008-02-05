#include "DQMServices/Core/src/SendTimeProfile.h"

ClassImp(SendTimeProfile)

// constructor
SendTimeProfile::SendTimeProfile(void)
{
  t_desc = new Float_t[N_SEND_MESS_MAX];
  t_obj = new Float_t[N_SEND_MESS_MAX];
  t_wait = new Float_t[N_SEND_WAIT_MAX];
  N_obj = new Int_t[N_SEND_MESS_MAX];
  reset();
}

// destructor
SendTimeProfile::~SendTimeProfile(void){
  delete [] t_desc; delete [] t_obj; delete [] t_wait; delete [] N_obj;
}

void SendTimeProfile::reset(void)
{
  tot_wait = tot_desc = tot_obj = tot_done = tot_ship = 0;
  N_mess = N_wait = 0;
}

