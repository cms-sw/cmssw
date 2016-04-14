/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com) 
*    
****************************************************************************/

#ifndef DataFormats_TotemDigi_TotemTriggerCounters
#define DataFormats_TotemDigi_TotemTriggerCounters

/**
 * Trigger counters from LoneG.
**/
struct TotemTriggerCounters
{
  unsigned char type;
  unsigned int event_num, bunch_num, src_id;
  unsigned int orbit_num;
  unsigned char revision_num;
  unsigned int run_num, trigger_num, inhibited_triggers_num, input_status_bits;
};

#endif
