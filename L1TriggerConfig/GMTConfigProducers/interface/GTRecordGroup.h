#ifndef CondTools_L1Trigger_GTRecordGroup_h
#define CondTools_L1Trigger_GTRecordGroup_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     T1GlobalTriggerGroup
// 
/*
 Description: Overriding definitions for RecordHelpers to correctly
              deal with the global trigger / global muon trigger
	      OMDS database layouts. Designate record types as belonging
	      to this group via

	      RH_ASSIGN_GROUP(RecordType, TGlobalTriggerGroup) 

	      in namespace scope.
*/


/** A tag class for the GT/GMT schema and its specialties. */
#include "L1TriggerConfig/GMTConfigProducers/interface/RecordHelper.h"
class TGlobalTriggerGroup;

/* Keep the default behaviour for most types. */
template <typename TOutput,  
	  typename TCType> 
struct GroupFieldHandler<TOutput, TGlobalTriggerGroup, TCType> { 
  typedef FieldHandler<TOutput, TCType, TCType> Type;
 };

/* But bool field are stored as '0'/'1' chars in the database*/
template <typename TOutput> 
struct GroupFieldHandler<TOutput, TGlobalTriggerGroup, bool> { 
  typedef ASCIIBoolFieldHandler<TOutput, '0'> Type;
 };

/* But int field are stored as short in the database*/
template <typename TOutput> 
struct GroupFieldHandler<TOutput, TGlobalTriggerGroup, int> { 
  typedef FieldHandler<TOutput,  int, short> Type;
 };

/* But unsigned int field are stored as short in the database*/
template <typename TOutput> 
struct GroupFieldHandler<TOutput, TGlobalTriggerGroup, unsigned int> { 
  typedef FieldHandler<TOutput,  unsigned int, short> Type;
 };

#endif // CondTools_L1Trigger_GTRecordGroup_h
