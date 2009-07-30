#ifndef EVF_TRIGGER_GENERATOR_WITH_PAYLOAD
#define EVF_TRIGGER_GENERATOR_WITH_PAYLOAD

#include <rubuilder/evm/TriggerGenerator.h>
#include "EventFilter/FEDInterface/interface/GlobalEventNumber.h"

namespace evf{
  
    struct tbits{
      uint64_t ttr;
      uint64_t ta1;
      uint64_t ta2;
      tbits():ttr(0ULL),ta1(0ULL),ta2(0ULL){}
      tbits &operator=(tbits *b){ta2=b->ta2;ta1=b->ta1;ttr=b->ttr; return *this;}
      tbits &operator=(uint64_t b){ta2=b; return *this;}
      tbits &operator<<(uint32_t s)
      {
	uint32_t t = s/64;
	switch(t)
	  {
	case 0: ta2 = ta2<<s; break; 
	case 1: ta1 = ta2; ta2=0ULL; ta1 = (ta1 << (s%64)); break;
	case 2: ttr = ta2; ta2=0ULL; ta1=0ULL; ttr = (ttr <<(s%64)); break;
	default: ta1=0ULL; ta2=0ULL; ttr=0ULL;
	}
      return *this;
      }
    };

  class l1cond{
  public:

    l1cond(int32_t ps, uint32_t eventNumber, tbits* bp = 0) 
      : recordScheme(evtn::BST52_3BX), patternScheme(ps)
      {

	switch(patternScheme)
	  {
	  case -1:
	    t.ttr = (eventNumber % 2) ? bitPattern1 : bitPattern2;
	    t.ta1 = (eventNumber % 2) ? bitPattern2 : bitPattern1;
	    t.ta2 = (eventNumber % 2) ? bitPattern1 : bitPattern2;
	    break;
	  case -2:
	    t.ttr = bitPatternf;
	    t.ta1 = bitPatternf;
	    t.ta2 = bitPatternf;
	    break;
	  case 256:
	    t = bp;
	    break;
	  default:
	    t = bitPattern3;
	    t = t<<patternScheme;
	  }
	
      }

    tbits t;
    evtn::EvmRecordScheme recordScheme;

  private:

    static const uint64_t bitPattern0   = 0ULL;
    static const uint64_t bitPattern1   = 0x5555555555555555ULL;
    static const uint64_t bitPattern2   = 0xaaaaaaaaaaaaaaaaULL;
    static const uint64_t bitPattern3   = 1ULL;
    static const uint64_t bitPatternf   = 0xffffffffffffffffULL;
    int32_t patternScheme; // patternScheme: -2=allbits; -1=built-in patterns; [0,191]=set ONE specific bit; 256=use struct

  };

  class TriggerGeneratorWithPayload : public rubuilder::evm::TriggerGenerator
    {
    public:
      
      toolbox::mem::Reference *generate
	(
	 toolbox::mem::MemoryPoolFactory *poolFactory,
	 toolbox::mem::Pool              *pool,
	 I2O_TID                          initiatorAddress,
	 I2O_TID                          targetAddress,
	 uint32_t                         triggerSourceId,
	 U32                              eventNumber,
	 U32                              eventType,
	 l1cond                          *toSet,   
	 uint32_t                         orbit // orbit number, used also to calculate ls
	);

      void fillPayload(char *, uint32_t, uint32_t, uint32_t, l1cond *);
      
    private:
      typedef rubuilder::evm::TriggerGenerator Base;
      static const uint32_t FAKE_FIXED_BX = 0x123;

    };
}
#endif
