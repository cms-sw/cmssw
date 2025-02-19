#include "EventFilter/CSCRawToDigi/interface/CSCBadCFEBWord.h"
#include<iostream>

std::ostream & operator<<(std::ostream & os, const CSCBadCFEBWord & word) 
{
  if(!word.check()) os << "Even the Bad CFEB word is bad!  Sheesh!" << std::endl;
  else 
    {
      switch(word.code_)
	{
	case 1:
	  os << "CFEB: SCA Capacitors Full  block " << word.word2_ 
	     << " FIFO1 count (4-bit) " << word.word1_ << std::endl;
	  break;
	case 2:
	  os << "CFEB: FPGA FIFO Full  FIFO3 count (4-bit) " << word.word2_ 
	     << " FIFO1 count (4-bit) " << word.word1_ << std::endl;
	  break;
	case 5:
	  os << "CFEB: DMB FIFO Full " << std::endl;
	  break;
	case 6:
	  os << "CFEB: DMB FPGA FIFO Full GFIFO count (4-bit)" << word.word2_ 
	     << " LFIFO count (4-bit) " << word.word1_ << std::endl;
	  break;
	default:
	  os << "Undefined CFEB error" << std::endl;
	  break;
	}
    }
  return os;
}

