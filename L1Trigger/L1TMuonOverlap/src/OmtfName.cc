#include "L1Trigger/L1TMuonOverlap/interface/OmtfName.h"


namespace {
  template <typename T> int sgn(T val) { return (T(0) < val) - (val < T(0)); } 
}

OmtfName::OmtfName(unsigned int iProcesor, int endcap)
{
  int iproc = (iProcesor <=5 ) ? static_cast<int>(iProcesor) : -1; 
  int position = (abs(endcap)==1) ? endcap : 0; 
  theBoard = static_cast<Board>( sgn(position)*(iproc+1) );
}

OmtfName::OmtfName(const std::string & board) 
{
  if (board=="OMTFn1") theBoard =  OMTFn1;
  else if (board=="OMTFn2") theBoard =  OMTFn2;
  else if (board=="OMTFn3") theBoard =  OMTFn3;
  else if (board=="OMTFn4") theBoard =  OMTFn4;
  else if (board=="OMTFn5") theBoard =  OMTFn5;
  else if (board=="OMTFn6") theBoard =  OMTFn6;
  else if (board=="OMTFp1") theBoard =  OMTFp1;
  else if (board=="OMTFp2") theBoard =  OMTFp2;
  else if (board=="OMTFp3") theBoard =  OMTFp3;
  else if (board=="OMTFp4") theBoard =  OMTFp4;
  else if (board=="OMTFp5") theBoard =  OMTFp5;
  else if (board=="OMTFp6") theBoard =  OMTFp6;
  else  theBoard =  OMTFp6;
}

std::string OmtfName::name() const  
{
  switch (theBoard) {
    case (OMTFn1) : return "OMTFn1";
    case (OMTFn2) : return "OMTFn2";
    case (OMTFn3) : return "OMTFn3";
    case (OMTFn4) : return "OMTFn4";
    case (OMTFn5) : return "OMTFn5";
    case (OMTFn6) : return "OMTFn6";
    case (OMTFp1) : return "OMTFp1";
    case (OMTFp2) : return "OMTFp2";
    case (OMTFp3) : return "OMTFp3";
    case (OMTFp4) : return "OMTFp4";
    case (OMTFp5) : return "OMTFp5";
    case (OMTFp6) : return "OMTFp6";
    default: return "UNKNOWN";
  }
}

int OmtfName::position() const { return sgn(theBoard); } 

unsigned int OmtfName::processor() const { return abs(theBoard)-1; }

