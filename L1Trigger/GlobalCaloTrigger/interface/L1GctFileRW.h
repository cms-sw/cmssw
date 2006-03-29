//-------------------------------------------------
//
/**  \class L1GctFileRW
 *
 *   RCT file reader/writer
 * 
 *   $Date: 2006/03/29 13:01:34 $
 *   $Revision: 1.1 $
 *
 *   \author  Maria Hansen, University of Bristol
 */  
//   
//--------------------------------------------------
#ifndef L1GCTFILERW_H
#define L1GCTFILERW_H

#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

#include <fstream>
#include <vector>

using std::ifstream;
using std::ofstream;
using std::vector;
using std::string;

/*RCT Input File Format 
  Line 1: Crossing no as "Crossing x" (2)     
  Line 2: isoe0 isoe1 isoe2 isoe3 nonIsoe0 nonIsoe1 nonIso2 nonIso3 (8) 
  Line 3: RC0mip0 RC0mip1 RC1mip0 RC1mip1 RC2mip0 RC2mip1 RC3mip0 RC3mip1 RC4mip0 RC4mip1 
          RC5mip0 RC5mip1 RC6mip0 RC6mip1 (14)
  Line 4: RC0qt0 RCqt1 RC1qt0 RC1qt1 RC2qt0 RC2qt1 RC3qt0 RC3qt1 RC4qt0 RC4qt1 
          RC5qt0 RC5qt1 RC6qt0 RC6qt1 (14)
  Line 5: RC0reg0 RC0reg1 RC1reg0 RC1reg1 RC2reg0 RC2reg1 RC3reg0 RC3reg1 RC4reg0 RC4reg1
          RC5reg0 RC5reg1 RC6reg0 RC6reg1 (14)
  Line 6: HF0eta0 HF0eta1 HF0eta2 HF0eta3 HF1eta0 HF1eta1 HF1eta2 HF1eta3 (8)
  ...
  ... 
  */ 


class L1GctFileRW {
 public:
  L1GctFileRW(L1GlobalCaloTrigger* gct);  
  ~L1GctFileRW();

  ///
  /// open input file
  void openInputFile(char file[256]);
  ///
  /// open output file
  void openOutputFile(char file[256]);
  ///
  /// read next event and push data to Source Cards
  void readBX();
  ///
  /// return true if current event is valid (false if EOF reached!)
  bool dataValid() { return valid; }

 private:

  ///
  /// back pointer to the GCT
  L1GlobalCaloTrigger* theGct;
  ///
  /// input file
  ifstream file;
  /// 
  /// output file
  ofstream ofile;
  ///
  /// event is valid
  bool valid;

  // private data members to store data?

};

#endif

