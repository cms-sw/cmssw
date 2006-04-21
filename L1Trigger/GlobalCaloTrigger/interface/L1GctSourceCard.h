#ifndef L1GCTSOURCECARD_H_
#define L1GCTSOURCECARD_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"

#include <vector>
#include <bitset>
#include <fstream>

using std::vector;
using std::bitset;
using std::ifstream;
using std::ofstream;


/**
  * Represents a GCT Source Card
  * author: Jim Brooke
  * date: 20/2/2006
  * 
  *RCT Input File Format 
  *Line 1: Crossing no as "Crossing x" (2)     
  *Line 2: isoe0 isoe1 isoe2 isoe3 nonIsoe0 nonIsoe1 nonIso2 nonIso3 (8) 
  *Line 3: RC0mip0 RC0mip1 RC1mip0 RC1mip1 RC2mip0 RC2mip1 RC3mip0 RC3mip1 RC4mip0 RC4mip1 
  *        RC5mip0 RC5mip1 RC6mip0 RC6mip1 (14)
  *Line 4: RC0qt0 RCqt1 RC1qt0 RC1qt1 RC2qt0 RC2qt1 RC3qt0 RC3qt1 RC4qt0 RC4qt1 
  *        RC5qt0 RC5qt1 RC6qt0 RC6qt1 (14)
  *Line 5: RC0reg0 RC0reg1 RC1reg0 RC1reg1 RC2reg0 RC2reg1 RC3reg0 RC3reg1 RC4reg0 RC4reg1
  *        RC5reg0 RC5reg1 RC6reg0 RC6reg1 (14)
  *Line 6: HF0eta0 HF0eta1 HF0eta2 HF0eta3 HF1eta0 HF1eta1 HF1eta2 HF1eta3 (8)
  *...
  *... 
  */ 


class L1GctSourceCard
{
public:
  L1GctSourceCard(); //(L1RctCrate* rc);
  ~L1GctSourceCard();
  
  ///
  /// open input file
  void openInputFile(char file[256]);
  ///
  /// read next event and push data to Source Cards
  void readBX();
  ///
  /// return true if current event is valid (false if EOF reached!)
  bool dataValid() { return valid; }
  ///
  /// close input file
  void closeInputFile();
  ///
  /// clear the buffers
  void reset();
  ///
  /// get the data from RCT/File/event/...
  void fetchInput();
  ///
  /// process the event
  void process();
  
  vector<L1GctEmCand> getIsoElectrons();
  vector<L1GctEmCand> getNonIsoElectrons();
  vector<L1GctRegion> getRegions();
  bitset<14> getMipBits();
  bitset<14> getQuietBits();

private:

  ///
  /// pointer to the RCT crate
  //L1RctCrate* rctCrate;
  ///
  /// file handle
  ifstream ifile;
  ///
  /// event data is valid	
  bool valid;

};

#endif /*L1GCTSOURCECARD_H_*/
