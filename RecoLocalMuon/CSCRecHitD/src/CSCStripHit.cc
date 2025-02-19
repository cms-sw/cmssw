#include "RecoLocalMuon/CSCRecHitD/src/CSCStripHit.h"
#include <iostream>

CSCStripHit::CSCStripHit() :
  theDetId(),
  theStripHitPosition(),
  theStripHitTmax(),     
  theStrips(),
  theStripHitADCs(),
  theStripHitRawADCs(),
  theConsecutiveStrips(),
  theClosestMaximum(),
  theDeadStrip()
{

/// Extract the lower byte for strip number
theStripsLowBits.clear();
for(int i=0; i<(int)theStrips.size(); i++){
        theStripsLowBits.push_back(theStrips[i] & 0x000000FF);
	}
/// Extract the middle byte for L1A phase
theStripsHighBits.clear();
for(int i=0; i<(int)theStrips.size(); i++){
        theStripsHighBits.push_back(theStrips[i] & 0x0000FF00);
	}
	
}

CSCStripHit::CSCStripHit( const CSCDetId& id, 
                          const float& sHitPos, 
                          const int& tmax, 
                          const ChannelContainer& strips, 
                          const StripHitADCContainer& s_adc,
                          const StripHitADCContainer& s_adcRaw,
			  const int& numberOfConsecutiveStrips,
                          const int& closestMaximum,
                          const short int & deadStrip) :
  theDetId( id ), 
  theStripHitPosition( sHitPos ),
  theStripHitTmax( tmax ),
  theStrips( strips ),
  theStripHitADCs( s_adc ),
  theStripHitRawADCs( s_adcRaw ),
  theConsecutiveStrips(numberOfConsecutiveStrips),
  theClosestMaximum(closestMaximum),
  theDeadStrip(deadStrip)
{

/// Extract the 2 lowest bytes for strip number
theStripsLowBits.clear();
for(int i=0; i<(int)theStrips.size(); i++){
        theStripsLowBits.push_back(theStrips[i] & 0x000000FF);
	}
/// Extract the 2 highest bytes for L1A phase
theStripsHighBits.clear();
for(int i=0; i<(int)theStrips.size(); i++){
        theStripsHighBits.push_back(theStrips[i] & 0x0000FF00);
	}
	
}

CSCStripHit::~CSCStripHit() {}


/// Debug
void
CSCStripHit::print() const {
  std::cout << "CSCStripHit in CSC Detector: " << std::dec << cscDetId() << std::endl;
  std::cout << "  sHitPos: " << sHitPos() << std::endl;
  std::cout << "  TMAX: " << tmax() << std::endl;
  std::cout << "  STRIPS: ";
  for (int i=0; i<(int)strips().size(); i++) {std::cout << std::dec << strips()[i] 
       << " (" << "HEX: " << std::hex << strips()[i] << ")" << " ";}
  std::cout << std::endl;

/// L1A  
  std::cout << "  L1APhase: ";
  for (int i=0; i<(int)stripsl1a().size(); i++) {
       //uint16_t L1ABitSet=(strips()[i] & 0xFF00);
       //std::cout << std::hex << (stripsl1a()[i] >> 15)<< " ";
       
       std::cout << "|";
       for (int k=0; k<8 ; k++){ 
       std::cout << ((stripsl1a()[i] >> (15-k)) & 0x1) << " ";}
       std::cout << "| ";       
       }           
  std::cout << std::endl;
  
  std::cout << "  S_ADC: ";
  for (int i=0; i<(float)s_adc().size(); i++) {std::cout << s_adc()[i] << " ";}
  std::cout << std::endl;
  std::cout << "  S_ADC_RAW: ";
  for (int i=0; i<(float)s_adcRaw().size(); i++) {std::cout << s_adcRaw()[i] << " ";}
  std::cout << std::endl;
}
