/** \file
 *
 *  $Date: 2012/01/02 15:35:28 $
 *  $Revision: 1.5 $
 *  \author  M. Zanetti - INFN Padova 
 * FRC 140906
 */

#include <EventFilter/DTRawToDigi/plugins/DTROS8Unpacker.h>
#include <CondFormats/DTObjects/interface/DTReadOutMapping.h>
#include <FWCore/Utilities/interface/Exception.h>
#include <DataFormats/MuonDetId/interface/DTWireId.h> 

#include <iostream>
#include <map>


using namespace std;
using namespace edm;
using namespace cms;

void DTROS8Unpacker::interpretRawData(const unsigned int* index, int datasize,
				      int dduID,
				      edm::ESHandle<DTReadOutMapping>& mapping, 
				      std::auto_ptr<DTDigiCollection>& product,
                                      std::auto_ptr<DTLocalTriggerCollection>& product2,
				      uint16_t rosList) {
 

  /// CopyAndPaste from P. Ronchese unpacker
  const int wordLength = 4;
  int numberOfWords = datasize / wordLength;
  int robID = 0;
  int rosID = 0;

  map<int,int> hitOrder;

  // Loop over the ROS8 words
  for ( int i = 1; i < numberOfWords; i++ ) {

    // The word
    uint32_t word = index[i];

    // The word type
    int type = ( word >> 28 ) & 0xF;

    // Event Header 
    if ( type == 15 ) {
      robID =   word        & 0x7;
      rosID = ( word >> 3 ) & 0xFF;
    } 

    // TDC Measurement
    else if ( type >= 4 && type <= 5 ) {
      
      int tdcID = ( word >> 24 ) & 0xF;
      int tdcChannel = ( word >> 19 ) & 0x1F;

      int channelIndex = robID << 7 | tdcID << 5 | tdcChannel;
      hitOrder[channelIndex]++;

      int tdcMeasurement =  word  & 0x7FFFF;
      tdcMeasurement >>= 2;


      try {

	// Check the ddu ID in the mapping been used
	dduID = pset.getUntrackedParameter<int>("dduID",730);

	// Map the RO channel to the DetId and wire
	DTWireId detId; 
	if ( ! mapping->readOutToGeometry(dduID, rosID, robID, tdcID, tdcChannel,detId)) {
	  if (pset.getUntrackedParameter<bool>("debugMode",false)) cout<<"[DTROS8Unpacker] "<<detId<<endl;
	  int wire = detId.wire();
	  
	  // Produce the digi
	  DTDigi digi(wire, tdcMeasurement, hitOrder[channelIndex]-1);
	  
	  // Commit to the event
	  product->insertDigi(detId.layerId(),digi);
	}
	else if (pset.getUntrackedParameter<bool>("debugMode",false)) 
	  cout<<"[DTROS8Unpacker] Missing wire!"<<endl;
      }

      catch (cms::Exception & e1) {
	cout<<"[DTUnpackingModule]: WARNING: Digi not build!"<<endl; 
	return;
      }
    }
    
  }
}
