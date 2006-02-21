/** \file
 *
 *  $Date: 2006/02/14 17:09:18 $
 *  $Revision: 1.8 $
 *  \author  M. Zanetti - INFN Padova 
 */

#include <EventFilter/DTRawToDigi/src/DTROS8Unpacker.h>
#include <EventFilter/DTRawToDigi/interface/DTDDUWords.h>
#include <EventFilter/DTRawToDigi/src/DTROSErrorNotifier.h>
#include <EventFilter/DTRawToDigi/src/DTTDCErrorNotifier.h>
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
				      std::auto_ptr<DTDigiCollection>& product) {
 

  /// CopyAndPaste from P. Ronchese unpacker
  const int wordLength = 4;
  int numberOfWords = datasize / wordLength;
  int robID = 0;
  int rosID = 0;
  int eventID = 0;
  int bunchID = 0;

  map<int,int> hitOrder;

  // Loop over the ROS8 words
  for ( int i = 1; i < numberOfWords; i++ ) {

    // The word
    int word = index[i];

    // The word type
    int type = ( word >> 28 ) & 0xF;

    // Event Header 
    if ( type == 15 ) {
      robID =   word        & 0x7;
      rosID = ( word >> 3 ) & 0xFF;
    } 

    // TDC Header/Trailer
    else if ( type <= 3 ) {
      eventID = ( word >> 12 ) & 0xFFF;
      bunchID =   word &         0xFFF; 
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

	// temporary for the mapping

	// Commissioning
	//dduID = 5;
	// Sector Test
	dduID = 734;

	// Map the RO channel to the DetId and wire
	DTWireId detId = mapping->readOutToGeometry(dduID, rosID, robID, tdcID, tdcChannel);
	int wire = detId.wire();

//   	cout<<"ROS: "<<rosID<<" ROB: "<<robID<<" TDC: "<<tdcID<<" TDCChannel: "<<tdcChannel<<endl;
//   	cout<<"Wheel: "<<detId.wheel()
//   	    <<" Station: "<<detId.station()
//   	    <<" Sector: "<<detId.sector()<<endl;

	
	// Produce the digi
	DTDigi digi(wire, tdcMeasurement, hitOrder[channelIndex]-1);

	// Commit to the event
	product->insertDigi(detId.layerId(),digi);
      }

      catch (cms::Exception & e1) {
	cout<<"[DTUnpackingModule]: WARNING: Digi not build!"<<endl; 
	return;
      }
    }
    
  }
}
