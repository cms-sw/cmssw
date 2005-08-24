#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigi.h"
//
#include <iostream>
#include<vector>

using namespace std;

// -----------------------------------------------------------------------------
// constructor
SiStripRawToDigi::SiStripRawToDigi( SiStripConnection& connections ) : 
  connections_(),
  verbosity_(3)
{
  if (verbosity_>1) cout << "[SiStripRawToDigi::SiStripRawToDigi] " 
			 << "constructing SiStripRawToDigi converter object..." << endl;
  connections_ = connections;


 
 //initialise vector to contain fed ids.
fedids.clear();fedids.reserve( 500 );

//fill vector containing fed ids
 vector<unsigned short> ifed;
 connections_.getConnectedFedNumbers(ifed);
 bool idquery = true;

 for (std::vector<unsigned short>::iterator itr = ifed.begin(); itr != ifed.end(); itr++) {
   for (std::vector<unsigned short>::iterator it = fedids.begin(); it != fedids.end(); it++) {
     
     if (*it == *itr) {idquery = false; break;}
     else {idquery = true;}
   }
   if (idquery == true) {fedids.push_back(*itr);}
 }

 //some debug

 if (verbosity_>2){
 map<unsigned short, cms::DetId> partitions;
 connections.getDetPartitions( partitions );
 std::cout << "number of feds: " << fedids.size() 
	    << ", number of partitions: " << partitions.size() << endl;
 }

  //initialise test vectors
 landau.clear(); landau.reserve( 30 );
 position.clear(); position.reserve( 128 );

  for (unsigned int i = 0; i < 30; i ++) {landau[i] = 0; position[i] = 0;}
for (unsigned int i = 30; i < 128; i ++) { position[i] = 0;}

}

// -----------------------------------------------------------------------------
// destructor
SiStripRawToDigi::~SiStripRawToDigi() {
  if (verbosity_>1) cout << "[SiStripRawToDigi::~SiStripRawToDigi] " 
			 << "destructing SiStripRawToDigi converter object..." << endl;
  /* anything here? */
}

// -----------------------------------------------------------------------------
// method to create a FEDRawDataCollection using a StripDigiCollection as input
void SiStripRawToDigi::createDigis( raw::FEDRawDataCollection& fed_buffers,
				    StripDigiCollection& digis ) { 
  if (verbosity_>2) cout << "[SiStripRawToDigi::createDigis] " 
			 << "creating StripDigiCollection using a FEDRawCollection as input..." << endl;
  try {
    
     // some temporary debug...
     vector<unsigned int> dets = digis.detIDs();
     if (verbosity_>2) cout << "[SiStripRawToDigi::createDigis] " 
 			   << "GET HERE! : StripDigiCollection::detIDs().size() = " 
			   << dets.size() << endl;
    
  cout << "SiStripRawToDigi::formatData(.)" << endl;
//// Creates Fed9UEvent object to iterate through the FED buffer
//// provided by DaqFEDRawData, extracts the ADC values from each FED
//// channel and creates digis for the Readout objects.

  if (verbosity_>2){ cout << "number of feds  = " << fedids.size() << ". Fed numbers stored (in order) are:" << endl;

   for (unsigned int i = 0; i < fedids.size(); i++) {
   cout << " number " << i + 1 << " " << fedids[i] << endl;
   }
  }

  for (std::vector<unsigned short>::iterator itr = fedids.begin(); itr != fedids.end(); itr++) {

   if (verbosity_>2) cout << "Extracting digis from fed number " << *itr << endl;

  raw::FEDRawData& FEDBuffer = fed_buffers.FEDData(static_cast<int>(*itr));
 
  //// get the data buffer (in I8 format), reinterpret as array in U32 format (as
  //// required by Fed9UEvent) and get buffer size (convert from units of I8 to U32)
  unsigned char* buffer = const_cast<unsigned char*>(FEDBuffer.data());
  unsigned long* buffer_u32 = reinterpret_cast<unsigned long*>( buffer );
  unsigned long size = (static_cast<unsigned long>(FEDBuffer.data_.size())) / 4; 

  if (verbosity_>2) cout << "SiStripRawToDigi::FEDBuffer size (32 bit words) = " << size << endl;

  //// get FED readout path (SLink/VME) from .orcarc (later from description)
  //// and remove VME header (if present) from start of FED buffer
  // static SimpleConfigurable<string> readout_path("SLINK", "SiStripDataFormatter:FedReadoutPath");

 if (size !=0) { //test loop

  if ( readoutPath == "VME" ) { 
    unsigned int shift = buffer_u32[11];
    unsigned int nchan = buffer_u32[13];
 
    unsigned int start = 10 + shift + nchan;
    size -= start; // recalculate buffer size (size after "start" pos)
  }   


 //// pass FED buffer into the Fed9UEvent object
  fedEvent_ = new Fed9U::Fed9UEvent(); //@@ temporary
  fedEvent_->Init(buffer_u32, description_, size); 

// check FED readout mode (SM, VR, PR, ZS) and extract digis using appropriate method
  //  try {

  //get readout mode from Fed9UEvent
  readoutMode = fed9UEvent()->getDaqMode();

     if ( readoutMode == 3 ) {scopeMode(digis, FEDBuffer, *itr ); }
     else if ( readoutMode == 2 ) {virginRaw(digis, FEDBuffer, *itr); } 
     else if ( readoutMode == 0 ) {processedRaw(digis,FEDBuffer, *itr); }
     else if ( readoutMode ==  1 ) {ZSMode(digis, FEDBuffer, *itr); } 

     else { } /* throw something! */  

 }// test loop


 }//end of loop over feds

  if (verbosity_>2) cout << "SiStripRawToDigi::Digis extracted from feds.  " << endl;

   //loop over all digis in StripDigiCollection
  if (1) { //coutV.testOut ) {

if (verbosity_>2) cout << "SiStripRawToDigi::Running test loop" << endl;

    vector<unsigned int> ids = digis.detIDs();
    short cntr = 0;

    for( std::vector<unsigned int>::iterator it = ids.begin(); it != ids.end(); it++ ) {
      StripDigiCollection::Range temp = digis.get(*it);
      for (StripDigiCollection::ContainerIterator idigi = temp.first; idigi != temp.second; idigi++) {
	if ( (*idigi).adc() ) {
	  position[ (*idigi).channel()%128 ]++; 
	  landau[ (*idigi).adc()<30 ? (*idigi).adc() : 29]++;
	  cntr++;
	  nDigis_++;
	}
      }
      
    }

   if ( cntr /* && coutV.debugOut*/ ) {cout << "SiStripRawToDigiFormatter::createDigis(...) : Extracting " << cntr << " digis from fed buffers" << endl;	
     }

   /*cout test vectors*/

 for (int i = 0; i < 128; i++) { cout << "position[" << i << "] =" << position[i] << endl;}
for (int i = 0; i < 30; i++) { cout << "landau[" << i << "] =" << landau[i] << endl;
 }

}
  


  }//end of 'try'

  catch ( string err ){
    cout << "SiStripRawToDigi::createFedBuffers]" 
	 << "Exception caught : " << err << endl;
  }
 delete fedEvent_; //@@ temporary

}

void SiStripRawToDigi::ZSMode(StripDigiCollection& digis, raw::FEDRawData& FEDBuffer, unsigned short fed_id) {

 if ( readoutMode != 1 ) {
    cout << "SiStripZSFormatter::createDigis(...) : WARNING : " 
	 << "readout != ZERO_SUPPRESSED" << endl;
  }
      
  // loop through FED channels
  for ( unsigned short ichan = 0; ichan < 96; ichan++ ) {

    uint32_t det_id = connections_.getDetId(fed_id,ichan).rawId();
	
	// iterate through payload data for this channel (skipping channel header of seven bytes)
  
	std::vector<StripDigi> channelDigis;
	channelDigis.clear();
	channelDigis.reserve (256);

	Fed9U::Fed9UEventIterator fed_iter = const_cast<Fed9U::Fed9UEventChannel&>(fed9UEvent()->channel( ichan )).getIterator();
	for (Fed9U::Fed9UEventIterator i = fed_iter+7; i.size() > 0; /**/) {
	  unsigned char first_strip = *i++; // first strip position of cluster
	  unsigned char width = *i++; // strip width of cluster 
	  for (unsigned short istr = 0; istr < width; istr++) {
	    channelDigis.push_back( StripDigi(static_cast<int>(first_strip+istr),static_cast<int>(*i)) );
*i++;				 
}
	  StripDigiCollection::ContainerIterator channelDigisStart = channelDigis.begin();
	  StripDigiCollection::ContainerIterator channelDigisFinish = channelDigis.end();
	  StripDigiCollection::Range stripDigiPtrs(channelDigisStart,channelDigisFinish);
	  digis.put( stripDigiPtrs,det_id);
	
	}
  }
}

void SiStripRawToDigi::scopeMode(StripDigiCollection& digis, raw::FEDRawData& FEDBuffer, unsigned short fed_id) {
}

void SiStripRawToDigi::virginRaw(StripDigiCollection& digis, raw::FEDRawData& FEDBuffer, unsigned short fed_id) {


////  coutV.debugOut << "SiStripRawFormatter::virginRaw(.)" << endl;
 nFeds_++; // counter for debug purposes

// // loop through FED channels

for ( unsigned short ichan = 0; ichan < fed9UEvent()->totalChannels(); ichan++ ) {
 // retrieve data samples from FED channel
    vector<unsigned short> adc = fed9UEvent()->channel( ichan ).getSamples();
    if ( adc.size() != 256 ) cout << "SiStripRawFormatter::virginRaw: ERROR: Number of ADC samples from FED buffer != 256" << endl;

    // retrieve DetUnit from FedToDetUnitMapper

    uint32_t det_id = connections_.getDetId(fed_id,ichan).rawId();
    vector<StripDigi> channelDigis; channelDigis.clear(); channelDigis.reserve( 256 );
    for ( unsigned short i = 0; i < adc.size(); i++ ) {
      int j = readoutOrder(i%128);
      (i/128) ? j=j*2+1 : j=j*2; // true=APV1, false=APV0
      channelDigis.push_back( StripDigi(j, adc[j]));
    }

      StripDigiCollection::ContainerIterator channelDigisStart = channelDigis.begin();
      StripDigiCollection::ContainerIterator channelDigisFinish = channelDigis.end();
      StripDigiCollection::Range stripDigiPtrs(channelDigisStart,channelDigisFinish);
      digis.put( stripDigiPtrs,det_id);

}

}


void SiStripRawToDigi::processedRaw(StripDigiCollection& digis, raw::FEDRawData& FEDBuffer, unsigned short fed_id) {
}
