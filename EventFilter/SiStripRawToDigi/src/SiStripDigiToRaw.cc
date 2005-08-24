#include "EventFilter/SiStripRawToDigi/interface/SiStripDigiToRaw.h"
//
#include <iostream>
#include<vector>

using namespace std;

// -----------------------------------------------------------------------------
// constructor
SiStripDigiToRaw::SiStripDigiToRaw( SiStripConnection& connections ) : 
  connections_(),
  verbosity_(3)
{
  if (verbosity_>1) cout << "[SiStripDigiToRaw::SiStripDigiToRaw] " 
			 << "constructing SiStripDigiToRaw converter object..." << endl;
  connections_ = connections;

  //initialise vector to contain fed ids
  fedids.clear(); fedids.reserve( 500 );

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

 //ouputs the fed ids registered in the map
 //cout << "connected feds:" << endl;
 //for (std::vector<unsigned short>::iterator it = fedids.begin(); it != fedids.end(); it++) {
 //  cout << *it << endl;
 //}
}

// -----------------------------------------------------------------------------
// destructor
SiStripDigiToRaw::~SiStripDigiToRaw() {
  if (verbosity_>1) cout << "[SiStripDigiToRaw::~SiStripDigiToRaw] " 
			 << "destructing SiStripDigiToRaw converter object..." << endl;
  /* anything here? */
}

// -----------------------------------------------------------------------------
// method to create a FEDRawDataCollection using a StripDigiCollection as input
void SiStripDigiToRaw::createFedBuffers( StripDigiCollection& digis,
					 raw::FEDRawDataCollection& fed_buffers ) {
  if (verbosity_>2) cout << "[SiStripDigiToRaw::createFedBuffers] " 
			 << "creating FEDRawCollection using a StripDigiCollection as input..." << endl;
  try {

    // some temporary debug...
    vector<unsigned int> dets = digis.detIDs();
    if (verbosity_>2) cout << "[SiStripDigiToRaw::createFedBuffers] " 
			   << "GET HERE! : StripDigiCollection::detIDs().size() = " 
			   << dets.size() << endl;
   

 const unsigned short chans_per_fed = 8 * 12;
 unsigned short strips_per_fed = 96 * 256; // channels * strips/channel

 //loop through fed ids
 for (std::vector<unsigned short>::iterator itr = fedids.begin(); itr != fedids.end(); itr++) {

    cout << "Building FED Buffer for fed id " << *itr << endl;

 // define container for ADC values (stored in raw-like mode, ie, 1 value/strip)
    vector<unsigned short> data_buffer(strips_per_fed,0);

    //loop through fed channels
    for (unsigned short ichan = 0; ichan < chans_per_fed; ichan++) {

    //retrieve det id from fed channel (using map). Then loop over corresponding digis.
      
      SiStripConnection::DetPair detpair(0,0);
      connections_.getDetPair(*itr, ichan, detpair);
      
      if (detpair.first != 0) {
	
	unsigned short det_id = detpair.second;
	
	StripDigiCollection::Range my_digis = digis.get(det_id);
	
	for (StripDigiCollection::ContainerIterator it = my_digis.first; it != my_digis.second; it++) {
	  
	  //calculate strip position (within scope of FED) of digi
	  
	  short strip = ichan*256 + (*it).strip()%256;
	  
	  if ( strip >= strips_per_fed ) {
	    cout << "SiStripDataFormatter::formatData(.): ERROR: strip >= strips_per_fed" << endl;
	  }
	  
	  //check if buffer has already been filled with digi ADC value. if not or if filled with different value, fill it.
	  
	  if ( data_buffer[strip] ) { // if yes, cross-check values
	    if ( data_buffer[strip] != (*it).adc() ) {
	      /*std::stringstream os; os*/ cout << "SiStripDataFormatter::formatData(.): " 
						<< "WARNING: Incompatible ADC values in buffer: "
						<< "FED id: " << *itr << ", FED channel: " << ichan
						<< ", detector strip: " << (*it).strip() 
						<< ", FED strip: " << strip
						<< ", ADC value: " << (*it).adc()
						<< ", data_buffer["<<strip<<"]: " << data_buffer[strip];
	      // cout << os.str() << endl;
	  }
	  } else { // if no, update buffer with digi ADC value
	    data_buffer[strip] = (*it).adc(); 
	  }
	}
      

    // instantiate appropriate buffer creator object depending on readout mode
    Fed9U::Fed9UBufferCreator* creator = 0;
    if ( readoutMode == "SCOPE_MODE" ) {
      throw string("WARNING : Fed9UBufferCreatorScopeMode not implemented yet!");
    } else if ( readoutMode == "VIRGIN_RAW" ) {
      creator = new Fed9U::Fed9UBufferCreatorRaw();
    } else if ( readoutMode == "PROCESSED_RAW" ) {
      creator = new Fed9U::Fed9UBufferCreatorProcRaw();
    } else if ( readoutMode == "ZERO_SUPPRESSED" ) {
      creator = new Fed9U::Fed9UBufferCreatorZS();
    } else {
      cout << "WARNING : UNKNOWN readout mode" << endl;
    }
    
    // generate FED buffer and pass to Daq
    Fed9U::Fed9UBufferGenerator generator( creator );
    generator.generateFed9UBuffer( data_buffer );

    raw::FEDRawData& fedrawdata = fed_buffers.FEDData( *itr ); 
    // calculate size of FED buffer in units of bytes (unsigned char)
    int nbytes = generator.getBufferSize() * 4;
    // resize (public) "data_" member of struct FEDRawData
    (fedrawdata.data_).resize( nbytes );
    // copy FED buffer to struct FEDRawData using Fed9UBufferGenerator
    unsigned char* chars = const_cast<unsigned char*>( fedrawdata.data() );
    unsigned int* ints = reinterpret_cast<unsigned int*>( chars );
    generator.getBuffer( ints );

      }// if nulldetpair == detpair
    }//loop over channels
    } //loop over feds
 } //loop over try
  
 catch ( string err ) {
    cout << "SiStripDigiToRaw::createFedBuffers] " 
	 << "Exception caught : " << err << endl;
 }
}
