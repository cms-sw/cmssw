#include "EventFilter/SiStripRawToDigi/interface/SiStripDigiToRaw.h"
//
#include <iostream>
#include<vector>

using namespace std;
//using namespace raw;

// -----------------------------------------------------------------------------
// constructor
SiStripDigiToRaw::SiStripDigiToRaw( SiStripConnection& connections ) : 
  connections_(),
  verbosity_(3)
{
  if (verbosity_>1) cout << "[SiStripDigiToRaw::SiStripDigiToRaw] " 
			 << "constructing SiStripDigiToRaw converter object..." << endl;
  connections_ = connections;
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

    /* real implementation here! */

  }
  catch ( string err ){
    cout << "SiStripDigiToRaw::createFedBuffers] " 
	 << "Exception caught : " << err << endl;
  }
}
