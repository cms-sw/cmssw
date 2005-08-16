#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigi.h"
//
#include <iostream>
#include<vector>

using namespace std;
//using namespace raw;

// -----------------------------------------------------------------------------
// constructor
SiStripRawToDigi::SiStripRawToDigi( SiStripConnection& connections ) : 
  connections_(),
  verbosity_(3)
{
  if (verbosity_>1) cout << "[SiStripRawToDigi::SiStripRawToDigi] " 
			 << "constructing SiStripRawToDigi converter object..." << endl;
  connections_ = connections;
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
    
//     // some temporary debug...
//     vector<unsigned int> dets = digis.detIDs();
//     if (verbosity_>2) cout << "[SiStripRawToDigi::createDigis] " 
// 			   << "GET HERE! : StripDigiCollection::detIDs().size() = " 
// 			   << dets.size() << endl;
    
    /* real implementation here! */
    
  }
  catch ( string err ){
    cout << "SiStripRawToDigi::createFedBuffers] " 
	 << "Exception caught : " << err << endl;
  }
}
