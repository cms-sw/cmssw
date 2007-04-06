// [Note: code is taken from Tutorial:
//          [CMSSW]/UserCode/SamvelKhalatyan/Tutorial/Clusters
//        with minor changes]

// Author : Samvel Khalatyan (samvel at fnal dot gov)
// Created: 12/05/06
// Licence: GPL

#include "AnalysisExamples/SiStripDetectorPerformance/interface/GetSiStripClusterXTalk.h"

double
  extra::getSiStripClusterXTalk( const std::vector<uint16_t>
                                   &roSTRIP_AMPLITUDES,
                                 const uint16_t                 &rnFIRST_STRIP,
                                 const std::vector<SiStripDigi> &roDIGIS,
                                 const GetClusterXTalk &roGetClusterXTalk) {

  // Given value is used to separate non-physical values
  // [Example: cluster with empty amplitudes vector]
  double dClusterXTalk = -99;

  // XTalk should be calculated only for 1 and 3 strips clusters: read notes
  // for interface
  switch( roSTRIP_AMPLITUDES.size()) {
    case 1: {
      // One non-zero strip cluster - "left" is undefined: check for Digi that 
      // correspond to found strip
      std::vector<SiStripDigi>::const_iterator oITER( roDIGIS.begin());
      for( ;
	   oITER != roDIGIS.end() && oITER->strip() != roSTRIP_AMPLITUDES[0];
	   ++oITER) {}

      // Check if Digi for given cluster strip was found
      if( oITER != roDIGIS.end()) {

	// Check if previous neighbouring strip exists: here we combine 
	// Declaration and Initialization in the same statement
	std::vector<SiStripDigi>::const_iterator oITER_PREV = 
	  ( oITER != roDIGIS.begin() &&
	    ( oITER->strip() - 1) == ( oITER - 1)->strip()) ?
	  oITER - 1 : 
	  roDIGIS.end();

	// Check if next neighbouring strip exists: here we combine Declaration 
	// and Initialization in the same statement
	std::vector<SiStripDigi>::const_iterator oITER_NEXT = 
	  oITER != roDIGIS.end() &&
	    oITER != ( roDIGIS.end() - 1) &&
	    ( oITER->strip() + 1) == ( oITER + 1)->strip() ? 
	  oITER + 1 : 
	  roDIGIS.end();

        // Now check if both neighbouring digis exist
	if( oITER_PREV != roDIGIS.end() && oITER_NEXT != roDIGIS.end()) {
	  // Both Digis are specified: Now Pick the one with max amplitude
	  dClusterXTalk = roGetClusterXTalk( oITER_PREV->adc(),
	                                     roSTRIP_AMPLITUDES[0],
				             oITER_NEXT->adc());
	} else {
	  // PREV and NEXT iterators point to the end of DIGIs vector. 
	  // Consequently it is assumed there are no neighbouring digis at all
	  // for given cluster. ClusterEta is forsed to be equal to 1 in order 
	  // to separate such kind of situations from the real, calculated 
	  // values - very useful in analysis
	  dClusterXTalk = 1;
	} // End check if any neighbouring digi is specified
      } else {
	// Digi for given Clusters strip was not found. ClusterEta is forsed
	// to be equal to 0 in order to separate such kind of situations from
	// the real, calculated values - very useful in analysis
	dClusterXTalk = 0;
      } // end check if Digi for given cluster strip was found
    }
    case 3: {
      // Pure 3 strips cluster
      dClusterXTalk = roGetClusterXTalk( roSTRIP_AMPLITUDES[0],
                                         roSTRIP_AMPLITUDES[1],
				         roSTRIP_AMPLITUDES[2]);
    }
    default:
      break;
  }

  return dClusterXTalk;
}
