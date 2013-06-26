// System
#include <fstream>
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
// #include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "Alignment/SurveyAnalysis/interface/SurveyDataReader.h"

using namespace std;
using namespace edm;

//__________________________________________________________________________________________________
void SurveyDataReader::readFile( const std::string& textFileName, const std::string& fileType, const TrackerTopology* tTopo) {
  
  std::ifstream myfile( textFileName.c_str() );
  if ( !myfile.is_open() ) 
    throw cms::Exception("FileAccess") << "Unable to open input text file for " << fileType.c_str();

  int nErrors = 0;
  align::ID m_detId = 0;
  int NINPUTS_align = 30;
  int NINPUTS_detId = 6;
  if (fileType == "TID") NINPUTS_detId++;

  std::vector<int> d_inputs;
  align::Scalars a_inputs;
  align::Scalars a_outputs;
  int itmpInput;
  float tmpInput; 

  while ( !myfile.eof() && myfile.good() )
	{
	  d_inputs.clear();
          a_inputs.clear();
          a_outputs.clear();

          if (fileType == "TIB") {
	    itmpInput = int(StripSubdetector::TIB) ; 
	  } else {
	    itmpInput = int(StripSubdetector::TID) ;
	  }

          d_inputs.push_back( itmpInput );

	  for ( int i=0; i<NINPUTS_detId; i++ )
		{
		  myfile >> itmpInput;
		  d_inputs.push_back( itmpInput );
		}
            
          // Calculate DetId(s)
          int ster = 0;  // if module is single-sided, take the module
                         // if double-sided get the glued module

	  if (fileType == "TID") {
	    
	    m_detId = tTopo->tidDetId(d_inputs[2], d_inputs[3], d_inputs[4], d_inputs[5], d_inputs[6], ster);
	  }
	  else if (fileType == "TIB") {
	     
	    m_detId = tTopo->tibDetId(d_inputs[2], d_inputs[3], d_inputs[4], d_inputs[5], d_inputs[6], ster);
	  }

          if (abs(int(m_detId) - int(d_inputs[1])) > 2) {  // Check DetId calculation ...
            std::cout << "ERROR : DetId - detector position mismatch! Found " << nErrors << std::endl;
            nErrors++;
	  }

          // std::cout << m_detId << " " << d_inputs[1] << std::endl;
	  // m_detId = d_inputs[1];
	  for ( int j=0; j<NINPUTS_align; j++ )
		{
		  myfile >> tmpInput;
		  a_inputs.push_back( tmpInput );
		}

	  // Check if read succeeded (otherwise, we are at eof)
	  if ( myfile.fail() ) break;

          a_outputs = convertToAlignableCoord( a_inputs );

	  theOriginalMap.insert( PairTypeOr( d_inputs, a_inputs ));
	  theMap.insert( PairType( m_detId, a_outputs ));
	 
	}

}
//__________________________________________________________________________________________________
align::Scalars 
SurveyDataReader::convertToAlignableCoord( const align::Scalars& align_params ) 
{
      align::Scalars align_outputs;

      // Convert to coordinates that TrackerAlignment can read in
     
      // Center of sensor 
      AlgebraicVector geomCent(3);
      AlgebraicVector surCent(3);
      for (int ii = 0; ii < 3; ii++) {
	geomCent[ii] = align_params[ii];
	surCent[ii] = align_params[ii+15];
      }
      surCent -= geomCent;
                 
      align_outputs.push_back( surCent[0] ); 
      align_outputs.push_back( surCent[1] );   
      align_outputs.push_back( surCent[2] );   
      
      // Rotation matrices
      for (int ii = 3; ii < 12; ii++) { 
	  align_outputs.push_back( align_params[ii] );
      }
      for (int ii = 18; ii < 27; ii++) { 
	  align_outputs.push_back( align_params[ii] );
      } 
           
      return align_outputs; 
}
