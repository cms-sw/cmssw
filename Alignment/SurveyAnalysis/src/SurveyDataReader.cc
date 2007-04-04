// System
#include <fstream>
#include <iostream>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "Alignment/SurveyAnalysis/interface/SurveyDataReader.h"

using namespace std;
using namespace edm;

//__________________________________________________________________________________________________
void SurveyDataReader::readFile( const std::string& textFileName ,const std::string& fileType )
{
  
  std::ifstream myfile( textFileName.c_str() );
  if ( !myfile.is_open() ) 
    throw cms::Exception("FileAccess") << "Unable to open input text file for " << fileType.c_str();

  DetIdType m_detId;
  int NINPUTS_align = 30;
  int NINPUTS_detId = 5;
  if (fileType == "TID") NINPUTS_detId++;

  std::vector<int> d_inputs;
  std::vector<float> a_inputs;
  std::vector<float> a_outputs;
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
            TIDDetId *myTDI; 
            // Riccardo: disks 1 and 2 in the backward part swapped in assembly
            if (d_inputs[4] == 0 && d_inputs[2] == 1) {
	      myTDI = new TIDDetId( d_inputs[1], 2, d_inputs[3], 0, d_inputs[5], ster);
	    } else if (d_inputs[4] == 0 && d_inputs[2] == 2) {
	      myTDI = new TIDDetId( d_inputs[1], 1, d_inputs[3], 0, d_inputs[5], ster);
	    } else {
	      myTDI = new TIDDetId( d_inputs[1], d_inputs[2], d_inputs[3], d_inputs[4], d_inputs[5], ster);
	    }
	    m_detId = myTDI->rawId();
	  }
	  else if (fileType == "TIB") {
	    TIBDetId *myTBI = new TIBDetId( d_inputs[1], d_inputs[2], d_inputs[3], d_inputs[4], d_inputs[5], ster); 
	    m_detId = myTBI->rawId();
	  }

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
std::vector<float> 
SurveyDataReader::convertToAlignableCoord( std::vector<float> align_params ) 
{
      std::vector<float> align_outputs;

      // Convert to coordinates that TrackerAlignment can read in
     
      // Center of sensor 
      AlgebraicVector geomCent(3);
      AlgebraicVector surCent(3);
      for (int ii = 0; ii < 3; ii++) {
	geomCent[ii] = align_params[ii];
	surCent[ii] = align_params[ii+15];
      }
      surCent -= geomCent;
                
      // Rotation matrix
      const CLHEP::Hep3Vector theBase1i(align_params[6],align_params[7],align_params[8]);
      const CLHEP::Hep3Vector theBase1j(align_params[9],align_params[10],align_params[11]);
      const CLHEP::Hep3Vector theBase1k(align_params[3],align_params[4],align_params[5]);
      const CLHEP::Hep3Vector theBase2i(align_params[21],align_params[22],align_params[23]);
      const CLHEP::Hep3Vector theBase2j(align_params[24],align_params[25],align_params[26]);
      const CLHEP::Hep3Vector theBase2k(align_params[18],align_params[19],align_params[20]);

      CLHEP::HepRotation baseChngMat1( theBase1i, theBase1j, theBase1k );
      CLHEP::HepRotation baseChngMat2( theBase2i, theBase2j, theBase2k );
      CLHEP::HepRotation theRotMatrix = baseChngMat2 * ( baseChngMat1.invert() );
            
      // Euler angles
      float thisPhi = theRotMatrix.getPhi();
      float thisTheta = theRotMatrix.getTheta();
      float thisPsi = theRotMatrix.getPsi();
     
      align_outputs.push_back( surCent[0] ); 
      align_outputs.push_back( surCent[1] );   
      align_outputs.push_back( surCent[2] );   
      // align_outputs.push_back( thisPhi );      
      // align_outputs.push_back( thisTheta );    
      // align_outputs.push_back( thisPsi );
      
      for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) { 
	  align_outputs.push_back( theRotMatrix[ii][jj] );
	}
      } 
      
      LogDebug("WriteValues") << " Dx = " << surCent[0]
			      << " Dy = " << surCent[1] << " Dz = " << surCent[2] << "\n Phi = " << thisPhi 
			      << " Theta = " << thisTheta << " Psi = " << thisPsi;
      
      return align_outputs; 
}
