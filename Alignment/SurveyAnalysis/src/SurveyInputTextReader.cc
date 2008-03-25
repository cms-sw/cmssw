// System
#include <fstream>
#include <iostream>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/SurveyAnalysis/interface/SurveyInputTextReader.h"

//__________________________________________________________________________________________________
void SurveyInputTextReader::readFile( const std::string& textFileName )
{

  std::ifstream myfile( textFileName.c_str() );
  if ( !myfile.is_open() )
        throw cms::Exception("FileAccess") << "Unable to open input text file";

	DetIdType m_detId;
	int m_alignObjId;
  TrackerAlignableId::UniqueId  m_uId;
  std::vector<float> m_inputs(NINPUTS);
  float tmpInput;
	
  while ( !myfile.eof() && myfile.good() )
	{
		m_inputs.clear();
		myfile >> m_detId >> m_alignObjId;
		m_uId.first = m_detId;
		m_uId.second = m_alignObjId;
		
		for ( int i=0; i<NINPUTS; i++ )
		{
			myfile >> tmpInput;
			m_inputs.push_back( tmpInput );
		}
		
		// Check if read succeeded (otherwise, we are at eof)
		if ( myfile.fail() ) break;
		
		theMap.insert( PairType( m_uId, m_inputs));
	}
}
