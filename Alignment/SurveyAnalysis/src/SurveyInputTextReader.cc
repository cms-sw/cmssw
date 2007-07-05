// System
#include <string>
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
	int m_alignObjId = 0; //initial value 
	AlignableObjectId  alignObjId;
  TrackerAlignableId::UniqueId  m_uId;
  std::vector<float> m_inputs(NINPUTS);
  float tmpInput;
	
  while ( !myfile.eof() && myfile.good() )
	{
		m_inputs.clear();
		
		char firstchar;
		firstchar = myfile.peek();

		if(firstchar == '#'){
			std::string line;
			getline(myfile,line);
		}
		else if (firstchar == '!'){
			std::string firststring;
			std::string structure; 
			myfile >> firststring >> structure;
			std::string endofline;
			getline(myfile,endofline);
			m_alignObjId = alignObjId.nameToType(structure);
			if (m_alignObjId == 0) std::cout << "=============> invalid Object ID" << std::endl;
		}
		else{
			myfile >> m_detId;
			m_uId.first = m_detId;
			m_uId.second = m_alignObjId;

			for ( int i=0; i<NINPUTS; i++ )
			{
				myfile >> tmpInput;
				m_inputs.push_back( tmpInput );
			}
			std::string endofline;
			getline(myfile,endofline);
			theMap.insert( PairType( m_uId, m_inputs));			
			
			// Check if read succeeded (otherwise, we are at eof)
			if ( myfile.fail() ) break;
			

		}
	}
}
