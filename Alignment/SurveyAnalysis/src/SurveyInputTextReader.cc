// System

#include <fstream>

#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/SurveyAnalysis/interface/SurveyInputTextReader.h"

//__________________________________________________________________________________________________
void SurveyInputTextReader::readFile( const std::string& textFileName )
{

  std::ifstream myfile( textFileName.c_str() );
  if ( !myfile.is_open() )
        throw cms::Exception("FileAccess") << "Unable to open input text file";

  while ( !myfile.eof() && myfile.good() )
    {
      align::Scalars m_inputs;
		
      UniqueId  m_uId;
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
	m_uId.second = AlignableObjectId::stringToId(structure.c_str());
      }
      else{
	myfile >> m_uId.first;

	for ( int i=0; i<NINPUTS; i++ )
	  {
	    float tmpInput;
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
