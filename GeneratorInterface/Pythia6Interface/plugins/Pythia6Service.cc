#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <fstream> 
#include <cmath>
#include <string>
#include <set>

#include <boost/bind.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>


#include "Pythia6Service.h"
#include "Pythia6Declarations.h"
// #include "GeneratorInterface/Core/interface/ParameterCollector.h"

extern "C"
{
   void fioopn_( int* unit, const char* line, int length );
   void fiocls_( int* unit );
   void pyslha_( int*, int*, int* );
   static int call_pyslha(int mupda, int kforig = 0)
   {
      int iretrn = 0;
      pyslha_(&mupda, &kforig, &iretrn);
      return iretrn;
   }

}

using namespace gen;
using namespace edm;

Pythia6Service::Pythia6Service( const ParameterSet& ps )
   : fUnitSLHA(24)
{

/*
   ParameterCollector collector(ps.getParameter<edm::ParameterSet>("PythiaParameters"));

   fParamGeneral.clear();
   fParamCSA.clear();
   fParamSLHA.clear();
   
   fParamGeneral = std::vector<std::string>(collector.begin(), collector.end());
   fParamCSA     = std::vector<std::string>(collector.begin("CSAParameters"), collector.end());
   fParamSLHA    = std::vector<std::string>(collector.begin("SLHAParameters"), collector.end());
*/
     

   // Set PYTHIA parameters in a single ParameterSet
   //
   edm::ParameterSet pythia_params = 
      ps.getParameter<edm::ParameterSet>("PythiaParameters") ;
      
   // read and sort Pythia6 cards
   //
   std::vector<std::string> setNames =
      pythia_params.getParameter<std::vector<std::string> >("parameterSets");
      
   // std::vector<std::string>	paramLines;
   fParamGeneral.clear();
   fParamCSA.clear();
   fParamSLHA.clear();
   

   for(std::vector<std::string>::const_iterator iter = setNames.begin();
	                                        iter != setNames.end(); ++iter) 
   {
      std::vector<std::string> lines =
         pythia_params.getParameter< std::vector<std::string> >(*iter);

      for(std::vector<std::string>::const_iterator line = lines.begin();
		                                   line != lines.end(); ++line ) 
      {
         if (line->substr(0, 7) == "MRPY(1)")
	    throw cms::Exception("PythiaError") <<
	    "Attempted to set random number"
	    " using Pythia command 'MRPY(1)'."
	    " Please use the"
	    " RandomNumberGeneratorService." <<
	    std::endl;

	 if ( *iter == "CSAParameters" )
	 {
	    fParamCSA.push_back(*line);
	 }
	 else if ( *iter == "SLHAParameters" )
	 {
	    fParamSLHA.push_back(*line);
	 }
	 else
	 {
	    fParamGeneral.push_back(*line);
	 }
      }
   }

}

Pythia6Service::~Pythia6Service()
{
   fParamGeneral.clear();
   fParamCSA.clear();
   fParamSLHA.clear();
}

void Pythia6Service::setGeneralParams()
{
   // now pass general config cards 
   //
   for(std::vector<std::string>::const_iterator iter = fParamGeneral.begin();
	                                        iter != fParamGeneral.end(); ++iter)
   {
      if (!call_pygive(*iter))
         throw cms::Exception("PythiaError")
	 << "Pythia did not accept \""
	 << *iter << "\"." << std::endl;
   }
   
   return ;
}

void Pythia6Service::setCSAParams()
{
      
   txgive_init_();
   
   for(std::vector<std::string>::const_iterator iter = fParamCSA.begin();
	                                        iter != fParamCSA.end(); ++iter)
   {
      txgive_( iter->c_str(), iter->length() );
   }   
   
   return ;
}

void Pythia6Service::openSLHA( const char* file )
{

        std::ostringstream pyCard1 ;
        pyCard1 << "IMSS(21)=" << fUnitSLHA;
        call_pygive( pyCard1.str() );
        std::ostringstream pyCard2 ;
        pyCard2 << "IMSS(22)=" << fUnitSLHA;
        call_pygive( pyCard2.str() );

	fioopn_( &fUnitSLHA, file, strlen(file) );
	
	return;

}

void Pythia6Service::closeSLHA() 
{

   fiocls_(&fUnitSLHA);
   
   return;

}

void Pythia6Service::setSLHAParams()
{
   for (std::vector<std::string>::const_iterator iter = fParamSLHA.begin();
                                                 iter != fParamSLHA.end(); iter++ )
   {

      	if( iter->find( "SLHAFILE", 0 ) == std::string::npos ) continue;
	std::string::size_type start = iter->find_first_of( "=" ) + 1;
	std::string::size_type end = iter->length() - 1;
	std::string::size_type temp = iter->find_first_of( "'", start );
	if( temp != std::string::npos ) {
			start = temp + 1;
			end = iter->find_last_of( "'" ) - 1;
	} 
	start = iter->find_first_not_of( " ", start );
	end = iter->find_last_not_of( " ", end );
	//std::cout << " start, end = " << start << " " << end << std::endl;		
	std::string shortfile = iter->substr( start, end - start + 1 );
	FileInPath f1( shortfile );
	std::string file = f1.fullPath();

/*
	//
	// override what might have be given via the external config
	//
        std::ostringstream pyCard ;
        pyCard << "IMSS(21)=" << fUnitSLHA;
        call_pygive( pyCard.str() );
        pyCard << "IMSS(22)=" << fUnitSLHA;
        call_pygive( pyCard.str() );

	fioopn_( &fUnitSLHA, file.c_str(), file.length() );
*/

        openSLHA( file.c_str() );

   }
   
   return;
}

void Pythia6Service::setSLHAFromHeader( const std::vector<std::string> &lines )
{

	std::set<std::string> blocks;
	unsigned int model = 0, subModel = 0;

	const char *fname = std::tmpnam(NULL);
	std::ofstream file(fname, std::fstream::out | std::fstream::trunc);
	std::string block;
	for(std::vector<std::string>::const_iterator iter = lines.begin();
	    iter != lines.end(); ++iter) {
		file << *iter;

		std::string line = *iter;
		std::transform(line.begin(), line.end(),
		               line.begin(), (int(*)(int))std::toupper);
		std::string::size_type pos = line.find('#');
		if (pos != std::string::npos)
			line.resize(pos);

		if (line.empty())
			continue;

		if (!boost::algorithm::is_space()(line[0])) {
			std::vector<std::string> tokens;
			boost::split(tokens, line,
			             boost::algorithm::is_space(),
			             boost::token_compress_on);
			if (!tokens.size())
				continue;
			block.clear();
			if (tokens.size() < 2)
				continue;
			if (tokens[0] == "BLOCK") {
				block = tokens[1];
				blocks.insert(block);
				continue;
			}

			if (tokens[0] == "DECAY") {
				block = "DECAY";
				blocks.insert(block);
			}
		} else if (block == "MODSEL") {
			std::istringstream ss(line);
			ss >> model >> subModel;
		} else if (block == "SMINPUTS") {
			std::istringstream ss(line);
			int index;
			double value;
			ss >> index >> value;
			switch(index) {
			    case 1:
				pydat1_.paru[103 - 1] = 1.0 / value;
				break;
			    case 2:
				pydat1_.paru[105 - 1] = value;
				break;
			    case 4:
				pydat2_.pmas[0][23 - 1] = value;
				break;
			    case 6:
				pydat2_.pmas[0][6 - 1] = value;
				break;
			    case 7:
				pydat2_.pmas[0][15 - 1] = value;
				break;
			}
		}
	}
	file.close();

	if (blocks.count("SMINPUTS"))
		pydat1_.paru[102 - 1] = 0.5 - std::sqrt(0.25 -
			pydat1_.paru[0] * M_SQRT1_2 *
			pydat1_.paru[103 - 1] /	pydat1_.paru[105 - 1] /
			(pydat2_.pmas[0][23 - 1] * pydat2_.pmas[0][23 - 1]));

/*
	int unit = 24;
	fioopn_(&unit, fname, std::strlen(fname));
	std::remove(fname);

	call_pygive("IMSS(21)=24");
	call_pygive("IMSS(22)=24");
*/
	
	openSLHA( fname ) ;
	std::remove( fname );
	
	if (model ||
	    blocks.count("HIGMIX") ||
	    blocks.count("SBOTMIX") ||
	    blocks.count("STOPMIX") ||
	    blocks.count("STAUMIX") ||
	    blocks.count("AMIX") ||
	    blocks.count("NMIX") ||
	    blocks.count("UMIX") ||
	    blocks.count("VMIX"))
		call_pyslha(1);
	if (model ||
	    blocks.count("QNUMBERS") ||
	    blocks.count("PARTICLE") ||
	    blocks.count("MINPAR") ||
	    blocks.count("EXTPAR") ||
	    blocks.count("SMINPUTS") ||
	    blocks.count("SMINPUTS"))
		call_pyslha(0);
	if (blocks.count("MASS"))
		call_pyslha(5, 0);
	if (blocks.count("DECAY"))
		call_pyslha(2);

      return ;

}
