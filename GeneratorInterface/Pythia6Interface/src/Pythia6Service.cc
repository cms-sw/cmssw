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
#include <boost/filesystem.hpp>

#include "CLHEP/Random/RandomEngine.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h"
#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Declarations.h"
// #include "GeneratorInterface/Core/interface/ParameterCollector.h"

// This will force the symbols below to be kept, even in the case pythia6
// is an archive library.
//extern "C" void pyexec_(void);
extern "C" void pyedit_(void);
__attribute__((visibility("hidden"))) void dummy()
{
  using namespace gen;
  int dummy = 0;
  double dummy2 = 0;
  char * dummy3 = 0;
  pyexec_();
  pystat_(0);
  pyjoin_(dummy, &dummy);
  py1ent_(dummy, dummy, dummy2, dummy2, dummy2);
  pygive_(dummy3, dummy);
  pycomp_(dummy);
  pylist_(0);
  pyevnt_();
  pyedit_();
}

extern "C"
{
   void fioopn_( int* unit, const char* line, int length );
   void fioopnw_( int* unit, const char* line, int length );
   void fiocls_( int* unit );
   void pyslha_( int*, int*, int* );
   static int call_pyslha(int mupda, int kforig = 0)
   {
      int iretrn = 0;
      pyslha_(&mupda, &kforig, &iretrn);
      return iretrn;
   }

   void pyupda_(int*, int*);
   static void call_pyupda( int opt, int iunit ){ 
     pyupda_( &opt, &iunit ); 
   }

   double gen::pyr_(int *idummy)
   {
      // getInstance will throw if no one used enter/leave
      // or this is the wrong caller class, like e.g. Herwig6Instance
      Pythia6Service* service = FortranInstance::getInstance<Pythia6Service>();
      return service->fRandomEngine->flat(); 
   }
}

using namespace gen;
using namespace edm;

Pythia6Service* Pythia6Service::fPythia6Owner = 0;

Pythia6Service::Pythia6Service()
  : fRandomEngine(nullptr), fUnitSLHA(24), fUnitPYUPDA(25)
{
}

Pythia6Service::Pythia6Service( const ParameterSet& ps )
  : fRandomEngine(nullptr), fUnitSLHA(24), fUnitPYUPDA(25)
{
   if (fPythia6Owner)
      throw cms::Exception("PythiaError") <<
	    "Two Pythia6Service instances claiming Pythia6 ownership." <<
	    std::endl;

   fPythia6Owner = this;

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
   fParamPYUPDA.clear();
   

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
	 else if ( *iter == "PYUPDAParameters" )
	 {
	    fParamPYUPDA.push_back(*line);
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
   if (fPythia6Owner == this)
      fPythia6Owner = 0;

   fParamGeneral.clear();
   fParamCSA.clear();
   fParamSLHA.clear();
   fParamPYUPDA.clear();
}

void Pythia6Service::enter()
{
   FortranInstance::enter();

   if (!fPythia6Owner) {
     edm::LogInfo("Generator|Pythia6Interface") <<
          "gen::Pythia6Service is going to initialise Pythia, as no other "
          "instace has done so yet, and Pythia service routines have been "
          "requested by a dummy instance." << std::endl;

     call_pygive("MSTU(12)=12345");
     call_pyinit("NONE", "", "", 0.0);

     fPythia6Owner = this;
   }
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
#define SETCSAPARBUFSIZE 514
   char buf[SETCSAPARBUFSIZE];
   
   txgive_init_();
   for(std::vector<std::string>::const_iterator iter = fParamCSA.begin();
	                                        iter != fParamCSA.end(); ++iter)
   {
      // Null pad the string should not be needed because it uses
      // read, which will look for \n, but just in case...
      for (size_t i = 0; i < SETCSAPARBUFSIZE; ++i)
        buf[i] = ' ';
      // Skip empty parameters.
      if (iter->length() <= 0)
        continue;
      // Limit the size of the string to something which fits the buffer.
      size_t maxSize = iter->length() > (SETCSAPARBUFSIZE-2) ? (SETCSAPARBUFSIZE-2) : iter->length();
      strncpy(buf, iter->c_str(), maxSize);
      // Add extra \n if missing, otherwise "read" continues reading. 
      if (buf[maxSize-1] != '\n')
      {
         buf[maxSize] = '\n';
         // Null terminate in case the string is passed back to C.
         // Not sure that is actually needed.
	 buf[maxSize + 1] = 0;
      }
      txgive_(buf, iter->length() );
   }   
   
   return ;
#undef SETCSAPARBUFSIZE
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

void Pythia6Service::openPYUPDA( const char* file, bool write_file )
{

        if (write_file) {
	  std::cout<<"=== WRITING PYUPDA FILE "<<file<<" ==="<<std::endl;
	  fioopnw_( &fUnitPYUPDA, file, strlen(file) );
  	  // Write Pythia particle table to this card file.
          call_pyupda(1, fUnitPYUPDA);
        } else {
	  std::cout<<"=== READING PYUPDA FILE "<<file<<" ==="<<std::endl;
  	  fioopn_( &fUnitPYUPDA, file, strlen(file) );
  	  // Update Pythia particle table with this card file.
          call_pyupda(3, fUnitPYUPDA);
        }
	
	return;

}

void Pythia6Service::closeSLHA() 
{

   fiocls_(&fUnitSLHA);
   
   return;
}

void Pythia6Service::closePYUPDA() 
{

   fiocls_(&fUnitPYUPDA);
   
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

void Pythia6Service::setPYUPDAParams(bool afterPyinit)
{
   std::string shortfile;
   bool write_file = false;
   bool usePostPyinit = false;

   //   std::cout<<"=== CALLING setPYUPDAParams === "<<afterPyinit<<" "<<fParamPYUPDA.size()<<std::endl;

   // This assumes that PYUPDAFILE only appears once ...

   for (std::vector<std::string>::const_iterator iter = fParamPYUPDA.begin();
                                                 iter != fParamPYUPDA.end(); iter++ )
   {
     //     std::cout<<"PYUPDA check "<<*iter<<std::endl;
      	if( iter->find( "PYUPDAFILE", 0 ) != std::string::npos ) {
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
	  shortfile = iter->substr( start, end - start + 1 );
        } else if ( iter->find( "PYUPDAWRITE", 0 ) != std::string::npos ) {
          write_file = true;
        } else if ( iter->find( "PYUPDApostPYINIT", 0 ) != std::string::npos ) {
          usePostPyinit = true;
        }
   }
   
   if (!shortfile.empty()) {
     std::string file;
     if (write_file) {
       file = shortfile;
     } else {
       // If reading, get full path to file and require it to exist.
       FileInPath f1( shortfile );
       file = f1.fullPath();
     }

     if (afterPyinit == usePostPyinit || (write_file && afterPyinit)) {
       openPYUPDA( file.c_str(), write_file );
     }
   }

   return;
}

void Pythia6Service::setSLHAFromHeader( const std::vector<std::string> &lines )
{

	std::set<std::string> blocks;
	unsigned int model = 0, subModel = 0;

        std::string fnamest = boost::filesystem::unique_path().string();
        const char *fname = fnamest.c_str();
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
