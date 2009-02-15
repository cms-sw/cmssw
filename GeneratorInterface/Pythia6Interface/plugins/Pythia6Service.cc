
#include "Pythia6Service.h"
#include "Pythia6Declarations.h"
// #include "GeneratorInterface/Core/interface/ParameterCollector.h"

using namespace gen;
using namespace edm;

Pythia6Service::Pythia6Service( const ParameterSet& ps )
{

/*
   ParameterCollector collector(ps.getParameter<edm::ParameterSet>("PythiaParameters"));

   paramGeneral.clear();
   paramCSA.clear();
   paramSLHA.clear();
   
   paramGeneral = std::vector<std::string>(collector.begin(), collector.end());
   paramCSA     = std::vector<std::string>(collector.begin("CSAParameters"), collector.end());
   paramSLHA    = std::vector<std::string>(collector.begin("SLHAParameters"), collector.end());
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
   paramGeneral.clear();
   paramCSA.clear();
   paramSLHA.clear();
   

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
	    paramCSA.push_back(*line);
	 }
	 else if ( *iter == "SLHAParameteters" )
	 {
	    paramSLHA.push_back(*line);
	 }
	 else
	 {
	    paramGeneral.push_back(*line);
	 }
      }
   }

}

Pythia6Service::~Pythia6Service()
{
   paramGeneral.clear();
   paramCSA.clear();
   paramSLHA.clear();
}

void Pythia6Service::setGeneralParams()
{
   // now pass general config cards 
   //
   for(std::vector<std::string>::const_iterator iter = paramGeneral.begin();
	                                        iter != paramGeneral.end(); ++iter)
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
   
   for(std::vector<std::string>::const_iterator iter = paramCSA.begin();
	                                        iter != paramCSA.end(); ++iter)
   {
      txgive_( iter->c_str(), iter->length() );
   }   
   
   return ;
}

void Pythia6Service::setSLHAParams()
{
   return;
}
