#ifndef Input_H2RootNtpl_h
#define Input_H2RootNtplSource_h

/** \class  H2RootNtplSource 
*
* Reads in HepMC events
* Joanna Weng & Filip Moortgat 1/2006 
***************************************/


#include "FWCore/Sources/interface/ExternalInputSource.h"
#include "IOMC/Input/interface/HepMCFileReader.h"
#include <map>
#include <string>
class Ntuple2HepMCFiller;

namespace edm
{
	class H2RootNtplSource : public ExternalInputSource {
		public:
		H2RootNtplSource(const ParameterSet &, const InputSourceDescription &  );
		virtual ~H2RootNtplSource();
		
		private:		
	       	void clear();	
		virtual bool produce(Event &e);		
		HepMC::GenEvent  *evt;
		EventID nextID_;	
		std::string filename_;
	public:
		unsigned int firstEvent_;
		Ntuple2HepMCFiller * reader_;
		
	};
} 

#endif
