#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//fill SiPixelTemplateDBObject
void
SiPixelTemplateDBObject::fillDB(const vstring& atitles){
	
	// Local variables 
	const char *tempfile;

	// Set the number of templates to be passed to the dbobject
	numOfTempl_ = atitles.size();
		
	//  open the template file(s) 
	for(int m=0; m<numOfTempl_; ++m){

		std::ostringstream tout;
		tout << "RecoLocalTracker/SiPixelRecHits/data/template_summary_zp" 
				 << std::setw(4) << std::setfill('0') << std::right << atitles[m] << ".out" << std::ends;
		std::string tempf = tout.str();
					
		edm::FileInPath file( tempf.c_str() );
		tempfile = (file.fullPath()).c_str();

		std::ifstream in_file(tempfile, std::ios::in);
			
	if(in_file.is_open()){
		edm::LogInfo("SiPixelTemplateDBObject") << "Opened Template File: " << file.fullPath().c_str() << std::endl;

		// Local variables 
		char title_char[80], c;
		char2float temp;
		float tempstore;
		int iter = 0;

		// Templates contain a header char - we must be clever about storing this
		for (iter = 0; (c=in_file.get()) != '\n'; ++iter) {
			if(iter < 79) {title_char[iter] = c;}
		}
		if(iter > 78) {iter=78;}
		title_char[iter+1] ='\n';

		for(int j=0; j<80; j+=4) {
			temp.c[0] = title_char[j];
			temp.c[1] = title_char[j+1];
			temp.c[2] = title_char[j+2];
			temp.c[3] = title_char[j+3];
			sVector_.push_back(temp.f);
			++maxIndex_;
		}
			
		// Fill the dbobject
		while(!in_file.eof()) {
			in_file >> tempstore;
			sVector_.push_back(tempstore);
			++maxIndex_;
		}
		--maxIndex_; // To account for incrementing before realizing we hit the eof
		in_file.close();
		tout.clear();
		tempf.clear();
	}
	else {
		// If file didn't open, report this
		edm::LogError("SiPixelTemplateDBObject") << "Error opening File" << tempfile << std::endl;
	}
	}
}

std::ostream& operator<<(std::ostream& s, const SiPixelTemplateDBObject& dbobject){
	int index = 0;
	bool new_templ = false;
	int txsize, tysize;
	int entries[4] = {0};
	for(int m=0; m < dbobject.numOfTempl_; ++m) 
	{
		SiPixelTemplateDBObject::char2float temp;
		for (int n=0; n < 20; ++n) {
			temp.f = dbobject.sVector_[index];
			s << temp.c[0] << temp.c[1] << temp.c[2] << temp.c[3];
			++index;
		}
		
		if(dbobject.sVector_[index] == 10 || dbobject.sVector_[index] == 12) new_templ = true;
		
		entries[0] = dbobject.sVector_[index+1];
		entries[1] = dbobject.sVector_[index+2]*dbobject.sVector_[index+3];
		entries[2] = dbobject.sVector_[index+4];
		entries[3] = dbobject.sVector_[index+5]*dbobject.sVector_[index+6]; 

		s         << dbobject.sVector_[index]    << "\t" << dbobject.sVector_[index+1]  << "\t" << dbobject.sVector_[index+2]
			<< "\t" << dbobject.sVector_[index+3]  << "\t" << dbobject.sVector_[index+4]  << "\t" << dbobject.sVector_[index+5]
			<< "\t" << dbobject.sVector_[index+6]  << "\t" << dbobject.sVector_[index+7]  << "\t" << dbobject.sVector_[index+8]
			<< "\t" << dbobject.sVector_[index+9]  << "\t" << dbobject.sVector_[index+10] << "\t" << dbobject.sVector_[index+11];
		if(new_templ) {s << "\t" << dbobject.sVector_[index+12] << "\t" << dbobject.sVector_[index+13] << std::endl; index += 14;}
		else {s << "\t" << dbobject.sVector_[index+12] << std::endl; index += 13;}
	
		if(new_templ) {txsize = 13; tysize = 21;}
		else {txsize = 7; tysize = 21;}
		for(int entry_it=0;entry_it<4;++entry_it) {
			for(int i=0;i < entries[entry_it];++i)
			{
				s         << dbobject.sVector_[index]    << "\t" << dbobject.sVector_[index+1]  << "\t" << dbobject.sVector_[index+2]
					<< "\t" << dbobject.sVector_[index+3]  << "\n" << dbobject.sVector_[index+4]  << "\t" << dbobject.sVector_[index+5]
					<< "\t" << dbobject.sVector_[index+6]  << "\t" << dbobject.sVector_[index+7]  << "\t" << dbobject.sVector_[index+8]
					<< "\t" << dbobject.sVector_[index+9]  << "\t" << dbobject.sVector_[index+10];
				if(new_templ) {s << "\t" << dbobject.sVector_[index+11]; index += 12;}
				else {index += 11;}
				s << "\n" << dbobject.sVector_[index]   << "\t" << dbobject.sVector_[index+1] << "\t" << dbobject.sVector_[index+2]
					<< "\t" << dbobject.sVector_[index+3] << "\t" << dbobject.sVector_[index+4] << "\t" << dbobject.sVector_[index+5]
					<< "\t" << dbobject.sVector_[index+6] << std::endl;
				index+=7;
				for(int j=0;j<2;++j)
				{
					for(int k=0;k<5;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(int j=0;j<9;++j)
				{
					for(int k=0;k<tysize;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(int j=0;j<2;++j)
				{
					for(int k=0;k<5;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(int j=0;j<9;++j)
				{
					for(int k=0;k<txsize;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(int j=0;j<4;++j)
				{
					for(int k=0;k<4;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(int j=0;j<4;++j)
				{
					for(int k=0;k<6;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(int j=0;j<4;++j)
				{
					for(int k=0;k<4;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(int j=0;j<4;++j)
				{
					for(int k=0;k<6;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(int j=0;j<4;++j)
				{
					for(int k=0;k<2;++k)
					{
						for(int l=0;l<2;++l)
						{
							s << dbobject.sVector_[index] << "\t";
							++index;
						}
					}
					s << std::endl;
				}
				for(int j=0;j<4;++j)
				{
					for(int k=0;k<4;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(int j=0;j<4;++j)
				{
					for(int k=0;k<4;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(int j=0;j<20;++j)
				{
					s << dbobject.sVector_[index] << "\t";
					++index;
					if(j==9 ||j==19) s << std::endl;
				}
			}
		}
		++index;
	}
	return s;
}
