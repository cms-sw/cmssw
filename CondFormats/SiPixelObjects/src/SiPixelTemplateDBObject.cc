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
	int m;
	
	// Set the number of templates to be passed to the dbobject
	numOfTempl_ = atitles.size();
	
	//  open the template file(s) 
	for(m=0; m<numOfTempl_; ++m){

		edm::FileInPath file( atitles[m].c_str() );
		tempfile = (file.fullPath()).c_str();

		std::ifstream in_file(tempfile, std::ios::in);
			
	if(in_file.is_open()){
		edm::LogInfo("SiPixelTemplateDBObject") << "Opened Template File: " << file.fullPath().c_str() << std::endl;

		// Local variables 
		char title_char[80], c;
		char2float temp;
		float tempstore;
		int iter,j;

		// Templates contain a header char - we must be clever about storing this
		for (iter = 0; (c=in_file.get()) != '\n'; ++iter) {
			if(iter < 79) {title_char[iter] = c;}
		}
		if(iter > 78) {iter=78;}
		title_char[iter+1] ='\n';

		for(j=0; j<80; j+=4) {
			temp.c[0] = title_char[j];
			temp.c[1] = title_char[j+1];
			temp.c[2] = title_char[j+2];
			temp.c[3] = title_char[j+3];
			sVector_.push_back(temp.f);
			++maxIndex_;
		}
			
		// Fill the dbobject
		in_file >> tempstore;
		while(!in_file.eof()) {
			++maxIndex_;
			sVector_.push_back(tempstore);
			in_file >> tempstore;
		}
		
		in_file.close();
	}
	else {
		// If file didn't open, report this
		edm::LogError("SiPixelTemplateDBObject") << "Error opening File" << tempfile << std::endl;
	}
	}
}

std::ostream& operator<<(std::ostream& s, const SiPixelTemplateDBObject& dbobject){
	//!-index to keep track of where we are in the object
	int index = 0;
	//!-these are modifiable parameters for the extended templates
	int txsize = 13, tysize = 21;
	//!-entries takes the number of entries in By,Bx,Fy,Fx from the object
	int entries[4] = {0};
	//!-local indicies for loops
	int i,j,k,l,m,n,entry_it;
	

	std::cout << "DBobject version: " << dbobject.version() << std::endl;

	for(m=0; m < dbobject.numOfTempl_; ++m) 
	{
		std::cout << "\n\n*********************************************************************************************" << std::endl;
		std::cout << "***************                  Reading Template ID " << dbobject.sVector_[index+20] << "\t(" << m+1 << "/" << dbobject.numOfTempl_ <<")                 ***************" << std::endl;
		std::cout << "*********************************************************************************************\n\n" << std::endl;
		SiPixelTemplateDBObject::char2float temp;
		for (n=0; n < 20; ++n) {
			temp.f = dbobject.sVector_[index];
			s << temp.c[0] << temp.c[1] << temp.c[2] << temp.c[3];
			++index;
		}
		
		entries[0] = (int)dbobject.sVector_[index+1];
		entries[1] = (int)(dbobject.sVector_[index+2]*dbobject.sVector_[index+3]);
		entries[2] = (int)dbobject.sVector_[index+4];
		entries[3] = (int)(dbobject.sVector_[index+5]*dbobject.sVector_[index+6]); 

		s         << dbobject.sVector_[index]    << "\t" << dbobject.sVector_[index+1]  << "\t" << dbobject.sVector_[index+2]
			<< "\t" << dbobject.sVector_[index+3]  << "\t" << dbobject.sVector_[index+4]  << "\t" << dbobject.sVector_[index+5]
			<< "\t" << dbobject.sVector_[index+6]  << "\t" << dbobject.sVector_[index+7]  << "\t" << dbobject.sVector_[index+8]
			<< "\t" << dbobject.sVector_[index+9]  << "\t" << dbobject.sVector_[index+10] << "\t" << dbobject.sVector_[index+11]
		  << "\t" << dbobject.sVector_[index+12] << "\t" << dbobject.sVector_[index+13] << std::endl;
		index += 14;
	
		for(entry_it=0;entry_it<4;++entry_it) {
			for(i=0;i < entries[entry_it];++i)
			{
				s         << dbobject.sVector_[index]    << "\t" << dbobject.sVector_[index+1]  << "\t" << dbobject.sVector_[index+2]
					<< "\t" << dbobject.sVector_[index+3]  << "\n" << dbobject.sVector_[index+4]  << "\t" << dbobject.sVector_[index+5]
					<< "\t" << dbobject.sVector_[index+6]  << "\t" << dbobject.sVector_[index+7]  << "\t" << dbobject.sVector_[index+8]
					<< "\t" << dbobject.sVector_[index+9]  << "\t" << dbobject.sVector_[index+10] << "\t" << dbobject.sVector_[index+11]
				  << "\n" << dbobject.sVector_[index+12] << "\t" << dbobject.sVector_[index+13] << "\t" << dbobject.sVector_[index+14]
					<< "\t" << dbobject.sVector_[index+15] << "\t" << dbobject.sVector_[index+16] << "\t" << dbobject.sVector_[index+17]
					<< "\t" << dbobject.sVector_[index+18] << std::endl;
				index+=19;
				for(j=0;j<2;++j)
				{
					for(k=0;k<5;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(j=0;j<9;++j)
				{
					for(k=0;k<tysize;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(j=0;j<2;++j)
				{
					for(k=0;k<5;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(j=0;j<9;++j)
				{
					for(k=0;k<txsize;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(j=0;j<4;++j)
				{
					for(k=0;k<4;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(j=0;j<4;++j)
				{
					for(k=0;k<6;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(j=0;j<4;++j)
				{
					for(k=0;k<4;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(j=0;j<4;++j)
				{
					for(k=0;k<6;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(j=0;j<4;++j)
				{
					for(k=0;k<2;++k)
					{
						for(l=0;l<2;++l)
						{
							s << dbobject.sVector_[index] << "\t";
							++index;
						}
					}
					s << std::endl;
				}
				for(j=0;j<4;++j)
				{
					for(k=0;k<4;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(j=0;j<4;++j)
				{
					for(k=0;k<4;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				for(j=0;j<20;++j)
				{
					s << dbobject.sVector_[index] << "\t";
					++index;
					if(j==9 ||j==19) s << std::endl;
				}
			}
		}
	}
	return s;
}
