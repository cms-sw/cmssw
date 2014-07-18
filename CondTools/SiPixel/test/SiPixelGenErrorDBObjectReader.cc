#include "CondTools/SiPixel/test/SiPixelGenErrorDBObjectReader.h"
#include <iomanip>
#include <fstream>
#include <iostream>
#include <cmath>
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "MagneticField/Engine/interface/MagneticField.h"

SiPixelGenErrorDBObjectReader::SiPixelGenErrorDBObjectReader(const edm::ParameterSet& iConfig):
	theGenErrorCalibrationLocation( iConfig.getParameter<std::string>("siPixelGenErrorCalibrationLocation") ),
	theDetailedGenErrorDBErrorOutput( iConfig.getParameter<bool>("wantDetailedGenErrorDBErrorOutput") ),
	theFullGenErrorDBOutput( iConfig.getParameter<bool>("wantFullGenErrorDBOutput") ),
	testGlobalTag( iConfig.getParameter<bool>("TestGlobalTag") ),
	hasTriggeredWatcher(false)
{
}

SiPixelGenErrorDBObjectReader::~SiPixelGenErrorDBObjectReader()
{
}

void 
SiPixelGenErrorDBObjectReader::beginJob()
{
}

void
SiPixelGenErrorDBObjectReader::analyze(const edm::Event& iEvent, const edm::EventSetup& setup)
{
	//To test with the ESProducer
	if(testGlobalTag) {
		edm::ESHandle<MagneticField> magfield;
		setup.get<IdealMagneticFieldRecord>().get(magfield);
		GlobalPoint center(0.0, 0.0, 0.0);
		float theMagField = magfield.product()->inTesla(center).mag();

		std::cout << "\nTesting global tag at magfield = " << theMagField << std::endl;
//		if(SiPixTemplDBObjWatcher_.check(setup)) {
		if(SiPixGenerDBObjWatcher_.check(setup)) {
			edm::ESHandle<SiPixelGenErrorDBObject> generrorH;
			setup.get<SiPixelGenErrorDBObjectESProducerRcd>().get(generrorH);
			dbobject = *generrorH.product();
			hasTriggeredWatcher=true;
		}
	}
	else {
		std::cout << "\nLoading from file " << std::endl;
//		if(SiPixTemplDBObjWatcher_.check(setup)) {
		if(SiPixGenerDBObjWatcher_.check(setup)) {
			edm::ESHandle<SiPixelGenErrorDBObject> generrorH;
			setup.get<SiPixelGenErrorDBObjectRcd>().get(generrorH);
			dbobject = *generrorH.product();
			hasTriggeredWatcher=true;
		}
	}

	if(hasTriggeredWatcher) {
		std::vector<short> tempMapId;
		
		if(theFullGenErrorDBOutput) std::cout << "Map info" << std::endl;
		std::map<unsigned int,short> templMap=dbobject.getGenErrorIDs();
		for(std::map<unsigned int,short>::const_iterator it=templMap.begin(); it!=templMap.end();++it) {
			if(tempMapId.size()==0) tempMapId.push_back(it->second);
			for(unsigned int i=0; i<tempMapId.size();++i) {
				if(tempMapId[i]==it->second) continue;
				else if(i==tempMapId.size()-1) {
					tempMapId.push_back(it->second);
					break;
				}
			}
			if(theFullGenErrorDBOutput)
				std::cout<< "DetId: "<< it->first<<" GenErrorID: "<< it->second<<"\n";
		}
		
		std::cout << "\nMap stores GenError Id(s): ";
		for(unsigned int vindex=0; vindex < tempMapId.size(); ++ vindex)
			std::cout << tempMapId[vindex] << " ";
		std::cout << std::endl;
		
		//local variables
		const char * tempfile;
		char c;
		int numOfTempl = dbobject.numOfTempl();
		int index = 0;
		float tempnum=0,diff=0;
		float tol = 1.0E-23;
		bool error=false,givenErrorMsg=false;;
		
		std::cout << "\nChecking GenError DB object version " << dbobject.version() << " containing " << numOfTempl << " calibration(s) at " << dbobject.sVector()[index+22] <<"T\n";
		for(int i=0; i<numOfTempl;++i) {
			
			//Removes header in db object from diff
			index+=20;
			
			//Tell the person viewing the output what the GenError ID and version are -- note that version is only valid for >=13
			std::cout << "Calibration " << i+1 << " of " << numOfTempl << ", with GenError ID " << dbobject.sVector()[index]
								<< "\tand Version " << dbobject.sVector()[index+1] <<"\t--------  ";
			
			//Opening the text-based GenError calibration
			std::ostringstream tout;
			tout << theGenErrorCalibrationLocation.c_str() << "/data/generror_summary_zp"
					 << std::setw(4) << std::setfill('0') << std::right << dbobject.sVector()[index] << ".out" << std::ends;
			
			edm::FileInPath file( tout.str());
			tempfile = (file.fullPath()).c_str();
			std::ifstream in_file(tempfile, std::ios::in);
			
			if(in_file.is_open()) {
				
				//Removes header in textfile from diff
				for(int header=0; (c=in_file.get()) != '\n'; ++header) {}
				
				//First read in from the text file -- this will be compared with index = 20
				in_file >> tempnum;
				
				//Read until the end of the current text file
				while(!in_file.eof())
				{
					//Calculate the difference between the text file and the db object
					diff = std::abs(tempnum - dbobject.sVector()[index]);
					
					//Is there a difference?
					if(diff > tol) {
						//We have the if statement to output the message only once
						if(!givenErrorMsg) std::cout << "does NOT match\n";
						//If there is an error we want to display a message upon completion
						error = true;
						givenErrorMsg = true;
						//Do we want more detailed output?
						if(theDetailedGenErrorDBErrorOutput) {
							std::cout << "from file = " << tempnum << "\t from dbobject = " << dbobject.sVector()[index]
												<< "\tdiff = "    << diff    << "\t db index = "      << index << std::endl;
						}
					}
					//Go to the next entries
					in_file >> tempnum;
					++index;
				}
				//There were no errors, the two files match.
				if(!givenErrorMsg) std::cout << "MATCHES\n";
			}//end current file
			in_file.close();
			givenErrorMsg = false;
		}//end loop over all files
		
		if(error && !theDetailedGenErrorDBErrorOutput)
			std::cout << "\nThe were differences found between the files and the database.\n"
								<< "If you would like more detailed information please set\n"
								<< "wantDetailedOutput = True in the cfg file. If you would like a\n"
								<< "full output of the contents of the database file please set\n"
								<< "wantFullOutput = True. Make sure that you pipe the output to a\n"
								<< "log file. This could take a few minutes.\n\n";
		
		if(theFullGenErrorDBOutput) std::cout << dbobject << std::endl;
	}
}


void 
SiPixelGenErrorDBObjectReader::endJob()
{
}

std::ostream& operator<<(std::ostream& s, const SiPixelGenErrorDBObject& dbobject){
	//!-index to keep track of where we are in the object
	int index = 0;
	//!-these are modifiable parameters for the extended GenErrors
	int txsize[4] = { 7,13,0,0};
	int tysize[4] = {21,21,0,0};
	//!-entries takes the number of entries in By,Bx,Fy,Fx from the object
	int entries[4] = {0};
	//!-local indicies for loops
	int i,j,k,l,m,n,entry_it;
	//!-changes the size of the GenErrors based on the version
	int sizeSetter=0,generrorVersion=0;

	std::cout << "\n\nDBobject version: " << dbobject.version() << std::endl;

	for(m=0; m < dbobject.numOfTempl(); ++m) 
	{
		//To change the size of the output based on which GenError version we are using"
		generrorVersion = (int) dbobject.sVector_[index+21];
		if(generrorVersion<=10) {
			std::cout << "*****WARNING***** This code will not format this GenError version properly *****WARNING*****\n";
			sizeSetter=0;
		}
		else if(generrorVersion<=16) sizeSetter=1;
		else std::cout << "*****WARNING***** This code has not been tested at formatting this version *****WARNING*****\n";
		
		std::cout << "\n\n*********************************************************************************************" << std::endl;
		std::cout << "***************                  Reading GenError ID " << dbobject.sVector_[index+20]
							<< "\t(" << m+1 << "/" << dbobject.numOfTempl_ <<")                 ***************" << std::endl;
		std::cout << "*********************************************************************************************\n\n" << std::endl;
		
		//Header Title
		SiPixelGenErrorDBObject::char2float temp;
		for (n=0; n < 20; ++n) {
			temp.f = dbobject.sVector_[index];
			s << temp.c[0] << temp.c[1] << temp.c[2] << temp.c[3];
			++index;
		}

		entries[0] = (int) dbobject.sVector_[index+3];                               // Y
		entries[1] = (int)(dbobject.sVector_[index+4]*dbobject.sVector_[index+5]);   // X
				
		//Header
		s        << dbobject.sVector_[index]   <<"\t"<< dbobject.sVector_[index+1]  <<"\t"<< dbobject.sVector_[index+2]
			<<"\t"<< dbobject.sVector_[index+3]  <<"\t"<< dbobject.sVector_[index+4]  <<"\t"<< dbobject.sVector_[index+5]
			<<"\t"<< dbobject.sVector_[index+6]  <<"\t"<< dbobject.sVector_[index+7]  <<"\t"<< dbobject.sVector_[index+8]
			<<"\t"<< dbobject.sVector_[index+9]  <<"\t"<< dbobject.sVector_[index+10] <<"\t"<< dbobject.sVector_[index+11]
      <<"\t"<< dbobject.sVector_[index+12] <<"\t"<< dbobject.sVector_[index+13] <<"\t"<< dbobject.sVector_[index+14]
			<<"\t"<< dbobject.sVector_[index+15] <<"\t"<< dbobject.sVector_[index+16] << std::endl;
		index += 17;
			
		//Loop over By,Bx,Fy,Fx
		for(entry_it=0;entry_it<4;++entry_it) {
			//Run,costrk,qavg,...,clslenx
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
				//YPar
				for(j=0;j<2;++j)
				{
					for(k=0;k<5;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				//YTemp
				for(j=0;j<9;++j)
				{
					for(k=0;k<tysize[sizeSetter];++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				//XPar
				for(j=0;j<2;++j)
				{
					for(k=0;k<5;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				//XTemp
				for(j=0;j<9;++j)
				{
					for(k=0;k<txsize[sizeSetter];++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				//Y average reco params
				for(j=0;j<4;++j)
				{
					for(k=0;k<4;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				//Yflpar
				for(j=0;j<4;++j)
				{
					for(k=0;k<6;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				//X average reco params
				for(j=0;j<4;++j)
				{
					for(k=0;k<4;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				//Xflpar
				for(j=0;j<4;++j)
				{
					for(k=0;k<6;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				//Chi2X,Y
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
				//Y average Chi2 params
				for(j=0;j<4;++j)
				{
					for(k=0;k<4;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				//X average Chi2 params
				for(j=0;j<4;++j)
				{
					for(k=0;k<4;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				//Y average reco params for CPE Generic
				for(j=0;j<4;++j)
				{
					for(k=0;k<4;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				//X average reco params for CPE Generic
				for(j=0;j<4;++j)
				{
					for(k=0;k<4;++k)
					{
						s << dbobject.sVector_[index] << "\t";
						++index;
					}
					s << std::endl;
				}
				//SpareX,Y
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
