//
// This class manages data and files used
// in the Delay25 calibration
//
//
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelDelay25Calib.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTimeFormatter.h"
#include <iostream>
#include <assert.h>
#include <map>
#include <sstream>

using namespace pos;

using namespace std;

PixelDelay25Calib::PixelDelay25Calib(vector< vector<string> > &tableMat) : 
  PixelCalibBase(),
  PixelConfigBase("","","")
{
  std::string mthn = "[PixelDelay25Calib::PixelDelay25Calib()]\t\t\t    " ;
  std::map<std::string , int > colM;
  std::vector<std::string > colNames;
  /**

  EXTENSION_TABLE_NAME: PIXEL_CALIB_CLOB (VIEW: CONF_KEY_PIXEL_CALIB_V)
  
  CONFIG_KEY				    NOT NULL VARCHAR2(80)
  KEY_TYPE				    NOT NULL VARCHAR2(80)
  KEY_ALIAS				    NOT NULL VARCHAR2(80)
  VERSION					     VARCHAR2(40)
  KIND_OF_COND  			    NOT NULL VARCHAR2(40)
  CALIB_TYPE					     VARCHAR2(200)
  CALIB_OBJ_DATA_FILE			    NOT NULL VARCHAR2(200)
  CALIB_OBJ_DATA_CLOB			    NOT NULL CLOB
  */

  colNames.push_back("CONFIG_KEY"  	  );
  colNames.push_back("KEY_TYPE"    	  );
  colNames.push_back("KEY_ALIAS"   	  );
  colNames.push_back("VERSION"     	  );
  colNames.push_back("KIND_OF_COND"	  );
  colNames.push_back("CALIB_TYPE"  	  );
  colNames.push_back("CALIB_OBJ_DATA_FILE");
  colNames.push_back("CALIB_OBJ_DATA_CLOB");
 
  for(unsigned int c = 0 ; c < tableMat[0].size() ; c++)
    {
      for(unsigned int n=0; n<colNames.size(); n++)
	{
	  if(tableMat[0][c] == colNames[n])
	    {
	      colM[colNames[n]] = c;
	      break;
	    }
	}
    }//end for
  for(unsigned int n=0; n<colNames.size(); n++)
    {
      if(colM.find(colNames[n]) == colM.end())
	{
	  std::cerr << "[[PixelDelay25Calib::PixelDelay25Calib()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
	  assert(0);
	}
    }

  
  std::istringstream in ;
  in.str(tableMat[1][colM["CALIB_OBJ_DATA_CLOB"]]) ;
  
  //Read initial SDa and RDa values, ranges,
  //and grid step size from file
  
  std::string tmp;

  in >> tmp;

  assert(tmp=="Mode:");
  in >> mode_;

  in >> tmp;

  assert(tmp=="Portcards:");
  in >> tmp;
  if(tmp=="All")
    {
      allPortcards_ = true;
    } else {
    allPortcards_ = false;
  }
  while (tmp!="AllModules:")
    {
      portcardNames_.insert(tmp);
      in >> tmp;
    }

  assert(tmp=="AllModules:");
  in >> allModules_;

  in >> tmp;

  assert(tmp=="OrigSDa:");
  in >> origSDa_;

  in >> tmp;

  assert(tmp=="OrigRDa:");
  in >> origRDa_;

  in >> tmp;

  assert(tmp=="Range:");
  in >> range_;

  in >> tmp;

  assert(tmp=="GridSize:");
  in >> gridSize_;

  in >> tmp;
  assert(tmp=="Tests:");
  in >> numTests_;

  in >> tmp;
  if(tmp=="Commands:") {
    in >> commands_;
  } else {
    commands_=0;
  }

  //Number of steps in the grid
  gridSteps_ = range_/gridSize_;

  // Added by Dario as a temporary patch for Debbie (this will disappear in the future)
  calibFileContent_ = in.str() ;
  //cout << __LINE__ << "] " << __PRETTY_FUNCTION__ << "\tcalibFileContent_\n " << calibFileContent_ << endl ;  
  // End of temporary patch
  
}


PixelDelay25Calib::PixelDelay25Calib(std::string filename) : 
  PixelCalibBase(),
  PixelConfigBase("","",""){

  std::string mthn = "[PixelDelay25Calib::PixelDelay25Calib()]\t\t\t    ";
  
  std::ifstream in(filename.c_str());
  
  if(!in.good()){
    std::cout << __LINE__ << "]\t" << mthn << "Could not open: " << filename << std::endl;
    assert(0);
  }
  else {
    std::cout << __LINE__ << "]\t" << mthn << "Opened: " << filename << std::endl;
  }

  //Read initial SDa and RDa values, ranges,
  //and grid step size from file
  
  std::string tmp;

  in >> tmp;

  assert(tmp=="Mode:");
  in >> mode_;

  //cout << __LINE__ << "]\t" << mthn  << "mode_="<<mode_<<endl;

  in >> tmp;

  assert(tmp=="Portcards:");
  in >> tmp;
  if(tmp=="All")
    {
      allPortcards_ = true;
    } else {
      allPortcards_ = false;
    }
  while (tmp!="AllModules:")
    {
      portcardNames_.insert(tmp);
      in >> tmp;
    }

  assert(tmp=="AllModules:");
  in >> allModules_;

  in >> tmp;

  assert(tmp=="OrigSDa:");
  in >> origSDa_;

  in >> tmp;

  assert(tmp=="OrigRDa:");
  in >> origRDa_;

  in >> tmp;

  assert(tmp=="Range:");
  in >> range_;

  in >> tmp;

  assert(tmp=="GridSize:");
  in >> gridSize_;

  in >> tmp;
  assert(tmp=="Tests:");
  in >> numTests_;

  in >> tmp;
  if(tmp=="Commands:") {
    in >> commands_;
  } else {
    commands_=0;
  }

  in.close();

  //Number of steps in the grid
  gridSteps_ = range_/gridSize_;

  // Added by Dario as a temporary patch for Debbie (this will disappear in the future)
  std::ifstream inTmp(filename.c_str());
  calibFileContent_ = "" ;
  while(!inTmp.eof())
  {
   std::string tmpString ;
   getline (inTmp, tmpString);
   calibFileContent_ += tmpString + "\n";
   //cout << __LINE__ << "]\t" << "[PixelCalibConfiguration::~PixelCalibConfiguration()]\t\t" << calibFileContent_ << endl ;
  }
  inTmp.close() ;
  // End of temporary patch
}

PixelDelay25Calib::~PixelDelay25Calib() {
}

void PixelDelay25Calib::openFiles(std::string portcardName, std::string moduleName, std::string path) {
  if (path!="") path+="/";
  graph_ = path+"graph_"+portcardName+"_"+moduleName+".dat";
  graphout_.open(graph_.c_str());
  return;
}

void PixelDelay25Calib::writeSettings(std::string portcardName, std::string moduleName) {
  graphout_ << "Portcard: " << portcardName << endl;
  graphout_ << "Module: " << moduleName << endl;
  graphout_ << "SDaOrigin: " << origSDa_ << endl;
  graphout_ << "RDaOrigin: " << origRDa_ << endl;
  graphout_ << "SDaRange: " << range_ << endl;
  graphout_ << "RDaRange: " << range_ << endl;
  graphout_ << "GridSize: " << gridSize_ << endl;
  graphout_ << "Tests: " << numTests_ << endl;
  return;
}

void PixelDelay25Calib::writeFiles( std::string tmp ) {
  graphout_ << tmp << endl;
  return;
}

void PixelDelay25Calib::writeFiles( int currentSDa, int currentRDa, int number ) {
  graphout_ << currentSDa << " " << currentRDa << " " << number << endl;
  return;
}

void PixelDelay25Calib::closeFiles() {
  graphout_.close();
  return;
}

void PixelDelay25Calib::writeASCII(std::string dir) const {


  //FIXME this is not tested for all the use cases...

  if (dir!="") dir+="/";
  std::string filename=dir+"delay25.dat";
  std::ofstream out(filename.c_str());

  out << "Mode: "<<mode_<<endl;
  
  out << "Portcards:" <<endl;

  std::set<std::string>::const_iterator i=portcardNames_.begin();
  while (i!=portcardNames_.end()) {
    out << *i << endl;
    ++i;
  }

  out << "AllModules:" <<endl;
  if (allModules_) {
    out << "1" <<endl;
  } else {
    out << "0" <<endl;
  }

  out << "OrigSDa:"<<endl;
  out << origSDa_<<endl;
  
  out << "OrigRDa:"<<endl;
  out << origRDa_<<endl;
  
  out << "Range:"<<endl;
  out << range_<<endl;
  
  out << "GridSize:"<<endl;
  out << gridSize_<<endl;
  
  out << "Tests:"<<endl;
  out << numTests_<<endl;

  out << "Commands:"<<endl;
  out << commands_<<endl;
  
  out.close();
}

//=============================================================================================
void PixelDelay25Calib::writeXMLHeader(pos::PixelConfigKey key, 
                                       int version, 
                                       std::string path, 
                                       std::ofstream *outstream,
                                       std::ofstream *out1stream,
                                       std::ofstream *out2stream) const
{
  std::string mthn = "[PixelDelay25Calib::writeXMLHeader()]\t\t\t    " ;
  std::stringstream maskFullPath ;

  writeASCII(path) ;

  maskFullPath << path << "/PixelCalib_Test_" << PixelTimeFormatter::getmSecTime() << ".xml";
  std::cout  << __LINE__ << "]\t" << mthn << "Writing to: " << maskFullPath.str() << std::endl ;

  outstream->open(maskFullPath.str().c_str()) ;
  
  *outstream << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"                                 << std::endl ;
  *outstream << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" 		 	          << std::endl ;
  *outstream << ""                                                                                        << std::endl ; 
  *outstream << " <!-- " << mthn << "-->"                                                                 << std::endl ; 
  *outstream << ""                                                                                        << std::endl ; 
  *outstream << " <HEADER>"                                                                               << std::endl ; 
  *outstream << "  <TYPE>"                                                                                << std::endl ; 
  *outstream << "   <EXTENSION_TABLE_NAME>PIXEL_CALIB_CLOB</EXTENSION_TABLE_NAME>"                        << std::endl ; 
  *outstream << "   <NAME>Calibration Object Clob</NAME>"                                                 << std::endl ; 
  *outstream << "  </TYPE>"                                                                               << std::endl ; 
  *outstream << "  <RUN>"                                                                                 << std::endl ; 
  *outstream << "   <RUN_TYPE>delay25</RUN_TYPE>"                                                         << std::endl ; 
  *outstream << "   <RUN_NUMBER>1</RUN_NUMBER>"                                                           << std::endl ; 
  *outstream << "   <RUN_BEGIN_TIMESTAMP>" << PixelTimeFormatter::getTime() << "</RUN_BEGIN_TIMESTAMP>"   << std::endl ; 
  *outstream << "   <LOCATION>CERN P5</LOCATION>"                                                         << std::endl ; 
  *outstream << "  </RUN>"                                                                                << std::endl ; 
  *outstream << " </HEADER>"                                                                              << std::endl ; 
  *outstream << ""                                                                                        << std::endl ; 
  *outstream << " <DATA_SET>"                                                                             << std::endl ;
  *outstream << ""                                                                                        << std::endl ;
  *outstream << "  <VERSION>"             << version      << "</VERSION>"                                 << std::endl ;
  *outstream << "  <COMMENT_DESCRIPTION>" << getComment() << "</COMMENT_DESCRIPTION>"			  << std::endl ;
  *outstream << "  <CREATED_BY_USER>"     << getAuthor()  << "</CREATED_BY_USER>"  			  << std::endl ;
  *outstream << ""                                                                                        << std::endl ;
  *outstream << "  <PART>"                                                                                << std::endl ;
  *outstream << "   <NAME_LABEL>CMS-PIXEL-ROOT</NAME_LABEL>"                                              << std::endl ;      
  *outstream << "   <KIND_OF_PART>Detector ROOT</KIND_OF_PART>"                                           << std::endl ;         
  *outstream << "  </PART>"                                                                               << std::endl ;

}

//=============================================================================================
void PixelDelay25Calib::writeXML( std::ofstream *outstream,
                                  std::ofstream *out1stream,
                                  std::ofstream *out2stream) const 
{
  std::string mthn = "[PixelDelay25Calib::writeXML()]\t\t\t    " ;
  
  std::cout  << __LINE__ << "]\t" << mthn << "Writing.." << std::endl ;

  *outstream << " "                                                                                       << std::endl ;
  *outstream << "  <DATA>"                                                                                << std::endl ;
  *outstream << "   <CALIB_OBJ_DATA_FILE>./delay25.dat</CALIB_OBJ_DATA_FILE>"                             << std::endl ;
  *outstream << "   <CALIB_TYPE>delay25</CALIB_TYPE>"                                                     << std::endl ;
  *outstream << "  </DATA>"                                                                               << std::endl ;
  *outstream << " "                                                                                       << std::endl ;
}

//=============================================================================================
void PixelDelay25Calib::writeXMLTrailer(std::ofstream *outstream,
                             	     	std::ofstream *out1stream,
                             	     	std::ofstream *out2stream ) const 
{
  std::string mthn = "[PixelDelay25Calib::writeXMLTrailer()]\t\t\t    " ;
  
  *outstream << " </DATA_SET>"		 								  << std::endl ;
  *outstream << "</ROOT>"  		 								  << std::endl ;
  
  outstream->close() ;
  std::cout  << __LINE__ << "]\t" << mthn << "Data written "   						  << std::endl ;

}


