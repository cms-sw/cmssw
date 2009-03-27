#include "CalibFormats/SiPixelObjects/interface/PixelFEDTestDAC.h"
#include <string.h>
#include <cassert>
#include <map>
#include <sstream>

using namespace std;

using namespace pos;

PixelFEDTestDAC::PixelFEDTestDAC(std::vector<std::vector<std::string> > & tableMat ){

  const unsigned long int UB=200;
  const unsigned long int B=500;
  const unsigned long int offset=0;
  vector <unsigned int> pulseTrain(256), pixelDCol(1), pixelPxl(2), pixelTBMHeader(3), pixelTBMTrailer(3);
  unsigned int DCol, LorR, start=15;
  std::string line;
  std::string::size_type loc1, loc2, loc3, loc4;
  unsigned long int npos=std::string::npos;
  int i;

  std::map<std::string , int > colM;
  std::vector<std::string > colNames;
  /**

     View's name: CONF_KEY_PIXEL_CALIB_MV
     CONFIG_KEY_ID                             NOT NULL NUMBER(38)
     CONFG_KEY                                 NOT NULL VARCHAR2(80)
     VERSION                                            VARCHAR2(40)
     KIND_OF_COND                              NOT NULL VARCHAR2(40)
     RUN_TYPE                                           VARCHAR2(40)
     RUN_NUMBER                                         NUMBER(38)
     CALIB_OBJ_DATA_FILE                       NOT NULL VARCHAR2(200)
     CALIB_OBJ_DATA_CLOB                       NOT NULL CLOB
  */

  colNames.push_back("CONFIG_KEY_ID"      );
  colNames.push_back("CONFG_KEY"          );
  colNames.push_back("VERSION"            );
  colNames.push_back("KIND_OF_COND"       );
  colNames.push_back("RUN_TYPE"           );
  colNames.push_back("RUN_NUMBER"         );
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
	  std::cerr << "[PixelMaxVsf::PixelMaxVsf()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
	  assert(0);
	}
    }

  
  std::istringstream fin ;
  fin.str(tableMat[1][colM["CALIB_OBJ_DATA_CLOB"]]) ;
  
  // Initialise the pulseTrain to offset+black
  for (unsigned int i=0;i<pulseTrain.size();++i)
    {
      pulseTrain[i]=offset+B;
    }

  i=start;
  
  getline(fin, line);
  mode_=line;
  assert(mode_=="EmulatedPhysics"||
         mode_=="FEDBaselineWithTestDACs"||
         mode_=="FEDAddressLevelWithTestDACs");

  while (!fin.eof())
    {
      getline(fin, line);
		
      if (line.find("TBMHeader")!=npos)
	{
	  loc1=line.find("("); if (loc1==npos) {cout<<"'(' not found after TBMHeader.\n"; break;}
	  loc2=line.find(")", loc1+1); if (loc2==npos) {cout<<"')' not found after TBMHeader.\n"; break;}
	  int TBMHeader=atoi(line.substr(loc1+1,loc2-loc1-1).c_str());
	  
	  pulseTrain[i]=UB;++i;
	  pulseTrain[i]=UB;++i;
	  pulseTrain[i]=UB;++i;
	  pulseTrain[i]=B;++i;
	  
	  pixelTBMHeader=decimalToBaseX(TBMHeader, 4, 4);

	  pulseTrain[i]=levelEncoder(pixelTBMHeader[3]);++i;
	  pulseTrain[i]=levelEncoder(pixelTBMHeader[2]);++i;
	  pulseTrain[i]=levelEncoder(pixelTBMHeader[1]);++i;
	  pulseTrain[i]=levelEncoder(pixelTBMHeader[0]);++i;
	}
      else if (line.find("ROCHeader")!=std::string::npos)
	{
	  loc1=line.find("("); if (loc1==npos) {cout<<"'(' not found after ROCHeader.\n"; break;}
	  loc2=line.find(")", loc1+1); if (loc2==npos) {cout<<"')' not found after ROCHeader.\n"; break;}
	  int LastDAC=atoi(line.substr(loc1+1,loc2-loc1-1).c_str());

          std::cout<<"--------------"<<std::endl;
	  
	  pulseTrain[i]=UB;++i;
	  pulseTrain[i]=B;++i;
	  pulseTrain[i]=levelEncoder(LastDAC); ++i;
	}
      else if (line.find("PixelHit")!=std::string::npos) {

	loc1=line.find("("); if (loc1==npos) {cout<<"'(' not found after PixelHit.\n"; break;}
	loc2=line.find(",", loc1+1); if (loc2==npos) {cout<<"',' not found after the first argument of PixelHit.\n"; break;}
	loc3=line.find(",", loc2+1); if (loc3==npos) {cout<<"'.' not found after the second argument of PixelHit.\n"; break;}
	loc4=line.find(")", loc3+1); if (loc4==npos) {cout<<"')' not found after the third argument of PixelHit.\n"; break;}
	int column=atoi(line.substr(loc1+1, loc2-loc1-1).c_str());
	int row=atoi(line.substr(loc2+1, loc3-loc2-1).c_str());
	int charge=atoi(line.substr(loc3+1, loc4-loc3-1).c_str());
	
	DCol=int(column/2);
	LorR=int(column-DCol*2);
	pixelDCol=decimalToBaseX(DCol, 6, 2);
	pixelPxl=decimalToBaseX((80-row)*2+LorR, 6, 3);

        std::cout<<"Pxl = "<<pixelPxl[2]<<pixelPxl[1]<<pixelPxl[0]<<", DCol= "<<pixelDCol[1]<<pixelDCol[0]<<std::endl;
	
	pulseTrain[i]=levelEncoder(pixelDCol[1]);++i;
	pulseTrain[i]=levelEncoder(pixelDCol[0]);++i;
	pulseTrain[i]=levelEncoder(pixelPxl[2]);++i;
	pulseTrain[i]=levelEncoder(pixelPxl[1]);++i;
	pulseTrain[i]=levelEncoder(pixelPxl[0]);++i;
	pulseTrain[i]=charge;++i;
			
      }
      else if (line.find("TBMTrailer")!=std::string::npos)
	{
	  loc1=line.find("("); if (loc1==npos) {cout<<"'(' not found after TBMTrailer.\n"; break;}
	  loc2=line.find(")", loc1+1); if (loc2==npos) {cout<<"')' not found after TBMTrailer.\n"; break;}
	  int TBMTrailer=atoi(line.substr(loc1+1,loc2-loc1-1).c_str());
	  
	  pulseTrain[i]=UB;++i;
	  pulseTrain[i]=UB;++i;
	  pulseTrain[i]=B; ++i;
	  pulseTrain[i]=B; ++i;
	  
	  pixelTBMTrailer=decimalToBaseX(TBMTrailer, 4, 4);
	  pulseTrain[i]=levelEncoder(pixelTBMTrailer[3]);++i;
	  pulseTrain[i]=levelEncoder(pixelTBMTrailer[2]);++i;
	  pulseTrain[i]=levelEncoder(pixelTBMTrailer[1]);++i;
	  pulseTrain[i]=levelEncoder(pixelTBMTrailer[0]);++i;
	}
    }
//   fin.close();
  dacs_=pulseTrain;
}


PixelFEDTestDAC::PixelFEDTestDAC(std::string filename){

  const unsigned long int UB=200;
  const unsigned long int B=500;
  const unsigned long int offset=0;
  vector <unsigned int> pulseTrain(256), pixelDCol(1), pixelPxl(2), pixelTBMHeader(3), pixelTBMTrailer(3);
  unsigned int DCol, LorR, start=15;
  std::string line;
  std::string::size_type loc1, loc2, loc3, loc4;
  unsigned long int npos=std::string::npos;
  int i;
  
  // Initialise the pulseTrain to offset+black
  for (unsigned int i=0;i<pulseTrain.size();++i)
    {
      pulseTrain[i]=offset+B;
    }
  
  ifstream fin(filename.c_str());

  i=start;

  getline(fin, line);
  mode_=line;
  assert(mode_=="EmulatedPhysics"||
         mode_=="FEDBaselineWithTestDACs"||
         mode_=="FEDAddressLevelWithTestDACs");

  while (!fin.eof())
    {
      getline(fin, line);
		
      if (line.find("TBMHeader")!=npos)
	{
	  loc1=line.find("("); if (loc1==npos) {cout<<"'(' not found after TBMHeader.\n"; break;}
	  loc2=line.find(")", loc1+1); if (loc2==npos) {cout<<"')' not found after TBMHeader.\n"; break;}
	  int TBMHeader=atoi(line.substr(loc1+1,loc2-loc1-1).c_str());
	  
	  pulseTrain[i]=UB;++i;
	  pulseTrain[i]=UB;++i;
	  pulseTrain[i]=UB;++i;
	  pulseTrain[i]=B;++i;
	  
	  pixelTBMHeader=decimalToBaseX(TBMHeader, 4, 4);

	  pulseTrain[i]=levelEncoder(pixelTBMHeader[3]);++i;
	  pulseTrain[i]=levelEncoder(pixelTBMHeader[2]);++i;
	  pulseTrain[i]=levelEncoder(pixelTBMHeader[1]);++i;
	  pulseTrain[i]=levelEncoder(pixelTBMHeader[0]);++i;
	}
      else if (line.find("ROCHeader")!=std::string::npos)
	{
	  loc1=line.find("("); if (loc1==npos) {cout<<"'(' not found after ROCHeader.\n"; break;}
	  loc2=line.find(")", loc1+1); if (loc2==npos) {cout<<"')' not found after ROCHeader.\n"; break;}
	  int LastDAC=atoi(line.substr(loc1+1,loc2-loc1-1).c_str());

          std::cout<<"--------------"<<std::endl;
	  
	  pulseTrain[i]=UB;++i;
	  pulseTrain[i]=B;++i;
	  pulseTrain[i]=levelEncoder(LastDAC); ++i;
	}
      else if (line.find("PixelHit")!=std::string::npos) {

	loc1=line.find("("); if (loc1==npos) {cout<<"'(' not found after PixelHit.\n"; break;}
	loc2=line.find(",", loc1+1); if (loc2==npos) {cout<<"',' not found after the first argument of PixelHit.\n"; break;}
	loc3=line.find(",", loc2+1); if (loc3==npos) {cout<<"'.' not found after the second argument of PixelHit.\n"; break;}
	loc4=line.find(")", loc3+1); if (loc4==npos) {cout<<"')' not found after the third argument of PixelHit.\n"; break;}
	int column=atoi(line.substr(loc1+1, loc2-loc1-1).c_str());
	int row=atoi(line.substr(loc2+1, loc3-loc2-1).c_str());
	int charge=atoi(line.substr(loc3+1, loc4-loc3-1).c_str());
	
	DCol=int(column/2);
	LorR=int(column-DCol*2);
	pixelDCol=decimalToBaseX(DCol, 6, 2);
	pixelPxl=decimalToBaseX((80-row)*2+LorR, 6, 3);

        std::cout<<"Pxl = "<<pixelPxl[2]<<pixelPxl[1]<<pixelPxl[0]<<", DCol= "<<pixelDCol[1]<<pixelDCol[0]<<std::endl;
	
	pulseTrain[i]=levelEncoder(pixelDCol[1]);++i;
	pulseTrain[i]=levelEncoder(pixelDCol[0]);++i;
	pulseTrain[i]=levelEncoder(pixelPxl[2]);++i;
	pulseTrain[i]=levelEncoder(pixelPxl[1]);++i;
	pulseTrain[i]=levelEncoder(pixelPxl[0]);++i;
	pulseTrain[i]=charge;++i;
			
      }
      else if (line.find("TBMTrailer")!=std::string::npos)
	{
	  loc1=line.find("("); if (loc1==npos) {cout<<"'(' not found after TBMTrailer.\n"; break;}
	  loc2=line.find(")", loc1+1); if (loc2==npos) {cout<<"')' not found after TBMTrailer.\n"; break;}
	  int TBMTrailer=atoi(line.substr(loc1+1,loc2-loc1-1).c_str());
	  
	  pulseTrain[i]=UB;++i;
	  pulseTrain[i]=UB;++i;
	  pulseTrain[i]=B; ++i;
	  pulseTrain[i]=B; ++i;
	  
	  pixelTBMTrailer=decimalToBaseX(TBMTrailer, 4, 4);
	  pulseTrain[i]=levelEncoder(pixelTBMTrailer[3]);++i;
	  pulseTrain[i]=levelEncoder(pixelTBMTrailer[2]);++i;
	  pulseTrain[i]=levelEncoder(pixelTBMTrailer[1]);++i;
	  pulseTrain[i]=levelEncoder(pixelTBMTrailer[0]);++i;
	}
    }
  fin.close();
  dacs_=pulseTrain;
}


unsigned int PixelFEDTestDAC::levelEncoder(int level){

  unsigned int pulse;
  
  switch (level)
    {
    case 0: pulse=450; break;
    case 1: pulse=500; break;
    case 2: pulse=550; break;
    case 3: pulse=600; break;
    case 4: pulse=650; break;
    case 5: pulse=700; break;
    default: assert(0); break;
    }
  
  return pulse;

}


vector<unsigned int> PixelFEDTestDAC::decimalToBaseX (unsigned int a, unsigned int x, unsigned int length){

  vector<unsigned int> ans(100,0);
  int i=0;
  
  while (a>0)
    {
      ans[i]=a%x;
      //ans.push_back(a%x);
      a=a/x;
      i+=1;
    }
  
  if (length>0) ans.resize(length); else ans.resize(i);
  
  return ans;
}

