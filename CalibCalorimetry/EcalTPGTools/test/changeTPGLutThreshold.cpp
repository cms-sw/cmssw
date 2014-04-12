#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>

int main (int argc, char* argv[])
{
  if (argc != 3) {
    std::cout << "Usage:" << std::endl;
    std::cout << "  " << argv[0] << " <input TPG.txt filename> <threshold in ADC count> " << std::endl;
    exit(-1);
  }

  std::string filename = argv[1] ;
  std::string s_threshold = argv[2] ;
  int threshold = atoi(s_threshold.c_str()) ;

  if (threshold<0 || threshold>1023) {
    std::cerr << "your threshold "<<std::dec<<threshold<< " must be in the range [0,1024]" << std::endl ;
  }

  std::cout << "Producing TPG_threshold_" << threshold << ".txt with LUT threshold = " << threshold << std::endl ;

  std::string dataCard ;
  std::ifstream infile ; 
  //int id;
  int data;
  std::stringstream oufilename ;
  oufilename << "TPG_threshold_" << threshold << ".txt" ;
  std::ofstream oufile(oufilename.str().c_str()) ; 
  std::string str ;

  infile.open(filename.c_str()) ;

  if (infile.is_open()) {
    while (!infile.eof()) {

      getline (infile,str) ;

      int pos = int(str.find("LUT")) ;
      if (pos == 0) {
	oufile << str << std::endl ;
	for (int i=0 ; i <1024 ; i++) {
	  infile >> std::hex >> data ;
	  if (i <= threshold) data = 0 ;
	  if (i < 1023) oufile << "0x" << std::hex << data << " " ;
	  else oufile << "0x" << std::hex << data ;
	}
      }
      else oufile << str << std::endl ;
      
    } 
  }

  return 0;
}
