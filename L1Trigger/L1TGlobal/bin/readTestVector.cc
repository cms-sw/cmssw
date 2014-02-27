#include <iostream>
#include <iomanip>
#include <fstream>
#include <memory>
#include <map>

#include "L1Trigger/L1TGlobal/src/L1TMenuEditor/L1TriggerMenu.hxx"

// The default XML file to load.
#define XML_OUTPUT "sample.xml"

const int MAX_ALGO_BITS = 512;

std::vector<int> l1taccepts;
std::vector<std::string> l1tnames;

void parseAlgo( std::string algo );


// The application.
int main( int argc, char** argv ){
  
  std::string vector_file = "";
  if( argc > 1 ) vector_file = argv[1];

  bool readXML = true;
  std::string xml_file = XML_OUTPUT;
  if( argc > 2) xml_file = argv[2];
  else readXML = false;

  l1taccepts.resize(512);
  for( unsigned int i=0; i<l1taccepts.size(); i++ ) l1taccepts[i] = 0;

  if( readXML ){
    // Load XML.
    std::auto_ptr<l1t::L1TriggerMenu> tm(l1t::l1TriggerMenu(xml_file));

    l1tnames.resize(512);
    l1t::AlgorithmList algorithms = tm->algorithms();
    for( l1t::AlgorithmList::algorithm_const_iterator i = algorithms.algorithm().begin();
         i != algorithms.algorithm().end(); ++i ){

      l1t::Algorithm algorithm = (*i);

      int index = algorithm.index();
      std::string name = algorithm.name();

      l1tnames[index] = name;
    }
  }

  std::ifstream file(vector_file);
  std::string line;

  std::vector<std::string> mu;
  std::vector<std::string> eg;
  std::vector<std::string> tau;
  std::vector<std::string> jet;
  mu.resize(8);
  eg.resize(12);
  tau.resize(8);
  jet.resize(12);

  std::string bx, et, ht, etm, htm, ex, alg, fin;
    
  int evt=0;
  int l1a=0;
  while(std::getline(file, line)){
    evt++;
    std::stringstream linestream(line);
    linestream >> bx 
	       >> mu[0] >> mu[1] >> mu[2] >> mu[3] >> mu[4] >> mu[5] >> mu[6] >> mu[7] 
	       >> eg[0] >> eg[1] >> eg[2] >> eg[3] >> eg[4] >> eg[5] >> eg[6] >> eg[7] >> eg[8] >> eg[9] >> eg[10] >> eg[11] 
	       >> tau[0] >> tau[1] >> tau[2] >> tau[3] >> tau[4] >> tau[5] >> tau[6] >> tau[7] 
	       >> jet[0] >> jet[1] >> jet[2] >> jet[3] >> jet[4] >> jet[5] >> jet[6] >> jet[7] >> jet[8] >> jet[9] >> jet[10] >> jet[11] 
	       >> et >> ht >> etm >> htm 
	       >> ex >> alg >> fin;

    parseAlgo(alg);

    int finOR = atoi( fin.c_str() );
    l1a += finOR;
  }

  printf("\t==== Summary of results ===\n");
  printf(" There were %d L1A out of %d events (%.1f%%)", l1a, evt, float(l1a)/float(evt)*100);
  printf("\n\n Triggers with nono-zero accepts");
  if( readXML && l1tnames.size()>0 ) printf("\n\t bit\t L1A\t Name\n");
  else                               printf("\n\t bit\t L1A\n");

  for( int i=0; i<MAX_ALGO_BITS; i++ ){
    if( l1taccepts[i]>0 ){
      if( readXML && l1tnames.size()>0 ) printf("\t %d \t %d \t %s\n", i, l1taccepts[i], l1tnames[i].c_str());
      else                               printf("\t %d \t %d\n", i, l1taccepts[i] );
    }
  }

  return 0;
}


// conversions.
void parseAlgo( std::string algo ){

  std::stringstream ss;
  std::string s;
  int pos, algRes, accept;
  
  for( int i=0; i<MAX_ALGO_BITS; i++ ){
    if( i>17 ) continue;
    pos = 127 - int( i/4 );
    std::string in(1,algo[pos]);
    char* endPtr = (char*) in.c_str();
    algRes = strtol( in.c_str(), &endPtr, 16 );
    accept = ( algRes >> (i%4) ) & 1;
    l1taccepts[i] += accept;
  }

  return;

}


// eof

