#include <iostream>
#include <iomanip>
#include <fstream>
#include <memory>
#include <map>

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "L1Trigger/L1TGlobal/src/L1TMenuEditor/L1TriggerMenu.hxx"

#include "FWCore/Utilities/interface/typedefs.h"

#include <boost/program_options.hpp>

// The default XML file to load.
#define XML_OUTPUT "sample.xml"

const int MAX_ALGO_BITS = 512;

std::string zeroes128 = std::string(128,'0');
std::string zeroes16 = std::string(16,'0');
std::string zeroes8  = std::string(8,'0');

std::vector<int> l1taccepts;
std::vector<std::string> l1tnames;
std::vector<int> l1taccepts_evt;

void parseMuons( std::vector<std::string> muons );
void parseEGs(   std::vector<std::string> egs );
void parseTaus(  std::vector<std::string> taus );
void parseJets(  std::vector<std::string> jets );
void parseEtSums(std::vector<std::string> etsums );

void parseAlgo( std::string algo );

l1t::Muon unpackMuons( std::string imu );
l1t::EGamma unpackEGs( std::string ieg );
l1t::Tau unpackTaus( std::string itau );
l1t::Jet unpackJets( std::string ijet );
l1t::EtSum unpackEtSums( std::string ietsum, l1t::EtSum::EtSumType type );



// The application.
int main( int argc, char** argv ){
  using namespace boost;
  namespace po = boost::program_options;

  std::string vector_file;// = "";
  std::string xml_file;// = XML_OUTPUT;
  bool dumpEvents;
  int maxEvents;

  po::options_description desc("Main options");
  desc.add_options()
    ("vector_file,i", po::value<std::string>(&vector_file)->default_value(""), "Input file")
    ("menu_file,m",   po::value<std::string>(&xml_file)->default_value(""), "Menu file")
    ("dumpEvents,d",  po::value<bool>(&dumpEvents)->default_value(false), "Dump event-by-event information")
    ("maxEvents,n",  po::value<int>(&maxEvents)->default_value(-1), "Number of events (default is all)")
    ("help,h", "Produce help message")
    ;

  po::variables_map vm, vm0;

  // parse the first time, using only common options and allow unregistered options 
  try{
    po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm0);
    po::notify(vm0);
  } catch(std::exception &ex) {
    std::cout << "Invalid options: " << ex.what() << std::endl;
    std::cout << "Use readTestVector --help to get a list of all the allowed options"  << std::endl;
    return 999;
  } catch(...) {
    std::cout << "Unidentified error parsing options." << std::endl;
    return 1000;
  }

  // if help, print help
  if(vm0.count("help")) {
    std::cout << "Usage: readTestVector [options]\n";
    std::cout << desc;
    return 0;
  }

  if( vector_file=="" ){
    std::cout << "No input file specified" << std::endl;
    return 99;
  }

  bool readXML = true;
  if( xml_file=="" ){
    readXML = false;
    std::cout << "No menu file specified" << std::endl;
  }


  l1taccepts.resize(512);
  l1taccepts_evt.resize(512);
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
  std::vector<std::string> etsum;
  mu.resize(8);
  eg.resize(12);
  tau.resize(8);
  jet.resize(12);
  etsum.resize(4);

  std::string bx, ex, alg, fin;

  int evt=0;
  int l1a=0;

  if( dumpEvents ) printf(" ==== Objects ===\n");

  while(std::getline(file, line)){
    evt++;
    if( evt==maxEvents ) break;
    std::stringstream linestream(line);
    linestream >> bx 
	       >> mu[0] >> mu[1] >> mu[2] >> mu[3] >> mu[4] >> mu[5] >> mu[6] >> mu[7] 
	       >> eg[0] >> eg[1] >> eg[2] >> eg[3] >> eg[4] >> eg[5] >> eg[6] >> eg[7] >> eg[8] >> eg[9] >> eg[10] >> eg[11] 
	       >> tau[0] >> tau[1] >> tau[2] >> tau[3] >> tau[4] >> tau[5] >> tau[6] >> tau[7] 
	       >> jet[0] >> jet[1] >> jet[2] >> jet[3] >> jet[4] >> jet[5] >> jet[6] >> jet[7] >> jet[8] >> jet[9] >> jet[10] >> jet[11] 
	       >> etsum[0] >> etsum[1] >> etsum[2] >> etsum[3] 
	       >> ex >> alg >> fin;

    if( dumpEvents ){
      printf("  <==== BX = %s ====>\n",bx.c_str());
      parseMuons(mu);
      parseEGs(eg);
      parseTaus(tau);
      parseJets(jet);
      parseEtSums(etsum);
    }

    parseAlgo(alg);

    int finOR = atoi( fin.c_str() );
    l1a += finOR;

    if( dumpEvents ){
      printf("    == Algos ==\n");
      if( finOR ){
	printf(" Triggers with nono-zero accepts\n");
	if( readXML && l1tnames.size()>0 ) printf("\t bit\t L1A\t Name\n");
	else                               printf("\t bit\t L1A\n");

	for( int i=0; i<MAX_ALGO_BITS; i++ ){
	  if( l1taccepts_evt[i]>0 ){
	    if( readXML && l1tnames.size()>0 ) printf("\t %d \t %d \t %s\n", i, l1taccepts_evt[i], l1tnames[i].c_str());
	    else                               printf("\t %d \t %d\n", i, l1taccepts_evt[i] );
	  }
	}
      }
      else{
	printf("\n No triggers passed\n");
      }
      // extra spacing between bx
      printf("\n\n");
    }

  }

  printf(" =========== Summary of results ==========\n");
  printf(" There were %d L1A out of %d events (%.1f%%)\n", l1a, evt, float(l1a)/float(evt)*100);
  printf("\n Triggers with nono-zero accepts\n");
  if( readXML && l1tnames.size()>0 ) printf("\t bit\t L1A\t Name\n");
  else                               printf("\t bit\t L1A\n");

  for( int i=0; i<MAX_ALGO_BITS; i++ ){
    if( l1taccepts[i]>0 ){
      if( readXML && l1tnames.size()>0 ) printf("\t %d \t %d \t %s\n", i, l1taccepts[i], l1tnames[i].c_str());
      else                               printf("\t %d \t %d\n", i, l1taccepts[i] );
    }
  }

  return 0;
}


// Parse Objects
void parseMuons( std::vector<std::string> muons ){

  printf("    == Muons ==\n");
  for( unsigned int i=0; i<muons.size(); i++ ){
    std::string imu = muons[i];
    if( imu==zeroes16 ) continue;

    l1t::Muon mu = unpackMuons( imu );

    printf(" l1t::Muon %d:\t pt = %d,\t eta = %d,\t phi = %d\n", i, mu.hwPt(), mu.hwEta(), mu.hwPhi());
  }
  return;
}
void parseEGs( std::vector<std::string> egs ){

  printf("    == EGammas ==\n");
  for( unsigned int i=0; i<egs.size(); i++ ){
    std::string ieg = egs[i];
    if( ieg==zeroes8 ) continue;

    l1t::EGamma eg = unpackEGs( ieg );

    printf(" l1t::EGamma %d:\t pt = %d,\t eta = %d,\t phi = %d\n", i, eg.hwPt(), eg.hwEta(), eg.hwPhi());
  }
  return;
}
void parseTaus( std::vector<std::string> taus ){

  printf("    == Taus ==\n");
  for( unsigned int i=0; i<taus.size(); i++ ){
    std::string itau = taus[i];
    if( itau==zeroes8 ) continue;

    l1t::Tau tau = unpackTaus( itau );

    printf(" l1t::Tau %d:\t pt = %d,\t eta = %d,\t phi = %d\n", i, tau.hwPt(), tau.hwEta(), tau.hwPhi());
  }
  return;
}
void parseJets( std::vector<std::string> jets ){

  printf("    == Jets ==\n");
  for( unsigned int i=0; i<jets.size(); i++ ){
    std::string ijet = jets[i];
    if( ijet==zeroes8 ) continue;

    l1t::Jet jet = unpackJets( ijet );

    printf(" l1t::Jet %d:\t pt = %d,\t eta = %d,\t phi = %d\n", i, jet.hwPt(), jet.hwEta(), jet.hwPhi());
  }
  return;
}
void parseEtSums( std::vector<std::string> etsum ){

  printf("    == EtSums ==\n");

  //et sum
  std::string iet = etsum[0];
  if( iet!=zeroes8 ){
    l1t::EtSum et = unpackEtSums( iet, l1t::EtSum::EtSumType::kTotalEt );
    printf(" l1t::EtSum TotalEt:\t Et = %d\n", et.hwPt());
  }
  //ht sum
  std::string iht = etsum[1];
  if( iht!=zeroes8 ){
    l1t::EtSum ht = unpackEtSums( iht, l1t::EtSum::EtSumType::kTotalHt );
    printf(" l1t::EtSum TotalHt:\t Ht = %d\n", ht.hwPt());
  }
  //etm
  std::string ietm = etsum[2];
  if( ietm!=zeroes8 ){
    l1t::EtSum etm = unpackEtSums( ietm, l1t::EtSum::EtSumType::kMissingEt );
    printf(" l1t::EtSum MissingEt:\t Et = %d,\t phi = %d\n", etm.hwPt(), etm.hwPhi());
  }
  //htm
  std::string ihtm = etsum[3];
  if( ihtm!=zeroes8 ){
    l1t::EtSum htm = unpackEtSums( ihtm, l1t::EtSum::EtSumType::kMissingHt );
    printf(" l1t::EtSum MissingHt:\t Et = %d,\t phi = %d\n", htm.hwPt(), htm.hwPhi());
  }

  return;
}


// Parse Algos
void parseAlgo( std::string algo ){

  // clear the algo per event
  l1taccepts_evt.clear();
  l1taccepts_evt.resize(512);

  // Return if no triggers fired
  if( algo==zeroes128 ) return;

  std::stringstream ss;
  std::string s;
  int pos, algRes, accept;
  
  for( int i=0; i<MAX_ALGO_BITS; i++ ){
    pos = 127 - int( i/4 );
    std::string in(1,algo[pos]);
    char* endPtr = (char*) in.c_str();
    algRes = strtol( in.c_str(), &endPtr, 16 );
    accept = ( algRes >> (i%4) ) & 1;
    l1taccepts_evt[i] += accept;
    l1taccepts[i] += accept;
  }

  return;

}

// Unpack Objects
l1t::Muon unpackMuons( std::string imu ){

  char* endPtr = (char*) imu.c_str();
  cms_uint64_t packedVal = strtoull( imu.c_str(), &endPtr, 16 );

  int pt  = (packedVal>>0)  & 0x1ff;
  int eta = (packedVal>>9)  & 0x1ff;
  int phi = (packedVal>>18) & 0x3ff;
  int iso = (packedVal>>34) & 0x1;
  int qual= (packedVal>>30) & 0x1;
  int charge = (packedVal>>29) & 0x1;
  int chargeValid = (packedVal>>28) & 0x1;
  int mip = 1;
  int tag = 1;

  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();
  l1t::Muon mu(*p4, pt, eta, phi, qual, charge, chargeValid, iso, mip, tag);

  return mu;

}
l1t::EGamma unpackEGs( std::string ieg ){

  char* endPtr = (char*) ieg.c_str();
  unsigned int packedVal = strtol( ieg.c_str(), &endPtr, 16 );

  int pt  = (packedVal>>0)  & 0x1ff;
  int eta = (packedVal>>9)  & 0xff;
  int phi = (packedVal>>17) & 0xff;
  int iso = (packedVal>>25) & 0x1;
  int qual= (packedVal>>26) & 0x1;

  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();
  l1t::EGamma eg(*p4, pt, eta, phi, qual, iso);

  return eg;
}
l1t::Tau unpackTaus( std::string itau ){

  char* endPtr = (char*) itau.c_str();
  unsigned int packedVal = strtol( itau.c_str(), &endPtr, 16 );

  int pt  = (packedVal>>0)  & 0x1ff;
  int eta = (packedVal>>9)  & 0xff;
  int phi = (packedVal>>17) & 0xff;
  int iso = (packedVal>>25) & 0x1;
  int qual= (packedVal>>26) & 0x1;

  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();
  l1t::Tau tau(*p4, pt, eta, phi, qual, iso);

  return tau;
}

l1t::Jet unpackJets( std::string ijet ){

  char* endPtr = (char*) ijet.c_str();
  unsigned int packedVal = strtol( ijet.c_str(), &endPtr, 16 );

  int pt  = (packedVal>>0)  & 0x7ff;
  int eta = (packedVal>>11) & 0xff;
  int phi = (packedVal>>19) & 0xff;
  int qual= (packedVal>>27) & 0x1;

  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();
  l1t::Jet jet(*p4, pt, eta, phi, qual);

  return jet;
}

l1t::EtSum unpackEtSums( std::string ietsum, l1t::EtSum::EtSumType type ){

  char* endPtr = (char*) ietsum.c_str();
  unsigned int packedVal = strtol( ietsum.c_str(), &endPtr, 16 );

  int pt  = (packedVal>>0)  & 0xfff;
  int phi = 0;
  if( type==l1t::EtSum::EtSumType::kMissingEt ||
      type==l1t::EtSum::EtSumType::kMissingHt ) phi = (packedVal>>12) & 0xff;
  
  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();
  l1t::EtSum sum(*p4, type, pt, 0,phi, 0); 

  return sum;
}




// eof

