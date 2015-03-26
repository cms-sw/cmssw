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

#include "TH1.h"
#include "TFile.h"

const int MAX_ALGO_BITS = 512;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::string zeroes128 = std::string(128,'0');
std::string zeroes16 = std::string(16,'0');
std::string zeroes8  = std::string(8,'0');

std::vector<int> l1taccepts;
std::vector<std::string> l1tnames;
std::vector<int> l1taccepts_evt;

void parseMuons( std::vector<std::string> muons, bool verbose );
void parseEGs(   std::vector<std::string> egs, bool verbose );
void parseTaus(  std::vector<std::string> taus, bool verbose );
void parseJets(  std::vector<std::string> jets, bool verbose );
void parseEtSums(std::vector<std::string> etsums, bool verbose );

void parseAlgo( std::string algo );

l1t::Muon unpackMuons( std::string imu );
l1t::EGamma unpackEGs( std::string ieg );
l1t::Tau unpackTaus( std::string itau );
l1t::Jet unpackJets( std::string ijet );
l1t::EtSum unpackEtSums( std::string ietsum, l1t::EtSum::EtSumType type );

double convertPtFromHW( int hwPt, double max, double step );
double convertEtaFromHW( int hwEta, double max, double step, int hwMax );
double convertPhiFromHW( int hwPhi, double step );


TH1D* h_l1mu_pt_;
TH1D* h_l1mu_eta_;
TH1D* h_l1mu_phi_;
TH1D* h_l1mu_charge_;
TH1D* h_l1mu_quality_;
TH1D* h_l1mu_isolation_;
TH1D* h_l1mu_num_;

TH1D* h_l1jet_pt_;
TH1D* h_l1jet_eta_;
TH1D* h_l1jet_phi_;
TH1D* h_l1jet_num_;

TH1D* h_l1eg_pt_;
TH1D* h_l1eg_eta_;
TH1D* h_l1eg_phi_;
TH1D* h_l1eg_num_;

TH1D* h_l1tau_pt_;
TH1D* h_l1tau_eta_;
TH1D* h_l1tau_phi_;
TH1D* h_l1tau_num_;

TH1D* h_l1ht_;
TH1D* h_l1et_;
TH1D* h_l1htm_et_;
TH1D* h_l1etm_et_;
TH1D* h_l1htm_phi_;
TH1D* h_l1etm_phi_;


double MaxLepPt_ = 255;
double MaxJetPt_ = 1023;
double MaxEt_ = 2047;

double MaxCaloEta_ = 5.0;
double MaxMuonEta_ = 2.45;

double PhiStepCalo_ = 144;
double PhiStepMuon_ = 576;

double EtaStepCalo_ = 230;
double EtaStepMuon_ = 450;

double PtStep_ = 0.5;


// The application.
int main( int argc, char** argv ){
  using namespace boost;
  namespace po = boost::program_options;

  std::string vector_file;
  std::string xml_file;
  std::string histo_file;
  bool dumpEvents;
  int maxEvents;

  po::options_description desc("Main options");
  desc.add_options()
    ("vector_file,i", po::value<std::string>(&vector_file)->default_value(""), "Input file")
    ("menu_file,m",   po::value<std::string>(&xml_file)->default_value(""), "Menu file")
    ("hist_file,o", po::value<std::string>(&histo_file)->default_value(""), "Output histogram file")
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

  bool output = true;
  if( histo_file=="" ){
    output = false;
  }

  TFile* histofile = NULL;
  if( output ){
    histofile = new TFile(histo_file.c_str(),"RECREATE");
    histofile->cd();
  }

  l1taccepts.resize(MAX_ALGO_BITS);
  l1taccepts_evt.resize(MAX_ALGO_BITS);
  for( unsigned int i=0; i<l1taccepts.size(); i++ ) l1taccepts[i] = 0;

  if( readXML ){
    // Load XML.
    std::auto_ptr<l1t::L1TriggerMenu> tm(l1t::l1TriggerMenu(xml_file));

    l1tnames.resize(MAX_ALGO_BITS);
    l1t::AlgorithmList algorithms = tm->algorithms();
    for( l1t::AlgorithmList::algorithm_const_iterator i = algorithms.algorithm().begin();
         i != algorithms.algorithm().end(); ++i ){

      l1t::Algorithm algorithm = (*i);

      int index = algorithm.index();
      std::string name = algorithm.name();

      l1tnames[index] = name;
    }
  }

  /// Setup histograms
  h_l1mu_pt_  = new TH1D("h_l1mu_pt", ";L1 #mu p_{T}", int((MaxLepPt_+PtStep_)/(PtStep_) + 1.001), 0, MaxLepPt_+PtStep_ );
  h_l1mu_eta_ = new TH1D("h_l1mu_eta",";L1 #mu #eta",  int(EtaStepMuon_/2+0.0001), -MaxMuonEta_, MaxMuonEta_ );
  h_l1mu_phi_ = new TH1D("h_l1mu_phi",";L1 #mu #phi",  PhiStepMuon_+1, 0, 2*M_PI );
  h_l1mu_charge_ = new TH1D("h_l1mu_charge_",";L1 #mu charge",  2, 0, 2 );
  h_l1mu_quality_ = new TH1D("h_l1mu_quality_",";L1 #mu quality",  16, 0, 16 );
  h_l1mu_isolation_ = new TH1D("h_l1mu_isolation_",";L1 #mu isolation",  4, 0, 4 );
  h_l1mu_num_ = new TH1D("h_l1mu_num",";L1 Number of #mu",  10, 0, 10 );

  h_l1jet_pt_  = new TH1D("h_l1jet_pt", ";L1 jet p_{T}", int((MaxJetPt_+PtStep_)/(4*PtStep_) + 1.001), 0, MaxJetPt_+PtStep_ );
  h_l1jet_eta_ = new TH1D("h_l1jet_eta",";L1 jet #eta",  int(EtaStepCalo_/2+0.0001), -MaxCaloEta_, MaxCaloEta_ );
  h_l1jet_phi_ = new TH1D("h_l1jet_phi",";L1 jet #phi",  PhiStepCalo_+1, 0, 2*M_PI );
  h_l1jet_num_ = new TH1D("h_l1jet_num",";L1 Number of jets",  13, 0, 13 );

  h_l1eg_pt_  = new TH1D("h_l1eg_pt", ";L1 EG p_{T}", int((MaxLepPt_+PtStep_)/(PtStep_) + 1.001), 0, MaxLepPt_+PtStep_ );
  h_l1eg_eta_ = new TH1D("h_l1eg_eta",";L1 EG #eta",  int(EtaStepCalo_/2+0.0001), -MaxCaloEta_, MaxCaloEta_ );
  h_l1eg_phi_ = new TH1D("h_l1eg_phi",";L1 EG #phi",  PhiStepCalo_+1, 0, 2*M_PI );
  h_l1eg_num_ = new TH1D("h_l1eg_num",";L1 Number of EGs",  13, 0, 13 );

  h_l1tau_pt_  = new TH1D("h_l1tau_pt", ";L1 #tau p_{T}", int((MaxLepPt_+PtStep_)/(PtStep_) + 1.001), 0, MaxLepPt_+PtStep_ );
  h_l1tau_eta_ = new TH1D("h_l1tau_eta",";L1 #tau #eta",  int(EtaStepCalo_/2+0.0001), -MaxCaloEta_, MaxCaloEta_ );
  h_l1tau_phi_ = new TH1D("h_l1tau_phi",";L1 #tau #phi",  PhiStepCalo_+1, 0, 2*M_PI );
  h_l1tau_num_ = new TH1D("h_l1tau_num",";L1 Number of #tau",  13, 0, 13 );

  h_l1ht_ = new TH1D("h_l1ht_", ";L1 #SigmaH_{T}", int((MaxEt_+PtStep_)/(16*PtStep_) + 1.001), 0, MaxEt_+PtStep_ );
  h_l1et_ = new TH1D("h_l1et_", ";L1 #SigmaE_{T}", int((MaxEt_+PtStep_)/(16*PtStep_) + 1.001), 0, MaxEt_+PtStep_ );
  h_l1htm_et_ = new TH1D("h_l1htm_et_", ";L1 Missing H_{T}", int((MaxEt_+PtStep_)/(16*PtStep_) + 1.001), 0, MaxEt_+PtStep_ );
  h_l1etm_et_ = new TH1D("h_l1etm_et_", ";L1 Missing E_{T}", int((MaxEt_+PtStep_)/(16*PtStep_) + 1.001), 0, MaxEt_+PtStep_ );
  h_l1htm_phi_ = new TH1D("h_l1htm_phi_", ";L1 Missing H_{T} #phi", PhiStepCalo_+1, 0, 2*M_PI );
  h_l1etm_phi_ = new TH1D("h_l1etm_phi_", ";L1 Missing E_{T} #phi", PhiStepCalo_+1, 0, 2*M_PI );


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
    std::stringstream linestream(line);
    linestream >> bx 
	       >> mu[0] >> mu[1] >> mu[2] >> mu[3] >> mu[4] >> mu[5] >> mu[6] >> mu[7] 
	       >> eg[0] >> eg[1] >> eg[2] >> eg[3] >> eg[4] >> eg[5] >> eg[6] >> eg[7] >> eg[8] >> eg[9] >> eg[10] >> eg[11] 
	       >> tau[0] >> tau[1] >> tau[2] >> tau[3] >> tau[4] >> tau[5] >> tau[6] >> tau[7] 
	       >> jet[0] >> jet[1] >> jet[2] >> jet[3] >> jet[4] >> jet[5] >> jet[6] >> jet[7] >> jet[8] >> jet[9] >> jet[10] >> jet[11] 
	       >> etsum[0] >> etsum[1] >> etsum[2] >> etsum[3] 
	       >> ex >> alg >> fin;

    if( dumpEvents ) printf("  <==== BX = %s ====>\n",bx.c_str());
    parseMuons(mu, dumpEvents);
    parseEGs(eg, dumpEvents);
    parseTaus(tau, dumpEvents);
    parseJets(jet, dumpEvents);
    parseEtSums(etsum, dumpEvents);

    parseAlgo(alg);

    int finOR = atoi( fin.c_str() );
    l1a += finOR;

    if( dumpEvents ){
      printf("    == Algos ==\n");
      if( finOR ){
	printf(" Triggers with non-zero accepts\n");
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

    if( evt==maxEvents ) break;
  }

  printf(" =========== Summary of results ==========\n");
  printf(" There were %d L1A out of %d events (%.1f%%)\n", l1a, evt, float(l1a)/float(evt)*100);
  printf("\n Triggers with non-zero accepts\n");
  if( readXML && l1tnames.size()>0 ) printf("\t bit\t L1A\t Name\n");
  else                               printf("\t bit\t L1A\n");

  for( int i=0; i<MAX_ALGO_BITS; i++ ){
    if( l1taccepts[i]>0 ){
      if( readXML && l1tnames.size()>0 ) printf("\t %d \t %d \t %s\n", i, l1taccepts[i], l1tnames[i].c_str());
      else                               printf("\t %d \t %d\n", i, l1taccepts[i] );
    }
  }


  if( output ){
    histofile->Write();
    histofile->Close();
  }


  return 0;
}


// Parse Objects
void parseMuons( std::vector<std::string> muons, bool verbose ){

  if( verbose) printf("    == Muons ==\n");
  int nmu=0;
  for( unsigned int i=0; i<muons.size(); i++ ){
    std::string imu = muons[i];
    if( imu==zeroes16 ) continue;
    nmu++;
    l1t::Muon mu = unpackMuons( imu );

    double pt = convertPtFromHW( mu.hwPt(), MaxLepPt_, PtStep_ );
    double eta = convertEtaFromHW( mu.hwEta(), MaxMuonEta_, EtaStepMuon_, 0x1ff );
    double phi = convertPhiFromHW( mu.hwPhi(), PhiStepMuon_ );

    int iso = mu.hwIso();
    int qual = mu.hwQual();
    int charge = mu.hwCharge();
    int chargeValid = mu.hwChargeValid();

    h_l1mu_pt_->Fill( pt );
    h_l1mu_eta_->Fill( eta );
    h_l1mu_phi_->Fill( phi );
    h_l1mu_charge_->Fill( charge );

    h_l1mu_quality_->Fill( qual );
    h_l1mu_isolation_->Fill( iso );

    if( verbose) printf(" l1t::Muon %d:\t pt = %d (%.1f),\t eta = %d (%+.2f),\t phi = %d (%.2f),\t iso = %d,\t qual = %d,\t charge = %d,\t chargeValid = %d\n", i, mu.hwPt(), pt, mu.hwEta(), eta, mu.hwPhi(), phi, iso, qual, charge, chargeValid);
  }
  h_l1mu_num_->Fill(nmu);

  return;
}
void parseEGs( std::vector<std::string> egs, bool verbose ){

  if( verbose) printf("    == EGammas ==\n");
  int neg=0;
  for( unsigned int i=0; i<egs.size(); i++ ){
    std::string ieg = egs[i];
    if( ieg==zeroes8 ) continue;
    neg++;
    l1t::EGamma eg = unpackEGs( ieg );

    double pt = convertPtFromHW( eg.hwPt(), MaxLepPt_, PtStep_ );
    double eta = convertEtaFromHW( eg.hwEta(), MaxCaloEta_, EtaStepCalo_, 0xff );
    double phi = convertPhiFromHW( eg.hwPhi(), PhiStepCalo_ );

    h_l1eg_pt_->Fill( pt );
    h_l1eg_eta_->Fill( eta );
    h_l1eg_phi_->Fill( phi );

    if( verbose) printf(" l1t::EGamma %d:\t pt = %d (%.1f),\t eta = %d (%+.2f),\t phi = %d (%.2f)\n", i, eg.hwPt(), pt, eg.hwEta(), eta, eg.hwPhi(), phi);
  }
  h_l1eg_num_->Fill(neg);

  return;
}
void parseTaus( std::vector<std::string> taus, bool verbose ){

  if( verbose) printf("    == Taus ==\n");
  int ntau=0;
  for( unsigned int i=0; i<taus.size(); i++ ){
    std::string itau = taus[i];
    if( itau==zeroes8 ) continue;
    ntau++;
    l1t::Tau tau = unpackTaus( itau );

    double pt = convertPtFromHW( tau.hwPt(), MaxLepPt_, PtStep_ );
    double eta = convertEtaFromHW( tau.hwEta(), MaxCaloEta_, EtaStepCalo_, 0xff );
    double phi = convertPhiFromHW( tau.hwPhi(), PhiStepCalo_ );

    h_l1tau_pt_->Fill( pt );
    h_l1tau_eta_->Fill( eta );
    h_l1tau_phi_->Fill( phi );

    if( verbose) printf(" l1t::Tau %d:\t pt = %d (%.1f),\t eta = %d (%+.2f),\t phi = %d (%.2f)\n", i, tau.hwPt(), pt, tau.hwEta(), eta, tau.hwPhi(), phi);
  }
  h_l1tau_num_->Fill(ntau);

  return;
}
void parseJets( std::vector<std::string> jets, bool verbose ){

  if( verbose) printf("    == Jets ==\n");
  int njet=0;
  for( unsigned int i=0; i<jets.size(); i++ ){
    std::string ijet = jets[i];
    if( ijet==zeroes8 ) continue;
    njet++;
    l1t::Jet jet = unpackJets( ijet );

    double pt = convertPtFromHW( jet.hwPt(), MaxJetPt_, PtStep_ );
    double eta = convertEtaFromHW( jet.hwEta(), MaxCaloEta_, EtaStepCalo_, 0xff );
    double phi = convertPhiFromHW( jet.hwPhi(), PhiStepCalo_ );

    h_l1jet_pt_->Fill( pt );
    h_l1jet_eta_->Fill( eta );
    h_l1jet_phi_->Fill( phi );

    if( verbose) printf(" l1t::Jet %d:\t pt = %d (%.1f),\t eta = %d (%+.2f),\t phi = %d (%.2f)\n", i, jet.hwPt(), pt, jet.hwEta(), eta, jet.hwPhi(), phi);
  }
  h_l1jet_num_->Fill(njet);

  return;
}
void parseEtSums( std::vector<std::string> etsum, bool verbose ){

  if( verbose) printf("    == EtSums ==\n");

  //et sum
  std::string iet = etsum[0];
  if( iet!=zeroes8 ){
    l1t::EtSum et = unpackEtSums( iet, l1t::EtSum::EtSumType::kTotalEt );
    double pt = convertPtFromHW( et.hwPt(), MaxEt_, PtStep_ );
    h_l1et_->Fill( pt );
    if( verbose) printf(" l1t::EtSum TotalEt:\t Et = %d\n", et.hwPt());
  }
  //ht sum
  std::string iht = etsum[1];
  if( iht!=zeroes8 ){
    l1t::EtSum ht = unpackEtSums( iht, l1t::EtSum::EtSumType::kTotalHt );
    double pt = convertPtFromHW( ht.hwPt(), MaxEt_, PtStep_ );
    h_l1ht_->Fill( pt );
    if( verbose) printf(" l1t::EtSum TotalHt:\t Ht = %d\n", ht.hwPt());
  }
  //etm
  std::string ietm = etsum[2];
  if( ietm!=zeroes8 ){
    l1t::EtSum etm = unpackEtSums( ietm, l1t::EtSum::EtSumType::kMissingEt );
    double pt = convertPtFromHW( etm.hwPt(), MaxEt_, PtStep_ );
    double phi = convertPhiFromHW( etm.hwPhi(), PhiStepCalo_ );
    h_l1etm_et_->Fill( pt );
    h_l1etm_phi_->Fill( phi );
    if( verbose) printf(" l1t::EtSum MissingEt:\t Et = %d,\t phi = %d\n", etm.hwPt(), etm.hwPhi());
  }
  //htm
  std::string ihtm = etsum[3];
  if( ihtm!=zeroes8 ){
    l1t::EtSum htm = unpackEtSums( ihtm, l1t::EtSum::EtSumType::kMissingHt );
    double pt = convertPtFromHW( htm.hwPt(), MaxEt_, PtStep_ );
    double phi = convertPhiFromHW( htm.hwPhi(), PhiStepCalo_ );
    h_l1htm_et_->Fill( pt );
    h_l1htm_phi_->Fill( phi );
    if( verbose) printf(" l1t::EtSum MissingHt:\t Et = %d,\t phi = %d\n", htm.hwPt(), htm.hwPhi());
  }

  return;
}


// Parse Algos
void parseAlgo( std::string algo ){

  // clear the algo per event
  l1taccepts_evt.clear();
  l1taccepts_evt.resize(MAX_ALGO_BITS);

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

  int pt  = (packedVal>>10) & 0x1ff;
  int eta = (packedVal>>23) & 0x1ff;
  int phi = (packedVal>>0)  & 0x3ff;
  int iso = (packedVal>>32) & 0x3;
  int qual= (packedVal>>19) & 0xf;
  int charge = (packedVal>>34) & 0x1;
  int chargeValid = (packedVal>>35) & 0x1;
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

// Conversion into physical coordinates from HW
double convertPtFromHW( int hwPt, double max, double step ){
  double pt = double(hwPt) * step;
  if( pt>max ) pt = max;
  return pt;
}

double convertEtaFromHW( int hwEta, double max, double step, int hwMax ){
  hwMax++;
  double binWidth = 2*max/step;
  double eta = ( hwEta<int(hwMax/2+1.001) ) ? double(hwEta)*binWidth+0.5*binWidth : -(double(hwMax-hwEta)*binWidth -0.5*binWidth);
  return eta;
}
double convertPhiFromHW( int hwPhi, double step ){
  double phi = double(hwPhi)/step * 2 * M_PI;
  return phi;
}


// eof

