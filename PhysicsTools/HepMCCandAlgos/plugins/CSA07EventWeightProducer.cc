/** \class CSA07EventWeightProducer 
 *
 * \author Filip Moortgat & Paolo Bartalini
 *
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "SimDataFormats/HepMCProduct/interface/AlpgenInfoProduct.h"
#include "FWCore/Framework/interface/Run.h"
#include <vector>

using namespace std;
namespace edm { class ParameterSet; }
namespace HepMC { class GenParticle; class GenEvent; }

class CSA07EventWeightProducer : public edm::EDProducer {
public:
  /// constructor
  CSA07EventWeightProducer( const edm::ParameterSet & );

private:
  void produce( edm::Event& evt, const edm::EventSetup& es );
  edm::InputTag src_;
  bool verbose;
  double overallLumi;
  double ttKfactor;
  // methods needed deal with alpgen
  // hardcoded xsecs and generated yields
  int FindIndex(int njet, double pT);
  vector<double> WCrossSection;
  vector<double> ZCrossSection;
  vector<double> WNevents;
  vector<double> ZNevents;
  vector<double> TTbarCrossSection;
  vector<double> TTbarNevents;
  
  vector<double> GetWCrossSection();
  vector<double> GetZCrossSection();
  vector<double> GetWNevents();
  vector<double> GetZNevents();
  vector<double> GetTTbarCrossSection();
  vector<double> GetTTbarNevents();
  
};

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/HepMCProduct/interface/GenInfoProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
using namespace edm;
using namespace std;
using namespace HepMC;

#include "DataFormats/Candidate/interface/Candidate.h"
using namespace reco;

CSA07EventWeightProducer::CSA07EventWeightProducer( const ParameterSet & p ) :
  src_( p.getParameter<InputTag>( "src" ) ), verbose ( p.getUntrackedParameter<bool> ("talkToMe", false)),
  overallLumi (p.getParameter<double> ("overallLumi")) ,
  ttKfactor (p.getParameter<double> ("ttKfactor")) {
  produces<double>("weight");
  produces<int>("AlpgenProcessID");

  // Read Alpgen Xsections and generated yields
  WCrossSection = GetWCrossSection();
  ZCrossSection = GetZCrossSection();
  TTbarCrossSection = GetTTbarCrossSection();
  WNevents = GetWNevents();
  ZNevents = GetZNevents();
  TTbarNevents = GetTTbarNevents();

}


void CSA07EventWeightProducer::produce( Event& evt, const EventSetup& es ) {

  /*
    Handle<HepMCProduct> mc;
    evt.getByLabel( src_, mc );
    const GenEvent * genEvt = mc->GetEvent();
    if( genEvt == 0 ) 
    throw edm::Exception( edm::errors::InvalidReference) << "HepMC has null pointer to GenEvent" << endl;
    Handle<GenInfoProduct> gi;
    evt.getRun().getByLabel( src_, gi);
  
    double processID = genEvt->signal_process_id();
    double pthat = genEvt->event_scale(); 
    double cross_section = gi->external_cross_section(); // is the one written in the cfg file -- units is pb-1!!
    double filter_eff = gi->filter_efficiency();

  */

  Handle<int> genProcessID;
  evt.getByLabel( "genEventProcID", genProcessID );
  double processID = *genProcessID;
 
  Handle<double> genEventScale;
  evt.getByLabel( "genEventScale", genEventScale );
  double pthat = *genEventScale;
 
 
  double filter_eff = -99.;
  double cross_section = -99.;
 
 
  if (processID != 4) {
   
    Handle<double> genFilterEff;
    evt.getByLabel( "genEventRunInfo", "FilterEfficiency", genFilterEff);
    filter_eff = *genFilterEff;
   
    Handle<double> genCrossSect;
    evt.getByLabel( "genEventRunInfo", "PreCalculatedCrossSection", genCrossSect); 
    cross_section = *genCrossSect;
    // for the cross section calculated at the end of every run: "genEventRunInfo:AutoCrossSection"
   
  } 
 
  // initialize ALPGEN procees id to -1, i.e. no ALPGEN event
  // the code will return 
  // 1000 + jet multiplicity for W+jets
  // 2000 + jet multiplicity for Z+jets
  // 3000 + jet multiplicity for ttbar
  auto_ptr<int> ALPGENid( new int(-1) );

  // initialize weight to write out 
  auto_ptr<double> weight( new double(1) );

  // for calculating below we assume 1 pb-1 and we rescale later
 
  // the event weight is defined as the effective cross section of a certain process divided by the produced number of events of a this process
 
  if  (processID != 4){  // the Pythia events (for ALPGEN see below)
    // min bias (diffractive part)
    if ( (filter_eff == 1.) && ( processID == 92 || processID == 93 || processID == 94 || processID == 95 )) {
      (*weight) = 25E+9 / 6.25E+06 ; // number = cross section of these processes (25mb), in pb-1, div by 0.31X20M = 6.25M events
    }
   
    // qcd (including min bias HS)
    if ((filter_eff == 1. || filter_eff == 0.964) && (processID == 11 || processID == 12 || processID == 13 || processID == 28 || processID == 68 || processID == 53)) {
  
      if (pthat > 0 && pthat < 15) { (*weight) = 53.0E+9 / (13.75E+06 * 53.0 / 55.0 + 0.75E+06); } //number = cross section in 1 pb-1 div by #events (MB HS+ QCD bin)
 
      if (pthat > 15 && pthat < 20) { (*weight) =  1.46E+9 / (13.75E+06 * 1.46 / 55.0 + 1.3E+06); } //number = cross section in 1 pb-1 div by #events (MB HS + QCD bin)

      if (pthat > 20 && pthat < 30) { (*weight) =  0.63E+9 / (13.75E+06 * 0.63 / 55.0 + 2.5E+06); } //number = cross section in 1 pb-1 div by #events (MB HS + QCD bin)

      if (pthat > 30 && pthat < 50) { (*weight) =  0.163E+9 / (13.75E+06 * 0.163 / 55.0 + 2.5E+06); } //number = cross section in 1 pb-1 div by #events (MB HS + QCD bin)
     
      if (pthat > 50 && pthat < 80) { (*weight) =  21.6E+06 / (13.75E+06 * 0.0216 / 55.0 + 2.5E+06); } //number = cross section in 1 pb-1 div by #events (MB HS + QCD bin)
     
      if (pthat > 80 && pthat < 120) { (*weight) =  3.08E+06 / (13.75E+06 * 0.00308 / 55.0 + 1.18E+06); } //number = cross section in 1 pb-1 div by #events (MB HS + QCD bin)
     
      if (pthat > 120 && pthat < 170) { (*weight) =  0.494E+06 / (1.25E+06); } //number = cross section in 1 pb-1 div by #events (QCD bin). MB negligible
     
      if (pthat > 170 && pthat < 230) { (*weight) =  0.101E+06 / (1.16E+06); } //number = cross section in 1 pb-1 div by #events (QCD bin). MB negligible
     
      if (pthat > 230 && pthat < 300) { (*weight) =  24.5E+03 / (1.20E+06); } //number = cross section in 1 pb-1 div by #events (QCD bin). MB negligible
     
      if (pthat > 300 && pthat < 380) { (*weight) =  6.24E+03 / (1.18E+06); } //number = cross section in 1 pb-1 div by #events (QCD bin). MB negligible
     
      if (pthat > 380 && pthat < 470) { (*weight) =  1.78E+03 / (1.19E+06); } //number = cross section in 1 pb-1 div by #events (QCD bin). MB negligible
     
      if (pthat > 470 && pthat < 600) { (*weight) =  0.683E+03 / (1.23E+06); } //number = cross section in 1 pb-1 div by #events (QCD bin). MB negligible
     
      if (pthat > 600 && pthat < 800) { (*weight) =  0.204E+03 / (0.5E+06); } //number = cross section in 1 pb-1 div by #events (QCD bin). MB negligible
     
      if (pthat > 800 && pthat < 1000) { (*weight) =  35.1E+00 / (0.1E+06); } //number = cross section in 1 pb-1 div by #events (QCD bin). MB negligible
     
      if (pthat > 1000 && pthat < 1400) { (*weight) =  10.9E+00 / (3.0E+04); } //number = cross section in 1 pb-1 div by #events (QCD bin). MB negligible
     
      if (pthat > 1400 && pthat < 1800) { (*weight) =  1.6E+00 / (3.0E+04); } //number = cross section in 1 pb-1 div by #events (QCD bin). MB negligible
     
      if (pthat > 1800 && pthat < 2200) { (*weight) =  0.145E+00 / (2.0E+04); } //number = cross section in 1 pb-1 div by #events (QCD bin). MB negligible
     
      if (pthat > 2200 && pthat < 2600) { (*weight) =  23.8E-03 / (1.0E+04); } //number = cross section in 1 pb-1 div by #events (QCD bin). MB negligible
     
      if (pthat > 2600 && pthat < 3000) { (*weight) =  4.29E-03 / (1.0E+04); } //number = cross section in 1 pb-1 div by #events (QCD bin). MB negligible
     
      if (pthat > 3000 && pthat < 3500) { (*weight) =  0.844E-03 / (1.0E+04); } //number = cross section in 1 pb-1 div by #events (QCD bin). MB negligible
     
      if (pthat > 3500) { (*weight) =  0.108E-03 / (1.0E+04); } //number = cross section in 1 pb-1 div by #events (QCD bin). MB negligible
    
    }
  
    // muon enriched
    if ( (filter_eff == 0.0008) && (processID == 11 || processID == 12 || processID == 13 || processID == 28 || processID == 68 || processID == 53 || processID == 95)) {
  
      (*weight) = cross_section * filter_eff / 20.E+06;  // number  = effective cross section for 1 pb-1 div by 20M 
  
    } 
 
    // electron enriched (weights calculated neglecting duplication from bbbar -> e, see below)
    if ( (filter_eff == 0.0097) && (processID == 11 || processID == 12 || processID == 13 || processID == 28 || processID == 68 || processID == 53 || processID == 95)) {
  
      (*weight) = cross_section * filter_eff / 8.7E+06 ;  // number  = effective cross section in 1 pb-1 div by 10M 
  
    } 
   
    // electron enriched from bbbar 5 < pt_hat < 50
    if ( (filter_eff == 0.00019) && (processID == 11 || processID == 12 || processID == 13 || processID == 28 || processID == 68 || processID == 53 || processID == 95)) {
  
      (*weight) = cross_section * filter_eff / 3.E+06 ;  // number  = effective cross section in 1 pb-1 div by 3M 
  
    } 
  
    // electron enriched from bbbar 50 < pt_hat < 170
    if ( (filter_eff == 0.0068) && (processID == 11 || processID == 12 || processID == 13 || processID == 28 || processID == 68 || processID == 53 || processID == 95)) {
  
      (*weight) = cross_section * filter_eff / 3.E+06 ;  // number  = effective cross section in 1 pb-1 div by 3M 
  
    } 
  
    // electron enriched from bbbar 170 < pt_hat
    if ( (filter_eff == 0.0195) && (processID == 11 || processID == 12 || processID == 13 || processID == 28 || processID == 68 || processID == 53 || processID == 95)) {
  
      (*weight) = cross_section * filter_eff / 2.6E+06 ;  // number  = effective cross section in 1 pb-1 div by 3M 
  
    } 
 
  
    // photon + jets
    if ( processID == 14 || processID == 18 || processID == 29 ) {
  
      if (pthat > 0 && pthat < 15) { (*weight) =  cross_section / 0.3E+06;}

      if (pthat > 15 && pthat < 20) { (*weight) =  cross_section / (0.52E+06); } 

      if (pthat > 20 && pthat < 30) { (*weight) =  cross_section / (0.6E+06); } 
  
      if (pthat > 30 && pthat < 50) { (*weight) =  cross_section / (0.51E+06); } 

      if (pthat > 50 && pthat < 80) { (*weight) =  cross_section / (0.52E+06); } 

      if (pthat > 80 && pthat < 120) { (*weight) =  cross_section / (0.53E+06); } 

      if (pthat > 120 && pthat < 170) { (*weight) =  cross_section / (0.56E+06); } 

      if (pthat > 170 && pthat < 300) { (*weight) =  cross_section / (2.0E+05); } 

      if (pthat > 300 && pthat < 500) { (*weight) =  cross_section / (3.0E+04); } 

      if (pthat > 500 && pthat < 7000) { (*weight) =  cross_section / (3.0E+04); } 

    }
  
    // Drell-Yan (Pythia branching ratio Z->leptons = 0.107)
    if ((filter_eff == 1.000) && (processID == 1 || processID == 15 || processID == 19 || processID == 30 || processID == 35 || processID == 141)) {
  
      (*weight) = cross_section * filter_eff * 0.107 / 3.E+06 ;  // number  = effective cross section in 1 pb-1 div by 3M 
  
    } 

    // CharmOnia (Pythia branching ratio J/Psi->2muons = 0.06)
    if ((processID > 420 && processID < 440)) {
  
      (*weight) = cross_section * filter_eff * 0.06 / 1.E+06 ;  // number  = effective cross section in 1 pb-1 div by 1M 
  
    } 

    // BottomOnia (Pythia branching ratio Y->2muons = 0.025)
    if ((processID > 460 && processID < 480)) {
  
      (*weight) = cross_section * filter_eff * 0.025 / 1.E+06 ;  // number  = effective cross section in 1 pb-1 div by 1M 
  
    } 
   
    // B -> J/Psi (branching ratio J/Psi->2muons = 0.06)
    // average "weighted" branching ratio pp -> b_hadron -> J/Psi X = 0.1 (guess, to be x-checked)
    if ((filter_eff == 0.00013)) {
  
      (*weight) = cross_section * filter_eff * 0.06 * 0.1 / 0.5E+06 ;  // number  = effective cross section in 1 pb-1 div by 0.5M 
  
    } 

    // top secret 
    if (processID == 102 || processID == 123 || processID == 124) {
      (*weight) = cross_section * filter_eff / 45200 ;   
    }

    if (processID == 141) {
      (*weight) = cross_section * filter_eff / 12300 ;   
    }
    


  }  // ALPGEN
  else if(processID == 4) { // this is the number for external ALPGEN events

    Handle<CandidateCollection> genParticles;
    evt.getByLabel( "genParticleCandidates", genParticles );    
    int id_process = -1; // 0 -> W+jets; 1-> Z+jets; 2->ttbar +jets
    double pT = 0.;
    const Candidate * mother = NULL;
    // first loop: which process?
    for( size_t i = 0; i < genParticles->size(); ++ i ) {
      const Candidate & p = (*genParticles)[ i ];
      int id = p.pdgId();
      int st = p.status();  
      const Candidate * mom = p.mother();
      // W+jets
      if(st == 3 && (id == 24 || id == -24) ) {
	mother = mom;
	id_process = 0;
	pT = p.pt();
	i = genParticles->size()-1; // get out of the loop
      }
      // Z+jets
      if(st == 3 && (id == 23 || id == -23) ) {
	mother = mom;
	id_process = 1;
	pT = p.pt();
	i = genParticles->size()-1; // get out of the loop       
      }
      // tt+jets
      if(st == 3 && (id == 6 || id == -6) ) {
	mother = mom;
	id_process = 2;
	i = genParticles->size()-1; // get out of the loop       
      }
    }
    // second loop: Find out  jet multiplicity and get the weight
    int njet = 0;
    for( size_t i = 0; i < genParticles->size(); ++ i ) {
      const Candidate & p = (*genParticles)[ i ];
      const Candidate * mom = p.mother();
      if (mom == mother) njet++;
    }
    

    if(id_process == 2) {
      njet += -2; // take out the two tops from the counting
      (*weight) = TTbarCrossSection[njet]*ttKfactor/TTbarNevents[njet];
    } else if(id_process == 0) {
      njet += -1;// take out the vector boson
      int indSample = FindIndex(njet,pT);
      // factor 3 for 3 lepton flavours?
      (*weight) = 3 * WCrossSection[indSample]/WNevents[indSample];
    } else if(id_process == 1) {
      njet += -1;// take out the vector boson
      int indSample = FindIndex(njet,pT);
      // factor 3 for 3 lepton flavours?
      (*weight) = 3 * ZCrossSection[indSample]/ZNevents[indSample];
    } // should wemake it crash if the process is not found?

    (*ALPGENid) = (id_process+1)*1000+njet;

    if (verbose) {
      if(id_process == 0) { cout << " -- Process: W + " << njet << " jets (with boson Pt = " << pT << ")" << endl;}
      if(id_process == 1) { cout << " -- Process: Z + " << njet << " jets (with boson Pt = " << pT << ")" << endl;} 
      if(id_process == 2) { cout << " -- Process: tt + " << njet << " jets" << endl;}
    }
    
    ///////////////////////////////////////////////////////////
    //  The code below will be used for unscrewed production //
    ///////////////////////////////////////////////////////////

    /*    
    // Get the AlpgenInfoProduct
    Handle<AlpgenInfoProduct> alpHandle;
    evt.getByLabel( src_, alpHandle);
    AlpgenInfoProduct alpInfo = *alpHandle;
    //    const AlpgenInfoProduct * alpInfo = alp->GetEvent();
    int nParton = alpInfo.nTot()-2; // initial partons taken out
    int idP1 = alpInfo.lundOut(0);               // lund ID of the first FS parton
    int idPlast   = alpInfo.lundOut(nParton-1);  // lund ID of the last  FS parton
    int idPntlast = alpInfo.lundOut(nParton-2);  // lund ID of the second last  FS parton
    if(abs(idP1) == 6) { // this is a ttbar events
    int njet = nParton-6; // take out the partons of the two t -> W(lnu) b decay chain
    (*weight) = TTbarCrossSection[njet]/TTbarNevents[njet];
    } else if((abs(idPlast) == 11 && abs(idPntlast) == 12) || // W -> e      nu_e
    (abs(idPlast) == 12 && abs(idPntlast) == 11) || // W -> nu_e   e
    (abs(idPlast) == 13 && abs(idPntlast) == 14) || // W -> mu     nu_mu
    (abs(idPlast) == 14 && abs(idPntlast) == 13) || // W -> nu_mu  mu
    (abs(idPlast) == 15 && abs(idPntlast) == 16) || // W -> tau    nu_tau
    (abs(idPlast) == 16 && abs(idPntlast) == 15)){  // W -> nu_tau tau
    // look for jet multiplicity
    int njet = nParton-2; // take out the W decay products
    // calculate pT(W)
    double pT = sqrt(pow(alpInfo.pxOut(idPlast)+alpInfo.pxOut(idPntlast),2.)+
    pow(alpInfo.pyOut(idPlast)+alpInfo.pyOut(idPntlast),2.));
    int indSample = FindIndex(njet,pT);
    (*weight) = WCrossSection[indSample]/WNevents[indSample];
    } else if((abs(idPlast) == 11 && abs(idPntlast) == 11) || // Z -> e      e
    (abs(idPlast) == 12 && abs(idPntlast) == 12) || // Z -> nu_e   nu_e
    (abs(idPlast) == 13 && abs(idPntlast) == 13) || // Z -> mu     mu
    (abs(idPlast) == 14 && abs(idPntlast) == 14) || // Z -> nu_mu  mu_mu
    (abs(idPlast) == 15 && abs(idPntlast) == 15) || // Z -> tau    tau
    (abs(idPlast) == 16 && abs(idPntlast) == 16)){  // Z -> nu_tau nu_tau
    // look for jet multiplicity
    int njet = nParton-2; // take out the Z decay products
    double pT = sqrt(pow(alpInfo.pxOut(idPlast)+alpInfo.pxOut(idPntlast),2.)+
    pow(alpInfo.pyOut(idPlast)+alpInfo.pyOut(idPntlast),2.));
    int indSample = FindIndex(njet,pT);
    (*weight) = ZCrossSection[indSample]/ZNevents[indSample];
    }
    */

    ///////////////////////////////////////////////
    //  End of the code for unscrewed production //
    ///////////////////////////////////////////////

  } 
  
  // renormalize to lumi from cfg
  (*weight) = (*weight) * overallLumi ;
  
  if (verbose) {
    cout << " -- Event weight : " << (*weight) << endl; 
  }

  evt.put( weight, "weight");
  evt.put( ALPGENid, "AlpgenProcessID" );
}

int CSA07EventWeightProducer::FindIndex(int njet, double pT) {
  int ipT=-10;
  if(njet == 0)       ipT = 6;
  else {
    if(pT<100.)       ipT = 1;
    else if(pT<300.)  ipT = 2;
    else if(pT<800.)  ipT = 3;
    else if(pT<1600.) ipT = 4;
    else if(pT<3200.) ipT = 5;
    else if(pT<5000.) ipT = 6;
  }
  return (njet-1)*6 + ipT;
}

// to call only once
vector<double> CSA07EventWeightProducer::GetWCrossSection() {
  vector<double> vec;
  // W0jet
  vec.push_back(1.51E+04);
  // W1jet
  vec.push_back(3.08E+03);
  vec.push_back(8.55E+01);
  vec.push_back(9.72E-01);
  vec.push_back(5.29E-03);
  vec.push_back(4.46E-05);
  vec.push_back(1.91E-08);
  // W2jet
  vec.push_back(8.46E+02);
  vec.push_back(7.50E+01);
  vec.push_back(1.35E+00);
  vec.push_back(1.05E-02);
  vec.push_back(1.01E-04);
  vec.push_back(5.59E-08);
  // W3jet
  vec.push_back(1.96E+02);
  vec.push_back(3.58E+01);
  vec.push_back(1.02E+00);
  vec.push_back(1.00E-02);
  vec.push_back(1.16E-04);
  vec.push_back(6.63E-08);
  // W4jet
  vec.push_back(4.12E+01);
  vec.push_back(1.26E+01);
  vec.push_back(5.21E-01);
  vec.push_back(6.27E-03);
  vec.push_back(7.47E-05);
  vec.push_back(4.13E-08);
  // W5jet
  vec.push_back(2.81E+01);
  vec.push_back(1.32E+01);
  vec.push_back(1.01E+00);
  vec.push_back(1.97E-02);
  vec.push_back(2.48E-04);
  vec.push_back(1.10E-07);
  return vec;
}

vector<double> CSA07EventWeightProducer::GetZCrossSection() {
  vector<double> vec;
  // Z0jet
  vec.push_back(1.5E+03);
  // Z1jet
  vec.push_back(3.1E+02);
  vec.push_back(1.0E+01);
  vec.push_back(1.2E-01);
  vec.push_back(7.1E-04);
  vec.push_back(5.6E-06);
  vec.push_back(3.0E-09);
  // Z2jet
  vec.push_back(9.0E+01);
  vec.push_back(9.4E+00);
  vec.push_back(1.8E-01);
  vec.push_back(1.3E-03);
  vec.push_back(1.3E-05);
  vec.push_back(7.2E-09);
  // Z3jet
  vec.push_back(2.3E+01);
  vec.push_back(4.3E+00);
  vec.push_back(1.3E-01);
  vec.push_back(1.3E-03);
  vec.push_back(1.4E-05);
  vec.push_back(8.4E-09);
  // Z4jet
  vec.push_back(4.6E+00);
  vec.push_back(1.4E+00);
  vec.push_back(6.7E-02);
  vec.push_back(8.2E-04);
  vec.push_back(9.3E-06);
  vec.push_back(4.4E-09);
  // Z5jet
  vec.push_back(2.9E+00);
  vec.push_back(1.7E+00);
  vec.push_back(1.5E-01);
  vec.push_back(2.5E-03);
  vec.push_back(3.0E-05);
  vec.push_back(1.3E-08);
  return vec;
}

vector<double> CSA07EventWeightProducer::GetWNevents() {
  vector<double> vec;
  // W0jet
  vec.push_back(8796412);
  // W1jet
  vec.push_back(9088026);
  vec.push_back(247023);
  vec.push_back(2.9E+03);
  vec.push_back(1.6E+01);
  vec.push_back(1.3E-01);
  vec.push_back(5.7E-05);
  // W2jet
  vec.push_back(2380315);
  vec.push_back(287472);
  vec.push_back(4.0E+03);
  vec.push_back(3.2E+01);
  vec.push_back(3.0E-01);
  vec.push_back(1.7E-04);
  // W3jet
  vec.push_back(352855);
  vec.push_back(117608);
  vec.push_back(3.1E+03);
  vec.push_back(3.0E+01);
  vec.push_back(3.5E-01);
  vec.push_back(2.0E-04);
  // W4jet
  vec.push_back(125849);
  vec.push_back(39719);
  vec.push_back(1.6E+03);
  vec.push_back(1.9E+01);
  vec.push_back(2.2E-01);
  vec.push_back(1.2E-04);
  // W5jet
  vec.push_back(62238);
  vec.push_back(43865);
  vec.push_back(3.0E+03);
  vec.push_back(5.9E+01);
  vec.push_back(7.4E-01);
  vec.push_back(3.3E-04);  
  return vec;
}

vector<double> CSA07EventWeightProducer::GetZNevents() {
  vector<double> vec;
  // Z0jet
  vec.push_back(3251851);
  // Z1jet
  vec.push_back(944726);
  vec.push_back(36135);
  vec.push_back(3.6E+02);
  vec.push_back(2.1E+00);
  vec.push_back(1.7E-02);
  vec.push_back(9.0E-06);
  // Z2jet
  vec.push_back(289278);
  vec.push_back(35285);
  vec.push_back(5.5E+02);
  vec.push_back(4.0E+00);
  vec.push_back(4.0E-02);
  vec.push_back(2.2E-05);
  // Z3jet
  vec.push_back(73182);
  vec.push_back(24316);
  vec.push_back(4.0E+02);
  vec.push_back(3.8E+00);
  vec.push_back(4.1E-02);
  vec.push_back(2.5E-05);
  // Z4jet
  vec.push_back(33083);
  vec.push_back(6616);
  vec.push_back(2.0E+02);
  vec.push_back(2.5E+00);
  vec.push_back(2.8E-02);
  vec.push_back(1.3E-05);
  // Z5jet
  vec.push_back(12136);
  vec.push_back(5966);
  vec.push_back(4.5E+02);
  vec.push_back(7.4E+00);
  vec.push_back(9.1E-02);
  vec.push_back(3.9E-05);
  return vec;
}

vector<double> CSA07EventWeightProducer::GetTTbarCrossSection() {
  vector<double> vec;
  vec.push_back(334.5);
  vec.push_back(95.4);
  vec.push_back(18.2);
  vec.push_back(3.2);
  vec.push_back(0.8);
  return vec;
}

vector<double> CSA07EventWeightProducer::GetTTbarNevents() {
  vector<double> vec;
  vec.push_back(1456646);
  vec.push_back(361835);
  vec.push_back(81215);
  vec.push_back(14036);
  vec.push_back(5352);
  return vec;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( CSA07EventWeightProducer );
