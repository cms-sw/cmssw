// -*- C++ -*-
//
// Package:    Compare
// Class:      Compare
// 
/**\class Compare Compare.cc CalibMuon/Compare/src/Compare.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Thomas Nummy,Bld. 32 Room 4-C21,+41227671337,
//         Created:  Thu Oct 29 13:55:15 CET 2009
// $Id$
//
//


// system include files
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
// class decleration
//

class Compare : public edm::EDAnalyzer {
   public:
      explicit Compare(const edm::ParameterSet&);
      ~Compare();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
Compare::Compare(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
 /*


 */

  //=========== Compare Pedesdtals =================

  const int MAX_SIZE = 252288;

  int old_index;
  float old_ped, old_rms;
  std::vector<int> old_index_id;
  std::vector<float> old_peds;
  //std::vector<float> old_pedrms;
  int new_index;
  float new_ped,new_rms;
  std::vector<int> new_index_id;
  std::vector<float> new_peds;
  //std::vector<float> new_pedrms;
  std::vector<float> diff;
  //std::vector<float> myoldpeds;

  //int counter,counter1;
  //int old_nrlines=0;
  //int new_nrlines=0;

  std::ifstream olddata; 
  olddata.open("old_dbpeds.dat",std::ios::in); 
  if(!olddata) {
    std::cerr <<"Error: old_dbpeds.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!olddata.eof() ) { 
    olddata >> old_index >> old_ped >> old_rms ; 
    old_index_id.push_back(old_index);
    old_peds.push_back(old_ped);
    //old_pedrms.push_back(old_rms);
    //old_nrlines++;
  }
  olddata.close();

  std::ifstream newdata;
  std::ofstream myPedsFile("diffPedsTest.dat",std::ios::out);
  newdata.open("goodPeds2009_08_31_run112487.dat",std::ios::in); 
  if(!newdata) {
    std::cerr <<"Error: goodPeds2009_08_31_run112487.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!newdata.eof() ) { 
    newdata >> new_index >> new_ped >> new_rms ; 
    new_index_id.push_back(new_index);
    new_peds.push_back(new_ped);
    //new_pedrms.push_back(new_rms);
    //new_nrlines++;
  }
  newdata.close();
  diff.resize(MAX_SIZE);
  //myoldpeds.resize(MAX_SIZE); // is myoldpeds needed?
  
  for(int i=0; i<MAX_SIZE;++i){
    // counter=old_index_id[i];  //are counter and counter1 needed?
    //myoldpeds[i]=old_peds[i];

    for (unsigned int k=0;k<new_index_id.size()-1;++k){
      //counter1=new_index_id[k];
      if(old_index_id[i] == new_index_id[k]){
	diff[k]=old_peds[i] - new_peds[k];
	new_peds.erase(new_peds.begin());
	new_index_id.erase(new_index_id.begin());
	//std::cout<<old_peds[i]<<" new_peds[k]"<<new_peds[k]<<std::endl;
	myPedsFile<<old_index_id[i]<<"  "<<diff[k]<<std::endl;	
      }
    }
  }

  // ============= Comparing Crosstalk ===================

  old_index = 0;
  float old_xtalk_left, old_xtalk_right, old_int_left, old_int_right;
  old_index_id.clear();
  std::vector<float> old_Rxtalk;
  std::vector<float> old_Lxtalk;
  std::vector<float> old_Rint;
  std::vector<float> old_Lint;
  std::vector<float> diffXtalkR;
  std::vector<float> diffXtalkL;
  std::vector<float> diffIntR;
  std::vector<float> diffIntL;
  
  std::vector<float> myoldxtalkR;
  std::vector<float> myoldxtalkL;
  std::vector<float> myoldintR;
  std::vector<float> myoldintL;

  new_index = 0;
  float new_xtalk_left, new_xtalk_right, new_int_left, new_int_right;
  new_index_id.clear();
  std::vector<float> new_Rxtalk;
  std::vector<float> new_Lxtalk;
  std::vector<float> new_Rint;
  std::vector<float> new_Lint;

  int counter,counter1;

  std::ifstream olddata1; 
  olddata1.open("old_dbxtalk.dat",std::ios::in); 
  if(!olddata1) {
    std::cerr <<"Error: old_dbxtalk.dat -> no such file!"<< std::endl;
    exit(1);
  }

  while (!olddata1.eof() ) { 
    olddata1 >> old_index >> old_xtalk_left >> old_int_left >> old_xtalk_right >> old_int_right ;
    old_index_id.push_back(old_index);
    old_Rxtalk.push_back(old_xtalk_right);
    old_Rint.push_back(old_int_right); 
    old_Lxtalk.push_back(old_xtalk_left);  
    old_Lint.push_back(old_int_left);
  }
  olddata1.close();

  std::ifstream newdata1;
  std::ofstream myXtalkFile("diffXtalkTest.dat",std::ios::out);

  newdata1.open("goodXtalk2009_08_31_run112486.dat",std::ios::in); 
  if(!newdata1) {
    std::cerr <<"Error: goodXtalk2009_08_31_run112486.dat  -> no such file!"<< std::endl;
    exit(1);
  }

  while (!newdata1.eof() ) { 
    newdata1 >> new_index >> new_xtalk_left >> new_int_left >> new_xtalk_right >> new_int_right ;
    new_index_id.push_back(new_index);
    new_Rxtalk.push_back(new_xtalk_right);
    new_Rint.push_back(new_int_right); 
    new_Lxtalk.push_back(new_xtalk_left);
    new_Lint.push_back(new_int_left);
  }
  newdata1.close();  

  diffXtalkR.resize(MAX_SIZE);
  diffXtalkL.resize(MAX_SIZE);
  diffIntR.resize(MAX_SIZE);
  diffIntL.resize(MAX_SIZE);
  myoldxtalkR.resize(MAX_SIZE);
  myoldxtalkL.resize(MAX_SIZE);
  myoldintR.resize(MAX_SIZE);
  myoldintR.resize(MAX_SIZE);

  for(int i=0; i<MAX_SIZE;++i){
    counter=old_index_id[i];  
    myoldxtalkR[i]=old_Rxtalk[i];
    myoldxtalkL[i]=old_Lxtalk[i];
    myoldintR[i]=old_Rint[i];
    myoldintR[i]=old_Lint[i];

    for (unsigned int k=0;k<new_index_id.size()-1;k++){
      counter1=new_index_id[k];
      if(counter == counter1){
	diffXtalkR[k]=old_Rxtalk[i] - new_Rxtalk[k];
	diffXtalkL[k]=old_Lxtalk[i] - new_Lxtalk[k];
	diffIntR[k]=old_Rint[i] - new_Rint[k];
	diffIntL[k]=old_Rint[i] - new_Rint[k];

	//	std::cout<<counter<<" "<<counter1<<"  "<<old_Rxtalk[i]<<" new_Rxtalk[k] "<<new_Rxtalk[k]<<std::endl;
	myXtalkFile<<counter<<"  "<<diffXtalkL[k]<<"  "<<diffIntL[k]<<"  "<<diffXtalkR[k]<<"  "<<diffIntR[k]<<"  "<<std::endl;	
      }
    }
  }


// ================= Comapring Gains ===============

  old_index=0;
  float old_slope, old_int, old_chi2;
  old_index_id.clear();
  std::vector<float> old_gains;
  std::vector<float> old_intercept;
  std::vector<float> old_chi;

  new_index=0;
  float new_slope,new_int, new_chi2;
  new_index_id.clear();
  std::vector<float> new_gains;
  std::vector<float> new_intercept;
  std::vector<float> new_chi;
 
  diff.clear();
  std::vector<float> myoldgains;

  counter=0;
  counter1=0;
  int old_nrlines=0;
  int new_nrlines=0;

  std::ifstream olddata2; 
  olddata2.open("old_dbgains.dat",std::ios::in); 
  if(!olddata2) {
    std::cerr <<"Error: old_dbgains.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!olddata2.eof() ) { 
    olddata2 >> old_index >> old_slope; // >> old_int >> old_chi2 ; 
    old_index_id.push_back(old_index);
    old_gains.push_back(old_slope);
    //old_intercept.push_back(old_int);
    //old_chi.push_back(old_chi2);
    old_nrlines++;
  }
  olddata2.close();

  std::ifstream newdata2;
  std::ofstream myGainsFile("diffGainsTest.dat",std::ios::out);
  newdata2.open("goodGains2009_08_31_run112484.dat",std::ios::in); 
  if(!newdata2) {
    std::cerr <<"Error: goodGains2009_08_31_run112484.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!newdata2.eof() ) { 
    newdata2 >> new_index >> new_slope >> new_int >> new_chi2 ; 
    new_index_id.push_back(new_index);
    new_gains.push_back(new_slope);
    new_intercept.push_back(new_int);
    new_chi.push_back(new_chi2);
    new_nrlines++;
  }
  newdata2.close();
  diff.resize(MAX_SIZE);
  myoldgains.resize(MAX_SIZE);
  
  for(int i=0; i<MAX_SIZE;++i){
    counter=old_index_id[i];  
    myoldgains[i]=old_gains[i];

    for (int k=0;k<new_index_id.size()-1;k++){
      counter1=new_index_id[k];
      if(counter == counter1){
	diff[k]=old_gains[i] - new_gains[k];
	//std::cout<<old_gains[i]<<" new_gains[k]"<<new_gains[k]<<std::endl;
	myGainsFile<<counter<<"  "<<diff[k]<<std::endl;	
      }
    }
  }


//=================== Comparing Noise Matrix ===============

  old_index=0;
  float old_elem33, old_elem34,old_elem35,old_elem44,old_elem45,old_elem46,old_elem55,old_elem56;
  float old_elem57,old_elem66,old_elem67,old_elem77;
  old_index_id.clear();
  std::vector<float> old_el33;
  std::vector<float> old_el34;
  std::vector<float> old_el35;
  std::vector<float> old_el44;
  std::vector<float> old_el45;
  std::vector<float> old_el46;
  std::vector<float> old_el55;
  std::vector<float> old_el56;
  std::vector<float> old_el57;
  std::vector<float> old_el66;
  std::vector<float> old_el67;
  std::vector<float> old_el77;

  new_index=0;
  float new_elem33,new_elem34,new_elem35,new_elem44,new_elem45,new_elem46,new_elem55,new_elem56;
  float new_elem57,new_elem66,new_elem67,new_elem77;
  new_index_id.clear();
  std::vector<float> new_el33;
  std::vector<float> new_el34;
  std::vector<float> new_el35;
  std::vector<float> new_el44;
  std::vector<float> new_el45;
  std::vector<float> new_el46;
  std::vector<float> new_el55;
  std::vector<float> new_el56;
  std::vector<float> new_el57;
  std::vector<float> new_el66;
  std::vector<float> new_el67;
  std::vector<float> new_el77;

  //differences
  std::vector<float> diff_el33;
  std::vector<float> diff_el34;
  std::vector<float> diff_el35;
  std::vector<float> diff_el44;
  std::vector<float> diff_el45;
  std::vector<float> diff_el46;
  std::vector<float> diff_el55;
  std::vector<float> diff_el56;
  std::vector<float> diff_el57;
  std::vector<float> diff_el66;
  std::vector<float> diff_el67;
  std::vector<float> diff_el77;

  //old vectors
  std::vector<float> myoldel33;
  std::vector<float> myoldel34;
  std::vector<float> myoldel35;
  std::vector<float> myoldel44;
  std::vector<float> myoldel45;
  std::vector<float> myoldel46;
  std::vector<float> myoldel55;
  std::vector<float> myoldel56;
  std::vector<float> myoldel57;
  std::vector<float> myoldel66;
  std::vector<float> myoldel67;
  std::vector<float> myoldel77;

  counter=0;
  counter1=0;
  //old_nrlines=0;
  //new_nrlines=0;
  
  std::ifstream olddata3; 
  olddata3.open("old_dbmatrix.dat",std::ios::in); 
  if(!olddata3) {
    std::cerr <<"Error: old_dbmatrix.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!olddata3.eof() ) { 
    olddata3 >> old_index >> old_elem33 >> old_elem34 >> old_elem44 >> old_elem35 >> old_elem45 >> old_elem55 >> old_elem46 >> old_elem56 >>old_elem66 >> old_elem57 >> old_elem67 >> old_elem77 ; 
    old_index_id.push_back(old_index);
    old_el33.push_back(old_elem33);
    old_el34.push_back(old_elem34);
    old_el35.push_back(old_elem35);
    old_el44.push_back(old_elem44);
    old_el45.push_back(old_elem45);
    old_el46.push_back(old_elem46);
    old_el55.push_back(old_elem55);
    old_el56.push_back(old_elem56);
    old_el57.push_back(old_elem57);
    old_el66.push_back(old_elem66);
    old_el67.push_back(old_elem67);
    old_el77.push_back(old_elem77);
    //old_nrlines++;
  }
  olddata3.close();

  std::ifstream newdata3;
  std::ofstream myMatrixFile("diffMatrixTest.dat",std::ios::out);
  newdata3.open("goodMatrix2009_08_31_run112487.dat",std::ios::in); 
  if(!newdata3) {
    std::cerr <<"Error: goodMatrix2009_08_31_run112487.dat -> no such file!"<< std::endl;
    exit(1);
  }
  
  while (!newdata3.eof() ) { 
    newdata3 >> new_index >> new_elem33 >> new_elem34 >> new_elem44 >> new_elem35 >> new_elem45 >> new_elem55 >> new_elem46 >> new_elem56 >>new_elem66 >> new_elem57 >> new_elem67 >> new_elem77; 
    new_index_id.push_back(new_index);
    new_el33.push_back(new_elem33);
    new_el34.push_back(new_elem34);
    new_el35.push_back(new_elem35);
    new_el44.push_back(new_elem44);
    new_el45.push_back(new_elem45);
    new_el46.push_back(new_elem46);
    new_el55.push_back(new_elem55);
    new_el56.push_back(new_elem56);
    new_el57.push_back(new_elem57);
    new_el66.push_back(new_elem66);
    new_el67.push_back(new_elem67);
    new_el77.push_back(new_elem77);
    //new_nrlines++;
  }
  newdata3.close();

  //resize
  diff_el33.resize(MAX_SIZE);
  diff_el34.resize(MAX_SIZE);
  diff_el35.resize(MAX_SIZE);
  diff_el44.resize(MAX_SIZE);
  diff_el45.resize(MAX_SIZE);
  diff_el46.resize(MAX_SIZE);
  diff_el55.resize(MAX_SIZE);
  diff_el56.resize(MAX_SIZE);
  diff_el57.resize(MAX_SIZE);
  diff_el66.resize(MAX_SIZE);
  diff_el67.resize(MAX_SIZE);
  diff_el77.resize(MAX_SIZE);

  myoldel33.resize(MAX_SIZE);
  myoldel34.resize(MAX_SIZE);
  myoldel35.resize(MAX_SIZE);
  myoldel44.resize(MAX_SIZE);
  myoldel45.resize(MAX_SIZE);
  myoldel46.resize(MAX_SIZE);
  myoldel55.resize(MAX_SIZE);
  myoldel56.resize(MAX_SIZE);
  myoldel57.resize(MAX_SIZE);
  myoldel66.resize(MAX_SIZE);
  myoldel67.resize(MAX_SIZE);
  myoldel77.resize(MAX_SIZE);

  for(int i=0; i<MAX_SIZE;++i){
    counter=old_index_id[i];  
    myoldel33[i]=old_el33[i];
    myoldel34[i]=old_el34[i];
    myoldel35[i]=old_el35[i];
    myoldel44[i]=old_el44[i];
    myoldel45[i]=old_el45[i];
    myoldel46[i]=old_el46[i];
    myoldel55[i]=old_el55[i];
    myoldel56[i]=old_el56[i];
    myoldel57[i]=old_el57[i];
    myoldel66[i]=old_el66[i];
    myoldel67[i]=old_el67[i];
    myoldel77[i]=old_el77[i];
    
    for (unsigned int k=0;k<new_index_id.size()-1;k++){
      counter1=new_index_id[k];
      if(counter == counter1){
	diff_el33[k]=old_el33[i] - new_el33[k];
	diff_el34[k]=old_el34[i] - new_el34[k];
	diff_el35[k]=old_el35[i] - new_el35[k];
	diff_el44[k]=old_el44[i] - new_el44[k];
	diff_el45[k]=old_el45[i] - new_el45[k];
	diff_el46[k]=old_el46[i] - new_el46[k];
	diff_el55[k]=old_el55[i] - new_el55[k];
	diff_el56[k]=old_el56[i] - new_el56[k];
	diff_el57[k]=old_el57[i] - new_el57[k];
	diff_el66[k]=old_el66[i] - new_el66[k];
	diff_el67[k]=old_el67[i] - new_el67[k];
	diff_el77[k]=old_el77[i] - new_el77[k];
	//std::cout<<old_el33[i]<<" new_el33[k]"<<new_el33[k]<<std::endl;
	myMatrixFile<<counter<<"  "<<diff_el33[k]<<"  "<< diff_el34[k]<<"  "<<diff_el35[k]<<"  "<<diff_el44[k]<<"  "<<diff_el45[k]<<"  "<<diff_el46[k]<<"  "<<diff_el55[k]<<"  "<<diff_el56[k]<<"  "<<diff_el57[k]<<"  "<<diff_el66[k]<<"  "<<diff_el67[k]<<"  "<<diff_el77[k]<<"  "<<std::endl;	
      }
    }
  }

}


Compare::~Compare()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
Compare::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;



#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}


// ------------ method called once each job just before starting event loop  ------------
void 
Compare::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
Compare::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(Compare);
