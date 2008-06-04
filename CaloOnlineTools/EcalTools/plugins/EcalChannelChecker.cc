// -*- C++ -*-
//
// Package:   EcalChannelChecker 
// Class:     EcalChannelChecker 
// 
/**\class EcalChannelChecker EcalChannelChecker.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
 */
// 
// Original Author:  Caterina DOGLIONI
//         Created:  Tu Apr 22 5:46:22 CEST 2008
// $Id: EcalChannelChecker.cc,v 1.7 2008/05/05 13:31:42 doglioni Exp $
//
//

// system include files

#include <iomanip>

#include "CaloOnlineTools/EcalTools/plugins/EcalChannelChecker.h"
#include "TCut.h"
#include "TTree.h"
#include "TFile.h"
#include "TH1F.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h" 
#define MAX_XTALS 61200

using namespace edm;

//
// constructors and destructor
//
EcalChannelChecker::EcalChannelChecker(const edm::ParameterSet& iConfig)
	:inputTreeFileName_ (iConfig.getUntrackedParameter<std::string>("inputTreeFileName","")),
	inputHistoFileName_(iConfig.getUntrackedParameter<std::string>("inputHistoFileName","")),
	outputFileName_(iConfig.getUntrackedParameter<std::string>("outputFileName","")),
	v_cuts_ (iConfig.getUntrackedParameter< std::vector<std::string> >  ("v_cuts") ),
	v_maskedHi_(iConfig.getUntrackedParameter < std::vector<int> > ("v_maskedHi") ),
	v_maskedSlices_(iConfig.getUntrackedParameter < std::vector<std::string> > ("v_maskedSlices") )

{
	//--- INITIALIZATIONS 
	nCuts_=v_cuts_.size();
	initHistTypeMaps();

	//--- CLEANING OF VECTORS
	for (int i=0; i < MAX_XTALS; i++) {	
		xtalBitmask_[i] = 0;
	}

	//--- OPENING FILES

	//TODO: try/catch 
	fin_tree_= TFile::Open(inputTreeFileName_.c_str(),"READ");
	if (fin_tree_->IsZombie()) {
		std::cout << "Error opening tree file" << std::endl;
		exit(-1);
	}

	//TODO: try/catch 
	fin_histo_= TFile::Open(inputHistoFileName_.c_str(),"READ");
	if (fin_histo_->IsZombie()) {
		std::cout << "Error opening histo file" << std::endl;
		exit(-1);
	}

	//TODO: try/catch
	fout_=TFile::Open(outputFileName_.c_str(),"RECREATE");
	if (fout_->IsZombie()) {
		std::cout << "Error opening output file" << std::endl;
		exit(-1);
	}

	//--- INIT TREE

	//TODO:name of tree from .cfg + try/catch

	t_=(TTree*)fin_tree_->Get("xtal_tree");

	//TODO: try/catch
	if (t_==0) std::cout << "no tree" << std::endl; 

	t_->SetBranchAddress("ic", &ic, &b_ic);
	t_->SetBranchAddress("slice", slice, &b_slice);
	t_->SetBranchAddress("ieta", &ieta, &b_ieta);
	t_->SetBranchAddress("iphi", &iphi, &b_iphi);
	t_->SetBranchAddress("hashedIndex", &hashedIndex, &b_hashedIndex);
	t_->SetBranchAddress("ped_avg", &ped_avg, &b_ped_avg);
	t_->SetBranchAddress("ped_rms", &ped_rms, &b_ped_rms);
	t_->SetBranchAddress("ampli_avg", &ampli_avg, &b_ampli_avg);
	t_->SetBranchAddress("ampli_rms", &ampli_rms, &b_ampli_rms);
	t_->SetBranchAddress("jitter_avg", &jitter_avg, &b_jitter_avg);
	t_->SetBranchAddress("jitter_rms", &jitter_rms, &b_jitter_rms);
	t_->SetBranchAddress("ampli_fracBelowThreshold", &ampli_fracBelowThreshold, &b_ampli_fracBelowThreshold);
	t_->SetBranchAddress("entries", &entries, &b_entries);
	t_->SetBranchAddress("entriesOverAvg",&entriesOverAvg, &b_entriesOverAvg);


}

EcalChannelChecker::~EcalChannelChecker()
{

}

//
// member functions
//

// ------------ method called for each event  ------------

	void
EcalChannelChecker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	//do nothing!
}//end analyze


// ------------ method called once each job just before starting event loop  ------------
	void 
EcalChannelChecker::beginJob(const edm::EventSetup&)
{

	//preparing TCuts for masking purposes
	//note: I cowardly refuse to template this function, but if it's a good idea I will.
	std::string maskedHi = makeCutFromMaskedVectorInt(v_maskedHi_, "hashedIndex");
	std::string maskedSlices = makeCutFromMaskedVectorString(v_maskedSlices_, "slice");

	std::vector<std::string> v_masked;
	if (maskedHi!="") v_masked.push_back(maskedHi);
	if (maskedSlices!="") v_masked.push_back(maskedSlices);

	//filling event list vector with lists produced by cuts
	//by construction, list made from cut in v_cuts_[i] is in v_eventList_[i] 

	fillEventListVector(v_cuts_,v_masked);

	//loop on all event lists 
	for (unsigned int i=0; i<v_eventList_.size(); i++){
		//loop on events in event lists
		for (unsigned int j=0; j<(unsigned int)v_eventList_[i].GetN(); j++) {

			t_->GetEntry(v_eventList_[i].GetEntry(j));

			//if bitmask hasn't been allocated for selected crystal, allocate it
			if (!xtalBitmask_[hashedIndex]) xtalBitmask_[hashedIndex] = new std::vector<bool> (v_cuts_.size());
			//flagging the cut relative to the entryList
			xtalBitmask_[hashedIndex]->operator[](i)=1;

		}
		//adding current event list to total event list (removing duplicate crystals)
		totalEventList_.Add(&v_eventList_[i]); 

	}

	//loop on events in event lists
	for (unsigned int j=0; j<(unsigned int)totalEventList_.GetN(); j++) {

		t_->GetEntry(totalEventList_.GetEntry(j));
		//dirty trick with enum + implicit converter 
		for (unsigned int i=0; i<EcalChannelChecker::NTYPES; i++) {
			writeHistFromFile(hashedIndex, slice, ic, (EcalChannelChecker::n_h1Type)i);
		} 

	}

}



// ------------ method called once each job just after ending the event loop  ------------
	void 
EcalChannelChecker::endJob() 
{

	//printouts
	edm::LogVerbatim("") << "------ORDERED LIST OF CUTS-----"; 

	for (unsigned int i=0; i<v_eventList_.size(); ++i){
		edm::LogVerbatim("")  << i << ". " << v_cuts_[i]; 
	}

	edm::LogVerbatim("") << "------MASKED-----";

	printMaskedHi();
	printMaskedSlices();

	edm::LogVerbatim("") << "------EVENT LISTS-----";

	/*for (unsigned int i=0; i<v_eventList_.size(); i++){
	  std::cout << v_eventList_[i].GetTitle() << std::endl; 
	  t_->SetEventList(&v_eventList_[i]);
	  t_->Scan();
	  }
	  t_->SetEventList(&totalEventList_);
	  t_->Scan();*/


	//----DIRTY HACK UNTIL THINGS ARE FIXED WITH ROOT OSTREAMS and above loops can be used

	//loop on all event lists 
	for (unsigned int i=0; i<v_eventList_.size(); i++){
		//loop on events in event lists

		edm::LogVerbatim("") << "Event List for cut: " <<  v_cuts_[i];
		printLogEventList(v_eventList_[i]);

	}

	//event list for all cuts
	edm::LogVerbatim("") << "Event List for all cuts";
	printLogEventList(totalEventList_);


	//---END DIRTY HACK

	//cleanup
	for (unsigned int i=0; i<MAX_XTALS; i++) {
		if (xtalBitmask_[i]!=0)  delete xtalBitmask_[i];
	}

	fin_tree_->Close();
	fin_histo_->Close();
	fout_->Close();
}

//-------------- "helper" methods ----------------

void EcalChannelChecker::printMaskedHi() {

	edm::LogVerbatim("") << "Masked crystals (hashedIndex)"; 
	std::string bufferString;

	if (v_maskedHi_.size()==0) edm::LogVerbatim("") << "no masked crystals";

	else {
		for (unsigned int i=0; i<v_maskedHi_.size(); i++) {
			bufferString = bufferString + intToString(v_maskedHi_[i]) + "\t";
			if(i!=0 && i%10==0) bufferString += "\n";
		}
		edm::LogVerbatim("") << bufferString;
	}
}


void EcalChannelChecker::printMaskedSlices() {

	edm::LogVerbatim("") << "Masked slices (sm)";     
	std::string bufferString;

	if (v_maskedSlices_.size()==0) edm::LogVerbatim("") << "no masked slices";

	else {
		for (unsigned int i=0; i<v_maskedSlices_.size(); i++) {
			bufferString = bufferString + v_maskedSlices_[i] + "\t";
			if(i%10==0) bufferString += "\n";
		}
		edm::LogVerbatim("") << bufferString;
	}
}


void EcalChannelChecker::printLogEventList(const TEventList & eventList) {

	edm::LogVerbatim("") << "slice" << "\t"
		<< "ic     " << "\t"    
		<< "hi     " << "\t"    
		<< "ieta   " << "\t"
		<< "iphi   " << "\t"
		<< "amp_avg" << "\t"
		<< "amp_rms" <<  "\t"
		<< "ped_avg" << "\t"
		<< "ped_rms" << "\t"
		<< "jit_avg" <<  "\t"
		<< "jit_rms" <<  "\t"
		<< "fracAmp" << "\t"
		<< "entries" << "\t"
		<< "fracEnt" << "\t"
		<< "failed " << "\t" ;

	for (unsigned int j=0; j<(unsigned int)eventList.GetN(); j++) {

		t_->GetEntry(eventList.GetEntry(j));

		edm::LogVerbatim("") << slice << "\t"
			<< ic << "\t"
			<< hashedIndex << "\t"
			<< ieta << "\t"
			<< iphi << "\t"
			<< std::setprecision(3) << ampli_avg << "\t"
			<< std::setprecision(3) << ampli_rms << "\t"
			<< std::setprecision(3) << ped_avg << "\t"
			<< std::setprecision(3) << ped_rms << "\t"
			<< std::setprecision(3) << jitter_avg << "\t"
			<< std::setprecision(3) << jitter_rms <<  "\t"
			<< std::setprecision(3) << ampli_fracBelowThreshold << "\t"
			<< std::setprecision(6) << entries << "\t"
			<< std::setprecision(3) << entriesOverAvg << "\t"
			<< printBitmaskCuts(xtalBitmask_[hashedIndex]);

	}

}


TEventList *
EcalChannelChecker::getEventListFromCut(const TCut& cut) {

	t_->Draw(">>List", cut);
	TEventList *list = (TEventList*)gDirectory->Get("List");

	return list;

}

void EcalChannelChecker::initHistTypeMaps() {

	//FIXME: change this when name of 1st analyzer is changed
	h1TypeToDirectoryMap_[EcalChannelChecker::H1_AMPLI]="ecalMipHists/XtalAmpli";
	h1TypeToDirectoryMap_[EcalChannelChecker::H1_PED]="ecalMipHists/XtalPed";
	h1TypeToDirectoryMap_[EcalChannelChecker::H1_JITTER]="ecalMipHists/XtalJitter";
	h1TypeToDirectoryMap_[EcalChannelChecker::PROF_PULSE]="ecalMipHists/XtalPulse";

	h1TypeToNameMap_[EcalChannelChecker::H1_AMPLI]="ampli";
	h1TypeToNameMap_[EcalChannelChecker::H1_PED]="ped";
	h1TypeToNameMap_[EcalChannelChecker::H1_JITTER]="jitter";
	h1TypeToNameMap_[EcalChannelChecker::PROF_PULSE]="pulse";
}

void EcalChannelChecker::fillEventListVector(const std::vector<std::string> & v_cuts, const std::vector<std::string> & v_masked) {

	std::string masked = "";

	//retrieving everything that has to be masked in a single cut
	for (unsigned int j=0; j<v_masked.size(); j++) {

		if (j==0) masked = masked + v_masked[j];
		else masked = masked + " && " + v_masked[j];            

	}

	TCut bufferMasked = masked.c_str();

	for (unsigned int i=0; i<v_cuts.size(); i++) {

		TCut bufferCut = v_cuts[i].c_str();
		TCut combinedCut = bufferCut && bufferMasked;
		TEventList eventList = *getEventListFromCut(combinedCut); 
		v_eventList_.push_back(eventList);

	}

}

//v_masked = vector of masked crystals/ism, type = name of quantity in cut e.g. hashedIndex, ieta, iphi...
std::string EcalChannelChecker::makeCutFromMaskedVectorInt(const std::vector<int> & v_masked, const std::string & type ) {

	//in case of no masked crystals/sms, return empty string 
	std::string cutString = "";
	std::string bufferString = "";

	for (unsigned int i=0; i<v_masked.size(); i++) {
		if (i==0) bufferString = type + "!=" + intToString(v_masked[i]);
		else bufferString = " && " + type + "!=" + intToString(v_masked[i]);
		cutString = cutString + bufferString ;  
	}

	//debug
	//std::cout << cutString << std::endl;

	return cutString;
}

//v_masked = vector of masked slices (sm), type = name of quantity in cut e.g. slice
std::string EcalChannelChecker::makeCutFromMaskedVectorString(const std::vector<std::string> & v_masked, const std::string & type ) {

	//in case of no masked crystals/sms, return empty string 
	std::string cutString = "";
	std::string bufferString = "";

	for (unsigned int i=0; i<v_masked.size(); i++) {
		if (i==0) bufferString = type + "!=\"" + v_masked[i] + "\"";
		else bufferString = " && " + type + "!=\"" + v_masked[i] + "\"";
		cutString = cutString + bufferString ;
	}

	//debug
	//std::cout << cutString << std::endl;

	return cutString;
}



void EcalChannelChecker::writeHistFromFile (const int hashedIndex, const char* slice, const int ic, const EcalChannelChecker::n_h1Type H1_TYPE) {

	//getting histogram from input file
	std::string dirbuffer = h1TypeToDirectoryMap_[H1_TYPE]+"/"+intToString(hashedIndex);
	fin_histo_->cd();
	TH1F * hist  = (TH1F*)fin_histo_->Get(dirbuffer.c_str());

	//name: ism_ic_typeOfHistogram_bitmask
	//title: ism_ic_typeOfHistogram:cut1:cut2
	std::string histName = std::string(slice) + "_" + intToString(ic) + "_" + h1TypeToNameMap_[H1_TYPE] + "_" + printBitmask(xtalBitmask_[hashedIndex]); 
	std::string histTitle = std::string(slice) + "_" + intToString(ic) + "_" + h1TypeToNameMap_[H1_TYPE] + printBitmaskCuts(xtalBitmask_[hashedIndex]); 
	hist->SetName(histName.c_str());
	hist->SetTitle(histTitle.c_str());

	//writing histogram on output file
	fout_->cd();
	hist->Write();

	//debug
	//        std::cout << "writing: " << histTitle << std::endl;
}

std::string EcalChannelChecker::intToString(int num)
{
	using namespace std;
	ostringstream myStream;
	myStream << num << flush;
	return(myStream.str()); //returns the string form of the stringstream object
}

std::string EcalChannelChecker::printBitmask(std::vector<bool>* bitmask) {

	std::string bitmaskString;

	for (unsigned int i=0; i< bitmask->size(); i++) {
		//using intToString to add all bits in bitmask to the string
		bitmaskString += intToString(int(bitmask->operator[](i)));
	}

	return bitmaskString;
}

std::string EcalChannelChecker::printBitmaskCuts(std::vector<bool>*bitmask) {

	std::string bitmaskCuts;

	for (unsigned int i=0; i< bitmask->size(); i++) {
		if (bitmask->operator[](i)) {
			//getting name of cut from TCuts vector
			bitmaskCuts += ":";
			bitmaskCuts += v_cuts_[i];
		}
	}

	return bitmaskCuts;
}
