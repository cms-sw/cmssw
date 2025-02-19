// -*- C++ -*-
//
// Package:    Tracker_OldtoNewConverter.cc
// Class:      Tracker_OldtoNewConverter
// 
/**\class MuonGeometryIntoNtuples MuonGeometryIntoNtuples.cc Alignment/MuonGeometryIntoNtuples/src/MuonGeometryIntoNtuples.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Nhan Tran
//         Created:  Mon Jul 16m 16:56:34 CDT 2007
// $Id: Tracker_OldtoNewConverter.cc,v 1.3 2011/09/15 08:12:03 mussgill Exp $
//
//

// system include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include <algorithm>
#include "TTree.h"
#include "TFile.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <fstream>
#include <iostream>

//
// class decleration
//

class Tracker_OldtoNewConverter : public edm::EDAnalyzer {
public:
	explicit Tracker_OldtoNewConverter(const edm::ParameterSet&);
	~Tracker_OldtoNewConverter();
	
	
private:
	virtual void beginJob();
	virtual void analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup);
	virtual void endJob() ;
	
	void createMap();
	void addBranches();
	
	// ----------member data ---------------------------

	std::string m_conversionType;
	std::string m_textFile;
	std::string m_inputFile;
	std::string m_outputFile;
	std::string m_treeName;
	
	std::map< uint32_t, uint32_t > theMap;

	TFile* m_inputTFile;
	TFile* m_outputTFile;
	TTree* m_inputTree;
	TTree* m_outputTree;
	
	
	uint32_t rawid_i, rawid_f;
	//int rawid_i, rawid_f;
	double x_i, y_i, z_i, a_i, b_i, c_i;
	double x_f, y_f, z_f, a_f, b_f, c_f;
	
	
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
Tracker_OldtoNewConverter::Tracker_OldtoNewConverter(const edm::ParameterSet& iConfig) :
  m_inputTFile(0), m_outputTFile(0), m_inputTree(0), m_outputTree(0),
  rawid_i(0), rawid_f(0),
  x_i(0.), y_i(0.), z_i(0.), a_i(0.), b_i(0.), c_i(0.),
  x_f(0.), y_f(0.), z_f(0.), a_f(0.), b_f(0.), c_f(0.)
{
	m_conversionType = iConfig.getUntrackedParameter< std::string > ("conversionType");
	m_inputFile = iConfig.getUntrackedParameter< std::string > ("inputFile");
	m_outputFile = iConfig.getUntrackedParameter< std::string > ("outputFile");
	m_textFile = iConfig.getUntrackedParameter< std::string > ("textFile");
	m_treeName = iConfig.getUntrackedParameter< std::string > ("treeName");
}


Tracker_OldtoNewConverter::~Tracker_OldtoNewConverter()
{}


//
// member functions
//

// ------------ method called to for each event  ------------
void Tracker_OldtoNewConverter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{}


// ------------ method called once each job just before starting event loop  ------------
void Tracker_OldtoNewConverter::beginJob()
{
	
	m_inputTFile = new TFile(m_inputFile.c_str());
	m_outputTFile = new TFile(m_outputFile.c_str(),"RECREATE");

	m_inputTree = (TTree*) m_inputTFile->Get(m_treeName.c_str());
	m_outputTree = new TTree(m_treeName.c_str(), m_treeName.c_str());
	
	createMap();
	addBranches();

	uint32_t nEntries = m_inputTree->GetEntries();
	uint32_t iter = 0;
	for (uint32_t i = 0; i < nEntries; ++i){
		m_inputTree->GetEntry(i);
		std::map< uint32_t, uint32_t >::const_iterator it = theMap.find(rawid_i);
		
		if (it == theMap.end()){
			edm::LogInfo("ERROR") << "Error: couldn't find rawId: " << rawid_i;
			iter++;
		}
		else{
			rawid_f = (it)->second;
			x_f = x_i;
			y_f = y_i;
			z_f = z_i;
			a_f = a_i;
			b_f = b_i;
			c_f = c_i;
			m_outputTree->Fill();
		}

	}
	edm::LogInfo("errors") << "Couldn't find: " << iter;
	m_outputTFile->cd();
	m_outputTree->Write();
	m_outputTFile->Close();
}



void Tracker_OldtoNewConverter::createMap() {

	std::ifstream myfile( m_textFile.c_str() );
	if (!myfile.is_open())
		throw cms::Exception("FileAccess") << "Unable to open input text file";

	uint32_t oldid;
	uint32_t newid;

	uint32_t ctr = 0;
	while( !myfile.eof() && myfile.good() ){

		myfile >> oldid >> newid;

		//depends on conversion type: OldtoNew or NewtoOld
		std::pair< uint32_t, uint32_t > pairType;
		if (m_conversionType == "OldtoNew") {pairType.first = oldid; pairType.second = newid;}
		if (m_conversionType == "NewtoOld") {pairType.first = newid; pairType.second = oldid;}


		theMap.insert( pairType );

		if (myfile.fail()) break;

		ctr++;
	}
	edm::LogInfo("Check") << ctr << " alignables read.";
}

void Tracker_OldtoNewConverter::addBranches(){

	m_inputTree->SetBranchAddress("rawid", &rawid_i);
	m_inputTree->SetBranchAddress("x", &x_i);
	m_inputTree->SetBranchAddress("y", &y_i);
	m_inputTree->SetBranchAddress("z", &z_i);
	m_inputTree->SetBranchAddress("alpha", &a_i);
	m_inputTree->SetBranchAddress("beta", &b_i);
	m_inputTree->SetBranchAddress("gamma", &c_i);

	m_outputTree->Branch("rawid", &rawid_f, "rawid/I");
	m_outputTree->Branch("x", &x_f, "x/D");
	m_outputTree->Branch("y", &y_f, "y/D");
	m_outputTree->Branch("z", &z_f, "z/D");
	m_outputTree->Branch("alpha", &a_f, "alpha/D");
	m_outputTree->Branch("beta", &b_f, "beta/D");
	m_outputTree->Branch("gamma", &c_f, "gamma/D");

}


// ------------ method called once each job just after ending the event loop  ------------
void Tracker_OldtoNewConverter::endJob() {
}



//define this as a plug-in
DEFINE_FWK_MODULE(Tracker_OldtoNewConverter);
