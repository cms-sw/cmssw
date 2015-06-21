#include "Riostream.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <math.h>
#include "TH2.h" 
#include "TH1.h" 
#include "math.h" 
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include "TVectorF.h"
#include "TF1.h"
#include "TLegend.h"
#include "TSystem.h"
#include "TInterpreter.h"
#include "toolbox/L1GtNtuple.h"
#include "L1PrescalesSimulator.C"



using namespace std;


class L1RatesAnalyzer: public L1PrescalesSimulator {
public:

	//constructor    
	L1RatesAnalyzer(std::string filename) 
	{
		Open(filename);
                rndng_ = new TRandom2();
                rndng_->SetSeed();
	}
	L1RatesAnalyzer() {
	}

	void run(int nevs=-1);

	~L1RatesAnalyzer() {
	}

private:

	void eventLoopForBits(const std::vector<unsigned int>& bits);
	bool getBit(const ULong64_t val, const unsigned int bit);
	ULong64_t bitmaskForBit(const unsigned int bit);
	void build192BitVec(const ULong64_t w1, const ULong64_t w2, const ULong64_t w3, std::vector<bool>& v);
	void getTopTriggers(std::vector<unsigned int>& topVec);

	void getLuminosities();

	// routines for correlation plot
	void addEvent2CorrelationPlot(const ULong64_t a1, const ULong64_t a2, const ULong64_t tt, const std::vector<unsigned int>& enabledBits);

	void drawCorrelationPlot(const Long64_t i2);

	void drawPlotsForBits();
	void drawPlotsForBitsCore(map<int, vector<map<int, float> > > dataMap, TCanvas* c1);

	toolbox tb_;

	// definitions
	double lumiSectTimeNs_;

	// luminosity range relevant for fit
	Axis_t xFitMin_;
	Axis_t xFitMax_;

	// luminosity range on x axis

	float xMin_;
	float xMax_;

	// constants
	double commonPrescale_;

	int numberOfBunches_;

	int nOfPlots_;

	// first int: run
	// second int: lumi section
	// the float: luminosity
	map<int, map<int, float> > lumiSec2Lumi_;

	// first int: run
	// index in vector: trigger bit
	// second int: lumi section
	// the float: rate
	map<int, vector<map<int, float> > > lumiSec2rate_;

	map<int, vector<map<int, float> > > lumiSec2ratePrescaled_;

	// stores parameters
	std::map<string,string> parameterMap_;

	std::vector<TVectorF> rateVecs_;

	TVectorF lumiVec_;

	// multiplicators for number of bxs (for each a seperate graph will be drawn

	vector<float> bxMultiplicatorVec_;

	int yMax_;

	// contains masks for the trigger bits
	// ind0: bits 0-63
	// ind1: bits 63-127
	// ind1: bits 128-192
	std::vector<ULong64_t> bitMaskVec_;

	TLegend* l_;

	TH2D *h2_;

	int nOfHists_;

	// buffers
	std::vector<unsigned int> topVec_;
};



///////////////////////////////////////////////////////////////////////////////////////

void L1RatesAnalyzer::run(int nevs) {

	cout << directory_<<endl;

	h2_ = new TH2D("h2","Correlations between two trigger bits",15,0,15,15,0,15);

	nOfHists_=15;

	lumiSectTimeNs_=23.31;

	// clear buffers and parse config file
	lumiSec2Lumi_.clear();
	lumiSec2rate_.clear();
	lumiSec2ratePrescaled_.clear();
	bitMaskVec_.clear();
	//rateVecs_.Resize(0);

	tb_.readConfigFile(tb_.getMacroDir()+"conf/RatesAnalyzer.conf",parameterMap_);
	
	// fill parameters
	xMin_ = toolbox::convertFromString<float>(parameterMap_["xMin"]);
	xMax_ = toolbox::convertFromString<float>(parameterMap_["xMax"]);
	xFitMin_ = toolbox::convertFromString<float>(parameterMap_["xFitMin"]);
	xFitMax_ = toolbox::convertFromString<float>(parameterMap_["xFitMax"]);

	commonPrescale_ = toolbox::convertFromString<float>(parameterMap_["commonPrescale"]);

	numberOfBunches_ = toolbox::convertFromString<int>(parameterMap_["nOfBunches"]);

	cout << "Number of bunches: " << numberOfBunches_ << endl;

	// get bitmasks from file
	std::map<string,string>::iterator itr;
	
	int tmp =1;

	tb_.readConfigFile(tb_.getMacroDir()+"conf/bitmasks.dat",parameterMap_);

	cout << "Bit masks: " << endl;	

	do 
	{
			ostringstream oss;

			oss << "m" << tmp;	
	
			itr=parameterMap_.find(oss.str());

		if (itr!=parameterMap_.end())
		{
			cout << (itr->second) << endl;			
			bitMaskVec_.push_back(toolbox::convertFromString<ULong64_t>(itr->second, 16));
		}
	
		tmp++;

	} while(itr!=parameterMap_.end());


	if (nevs) {
		hreset();

		getLuminosities();

		getTopTriggers(topVec_); 
		
		getPrescales();

		cout << "Top bits: " << endl;

		for (unsigned int k=0; k< topVec_.size(); k++)
		{
			cout <<  topVec_.at(k) << endl;
		} 


		eventLoopForBits(topVec_);

	}

}


void  L1RatesAnalyzer::drawCorrelationPlot(const Long64_t i2)
{
	const double scaler = pow(i2,-1)*100;
	h2_->Scale(scaler);
	h2_->Draw();

	TCanvas* c1 = new TCanvas("c1","",900,700);
	// to make warning disappear	
	c1->GetEvent();

        gStyle->SetPalette(1);

        ostringstream buf;

        buf << "Bits: ";

	std::vector<unsigned int> topVecSorted= topVec_;

  	sort(topVecSorted.begin(), topVecSorted.end());
	
        for (unsigned int i=0; i<topVecSorted.size(); i++)
   		buf << topVecSorted.at(i) << ", ";


        h2_->SetTitle(buf.str().c_str());

  	h2_->GetXaxis()->SetTitle("Bit number");
  	h2_->GetYaxis()->SetTitle("Bit number");

 	gStyle->SetPaintTextFormat("5.2f"); 
  	gStyle->SetOptStat(0); 
  
        h2_->Draw("COLZTEXT[cutg]");

}

void L1RatesAnalyzer::eventLoopForBits(const std::vector<unsigned int>& bits)
{
	std::cout << "Start looping..." << std::endl;

	std::vector<unsigned int> bitsSorted = bits;

	std::sort(bitsSorted.begin(), bitsSorted.end());

	//number of events to process

	Long64_t nevents(-1);

	if (nevents == -1 || nevents > GetEntries())
		nevents = GetEntries();
	std::cout << nevents << " to process ..." << std::endl;

	//loop over the events
	for (Long64_t i = 0; i < nevents; i++) {
		//load the i-th event 

		Long64_t ientry = LoadTree(i);
		if (ientry < 0)
			break;
		GetEntry(i);

		const ULong64_t a1 = gt_->tw1.at(2) & bitMaskVec_[0];
		const ULong64_t a2 = gt_->tw2.at(2) & bitMaskVec_[1];
		const ULong64_t tt = gt_->tt.at(2) & bitMaskVec_[2];
		const int run = event_->run;
		const int ls = event_->lumi;

		if (i != 0 && (i % 10000) == 0) {
			std::cout << "- processing event " << i << ", Run: " << run
					<< " ,tw2: " << a2 << " ,LS: " << ls << std::endl;
		}


		// prescale the bits

		ULong64_t a1Presc = a1;
		ULong64_t a2Presc = a2;
		ULong64_t ttPresc = tt;


		for (std::map<unsigned int, float>::const_iterator it = bit2prescale_.begin(); it
				!= bit2prescale_.end(); it++) {
			unsigned int bit = it->first;
			float presc = it->second;

			if (presc != -1) {
				if (bit < 64) {
					prescale(a1Presc, bit, presc);
				} else if (bit < 128) {
					prescale(a2Presc, bit-64, presc);
				} else {
					prescale(ttPresc, bit-128, presc);
				}

			}

		}


		// process event for correlation plot
		addEvent2CorrelationPlot(a1, a2, tt, bitsSorted);
		
	
		// loop over the 15 most frequent bits  to count rates
		for (unsigned int k = 0; k < bits.size(); k++) {

			if (bits.at(k) < 64) {
				if (getBit(a1, bits.at(k))) {
					lumiSec2rate_[run].at(k)[ls]++;
				}

				if (getBit(a1Presc, bits.at(k))) {
					lumiSec2ratePrescaled_[run].at(k)[ls]++;
				}

			} else if (bits.at(k) < 128) {

				if (getBit(a2, bits.at(k)-64)) {
					lumiSec2rate_[run].at(k)[ls]++;
				}

				if (getBit(a2Presc, bits.at(k))) {
					lumiSec2ratePrescaled_[run].at(k)[ls]++;
				}

			} else {

				if (getBit(tt, bits.at(k)-128)) {
					lumiSec2rate_[run].at(k)[ls]++;
				}

				if (getBit(ttPresc, bits.at(k))) {
					lumiSec2ratePrescaled_[run].at(k)[ls]++;
				}
			}

		}

	}

	drawCorrelationPlot(nevents);

	drawPlotsForBits();

	return;
}


void L1RatesAnalyzer::addEvent2CorrelationPlot(const ULong64_t a1, const ULong64_t a2, const ULong64_t tt, const std::vector<unsigned int>& enabledBits)
{	
	vector<bool> v;

	build192BitVec(a1, a2, tt, v);
  	// loop over the vector

	for (unsigned int r=0; r<enabledBits.size(); r++)
  	{			
		std::ostringstream oss;
		oss << enabledBits.at(r);

		h2_->GetXaxis()->SetBinLabel(r+1, oss.str().c_str());
		h2_->GetYaxis()->SetBinLabel(r+1, oss.str().c_str());

		for (unsigned int k=0; k<enabledBits.size(); k++)
		{
			if (v.at(enabledBits.at(k)) && 	v.at(enabledBits.at(r)) )
				h2_->Fill(r,k,1);
		}
	}
}


void  L1RatesAnalyzer::getLuminosities() {

	
	TString lumiDir;
	lumiDir.Append(directory_+"/lumis/lumis.root");

	cout << directory_ << "/lumis/lumis.root";

	TFile f(lumiDir.Data()); // open the file

	cout << "before loop" << endl;


	TTree* const tree = dynamic_cast<TTree*>(f.Get("ntuple"));
	if(!tree)
		throw std::runtime_error("Expected \"ntuple\" in luminosities file to be of type TTree, but it is not.");

	float run;
	float ls;
	float lumiReported,lumiDelivered;

	// Linking the local variables to the tree branches
	tree->SetBranchAddress("run", &run);
	tree->SetBranchAddress("ls", &ls);
	tree->SetBranchAddress("lumiDelivered", &lumiDelivered);
	tree->SetBranchAddress("lumiReported", &lumiReported);

	const Long64_t nEntries = tree->GetEntries();

	map<float,bool> tmp;
	std::set<float> encounteredRuns;
	
	cout << "before loop" << endl;

	for(Long64_t iEnt=0; iEnt<nEntries; ++iEnt)
	{
		tree->GetEntry(iEnt); // Gets the next entry (filling the linked variables)
		lumiSec2Lumi_[run][ls]=lumiReported;
		encounteredRuns.insert(run);
	}

	// set buffers to correct size
	for(std::set<float>::iterator itr=encounteredRuns.begin(); itr!=encounteredRuns.end(); ++itr)
	{
		lumiSec2rate_[*itr].resize(nOfHists_);
		lumiSec2ratePrescaled_[*itr].resize(nOfHists_);
	}

}



//////////////////////////////// HELPERS /////////////////////////////////////////////


void L1RatesAnalyzer::getTopTriggers(std::vector<unsigned int>& topVec) {
	map<unsigned int, unsigned int> countMap;

	for (int i = 0; i < 10000; i++) {

		Long64_t ientry = LoadTree(i);
		if (ientry < 0)
			break;
		GetEntry(i);

		for(unsigned int k=0; k<192; ++k)
			countMap[k]=0;

		for (unsigned int k = 0; k < 192; k++) {
			

			const ULong64_t a1 = gt_->tw1.at(2) & bitMaskVec_[0];
			const ULong64_t a2 = gt_->tw2.at(2) & bitMaskVec_[1];
			const ULong64_t tt = gt_->tt.at(2) & bitMaskVec_[2];

			if (k < 64) {
				if (getBit(a1, k))
					countMap[k]++;
			} else if (k < 128) {
				if (getBit(a2, k-64))
					countMap[k]++;
			} else {
				if (getBit(tt, k-128))
					countMap[k]++;
			}
		}
	}


	for (int i = 0; i < nOfHists_; i++) {
		
		std::map<unsigned int, unsigned int>::iterator itr = std::max_element(countMap.begin(),
				countMap.end());

		topVec.push_back(itr->first);

		countMap.erase(itr);
	}

}

void L1RatesAnalyzer::drawPlotsForBitsCore(map<int, vector<map<int, float> > > dataMap, TCanvas* const)
{
	TLegend* l = new TLegend(0.6, 0.7, 0.85, 0.9);

	for (int i = 0; i < nOfHists_; i++) {

		TVectorF rateVec2;
		TVectorF lumiVec2;

		vector<float> rateVecCombined, lumiVecCombined;

		// loop over runs
		for (map<int, map<int, float> >::iterator itr =
				lumiSec2Lumi_.begin(); itr != lumiSec2Lumi_.end(); itr++) {

			vector<float> rateVec, lumiVec;

			map<int, float>& ls2lumi = itr->second;

			const int run = itr->first;

			cout << "RUN:" << run << endl;

			lumiVec.resize(ls2lumi.size());
			rateVec.resize(ls2lumi.size());


			// loop over lumi sections
			for(unsigned int k = 0; k < ls2lumi.size(); ++k)
			{
				if (dataMap[run].at(i)[k] != 0)
				{
					std::cout << "RUN: " << run << " k:" << k << " LUMI: "
							<< lumiSec2Lumi_[run][k] << ":"
							<< (dataMap[run].at(i)[k] * commonPrescale_
									/ lumiSectTimeNs_) << endl;
					if (!(dataMap[run].at(i)[k] == 0))
					{
						lumiVec[k] = lumiSec2Lumi_[run][k] / float(numberOfBunches_);
						rateVec[k] = (dataMap[run].at(i)[k]
								* commonPrescale_ / lumiSectTimeNs_);
					}
				}
			}

			for (unsigned int b = 0; b < lumiVec.size(); b++) {
				rateVecCombined.push_back(rateVec[b]);
				lumiVecCombined.push_back(lumiVec[b]);
			}

		}

		rateVec2.ResizeTo(lumiVecCombined.size());
		lumiVec2.ResizeTo(lumiVecCombined.size());

		for (unsigned int b = 0; b < lumiVecCombined.size(); b++) {
			if (rateVecCombined[b] != 0) {
				cout << "Rate: " << rateVecCombined[b] << ",LUMI: "
						<< lumiVecCombined[b] << endl;
				rateVec2[b] = rateVecCombined[b];
				lumiVec2[b] = lumiVecCombined[b];
			}
		}

		TGraph* g2 = new TGraph(lumiVec2, rateVec2);

		g2->GetXaxis()->SetLimits(xMin_, xMax_);

		g2->Fit("pol2", "", "", xFitMin_, xFitMax_);

		g2->Fit("pol2", "", "");

		std::cout << "Done " << endl;

		TF1 *fitFkt = g2->GetFunction("pol2");
		fitFkt->SetLineColor(tb_.getColor(i));

		// add entry to legend
		ostringstream oss2;
		oss2 << "Bit " << topVec_.at(i);

		l->AddEntry(fitFkt, oss2.str().c_str(), "l");

		l->SetHeader("Trigger rates");

		if (i == 0) {
			g2->Draw("A*");
			g2->GetXaxis()->SetTitle("Luminosity per bunch [10^{30} Hz/cm^{2}]");
			g2->GetYaxis()->SetTitle("Rate [Hz]");
			g2->GetYaxis()->SetTitleOffset(2.0);
			gPad->SetLeftMargin(0.15);
			gStyle->SetOptTitle(kFALSE);
			g2->SetTitle();

		} else {

			g2->Draw("*");
		}

	}

	l->Draw();
}


void  L1RatesAnalyzer::drawPlotsForBits() {

	TCanvas* c2 = new TCanvas("c2", "", 900, 900);
	c2->SetTitle("Rates");

	drawPlotsForBitsCore(lumiSec2rate_, c2);

	cout << "@@@@@@@@@@@ Draw prescaled plots @@@@@@@@@@@@@@" << endl;	

	TCanvas* c3 = new TCanvas("c3", "", 900, 900);
	c3->SetTitle("Rates prescaled");

	drawPlotsForBitsCore(lumiSec2ratePrescaled_,c3);

}


// returns a bitmask to extract bit i from another bitmask
ULong64_t L1RatesAnalyzer::bitmaskForBit(const unsigned int bit)
{
	if(bit>63)
		throw std::range_error("The parameter \"bit\" in L1RatesAnalyzer::bitmaskForBit has a value > 63.");

	return 1ULL<<bit;
}

// merges thre words to a 192 bit vector
void L1RatesAnalyzer::build192BitVec(const ULong64_t w1, const ULong64_t w2, const ULong64_t w3, vector<bool>& v)
{
	v.clear();
	v.reserve(192);

	for (unsigned int r = 0; r < 64; ++r)
		v.push_back(getBit(w1, r));

	for (unsigned int r = 0; r < 64; ++r)
		v.push_back(getBit(w2, r));

	for (unsigned int r = 0; r < 64; ++r)
		v.push_back(getBit(w3, r));
}

// extracts a bit from a ULL type
bool L1RatesAnalyzer::getBit(const ULong64_t val, const unsigned int bit)
{
	return (val & bitmaskForBit(bit))!=0;
}

