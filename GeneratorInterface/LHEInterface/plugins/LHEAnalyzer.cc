#include <functional>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <cmath>

#include <boost/bind.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

#include <TH1.h>

#include <HepMC/GenEvent.h>
#include <HepMC/SimpleVector.h>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "GeneratorInterface/LHEInterface/interface/JetInput.h"
#include "GeneratorInterface/LHEInterface/interface/JetClustering.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

using namespace lhef;

class LHEAnalyzer : public edm::EDAnalyzer {
    public:
	explicit LHEAnalyzer(const edm::ParameterSet &params);
	virtual ~LHEAnalyzer();

    protected:
	virtual void analyze(const edm::Event &event,
	                     const edm::EventSetup &es);


    private:
	void fillDJRSched(unsigned int min, unsigned int max);

	edm::InputTag				sourceLabel;
	JetInput				jetInput;

	double					defaultDeltaCut;
	double					defaultPtCut;
	double					maxEta;
	bool					useEt;
	double					ptFraction;

	unsigned int				binsDelta;
	double					minDelta;
	double					maxDelta;
	unsigned int				binsPt;
	double					minPt;
	double					maxPt;
	unsigned int				minDJR;
	unsigned int				maxDJR;

	boost::ptr_vector<JetClustering>	deltaClustering;
	std::auto_ptr<JetClustering>		ptClustering;
	std::vector<unsigned int>		djrSched;

	std::vector<TH1*>			histoDelta;
	std::vector<TH1*>			histoPt;
};

LHEAnalyzer::LHEAnalyzer(const edm::ParameterSet &params) :
	sourceLabel(params.getParameter<edm::InputTag>("src")),
	jetInput(params.getParameter<edm::ParameterSet>("jetInput")),
	defaultDeltaCut(params.getParameter<double>("defaultDeltaCut")),
	defaultPtCut(params.getParameter<double>("defaultPtCut")),
	maxEta(params.getParameter<double>("maxEta")),
	useEt(params.getParameter<bool>("useEt")),
	ptFraction(params.getUntrackedParameter<double>("ptFraction", 0.75)),
	binsDelta(params.getParameter<unsigned int>("binsDelta")),
	minDelta(params.getParameter<double>("minDelta")),
	maxDelta(params.getParameter<double>("maxDelta")),
	binsPt(params.getParameter<unsigned int>("binsPt")),
	minPt(params.getParameter<double>("minPt")),
	maxPt(params.getParameter<double>("maxPt")),
	minDJR(params.getParameter<unsigned int>("minDJR")),
	maxDJR(params.getParameter<unsigned int>("maxDJR"))
{
	edm::ParameterSet jetClusPSet =
		params.getParameter<edm::ParameterSet>("jetClustering");

	for(unsigned int i = 0; i <= binsDelta; i++) {
		double deltaCut =
			minDelta + (maxDelta - minDelta) * i / binsDelta;
		jetClusPSet.addParameter("coneRadius", deltaCut);
		edm::ParameterSet tmp;
		tmp.addParameter("algorithm", jetClusPSet);
		deltaClustering.push_back(
			new JetClustering(tmp, defaultPtCut * ptFraction));
	}

	jetClusPSet.addParameter("coneRadius", defaultDeltaCut);
	edm::ParameterSet tmp;
	tmp.addParameter("algorithm", jetClusPSet);
	ptClustering.reset(new JetClustering(tmp, minPt * ptFraction));

	fillDJRSched(minDJR <= 0 ? 1 : minDJR, maxDJR - 1);

	edm::Service<TFileService> fs;
	for(unsigned int i = minDJR; i < maxDJR; i++) {
		std::ostringstream ss, ss2;
		ss << (i + 1) << "#rightarrow" << i << " jets";
		ss2 << i;
		TH1 *h = fs->make<TH1D>(("delta" + ss2.str()).c_str(),
		                        ("DJR " + ss.str()).c_str(),
		                        binsDelta, minDelta, maxDelta);
		h->SetXTitle("p_{T} [GeV/c^2]");
		h->SetYTitle("#delta#sigma [mb]");

		if (i == 0) {
			h->Delete();
			h = 0;
		}
		histoDelta.push_back(h);

		std::string what = useEt ? "E" : "p";
		h = fs->make<TH1D>(("pt" + ss2.str()).c_str(),
		                   ("DJR " + ss.str()).c_str(), binsPt,
		                   std::log10(minPt), std::log10(maxPt));
		h->SetXTitle(("log_{10}(" + what +
		              		"_{T} [GeV/c^{2}])").c_str());
		h->SetYTitle("#delta#sigma [mb]");

		histoPt.push_back(h);
	}
}

LHEAnalyzer::~LHEAnalyzer()
{
}

void LHEAnalyzer::fillDJRSched(unsigned int min, unsigned int max)
{
	unsigned int middle = (min + max) / 2;

	djrSched.push_back(middle);

	if (min < middle)
		fillDJRSched(min, middle - 1);
	if (middle < max)
		fillDJRSched(middle + 1, max);
}

void LHEAnalyzer::analyze(const edm::Event &event, const edm::EventSetup &es)
{
	using boost::bind;
	typedef JetClustering::Jet Jet;

	edm::Handle<edm::HepMCProduct> hepmc;
	event.getByLabel(sourceLabel, hepmc);

	std::auto_ptr<HepMC::GenEvent> clonedEvent;
	const HepMC::GenEvent *genEvent = hepmc->GetEvent();
	if (!genEvent->signal_process_vertex()) {
		clonedEvent.reset(new HepMC::GenEvent(*genEvent));
		const HepMC::GenVertex *signalVertex =
			LHEEvent::findSignalVertex(clonedEvent.get());
		clonedEvent->set_signal_process_vertex(
			const_cast<HepMC::GenVertex*>(signalVertex));
		genEvent = clonedEvent.get();
	}

	JetInput::ParticleVector particles = jetInput(genEvent);

	std::vector<Jet> ptJets = (*ptClustering)(particles);
	std::sort(ptJets.begin(), ptJets.end(),
	          bind(std::greater<double>(),
	               bind(useEt ? &Jet::et : &Jet::pt, _1),
	               bind(useEt ? &Jet::et : &Jet::pt, _2)));

	typedef std::pair<int, int> Pair;
	std::vector<Pair> deltaJets(maxDJR - minDJR + 1,
	                            Pair(-1, binsDelta + 1));

	for(std::vector<unsigned int>::const_iterator djr = djrSched.begin();
	    djr != djrSched.end(); ++djr) {
//std::cout << "DJR schedule " << (*djr + 1) << " -> " << *djr << std::endl;
		int result = -1;
		for(;;) {
//for(int i = minDJR; i <= maxDJR; i++)
//std::cout << "+++ " << i << ": (" << deltaJets[i - minDJR].first << ", " << deltaJets[i - minDJR].second << ")" << std::endl;
			int upper = binsDelta + 1;
			for(int i = *djr; i >= (int)minDJR; i--) {
				if (deltaJets[i - minDJR].second <=
							(int)binsDelta) {
					upper = deltaJets[i - minDJR].second;
					break;
				}
			}
			int lower = -1;
			for(int i = *djr + 1; i <= (int)maxDJR; i++) {
				if (deltaJets[i - minDJR].first >= 0) {
					lower = deltaJets[i - minDJR].first;
					break;
				}
			}
//std::cout << "\t" << lower << " < " << upper << std::endl;

			result = (lower + upper + 2) / 2 - 1;
			if (result == lower)
				break;
			else if (result < lower) {
				result = -1;
				break;
			}

			std::vector<Jet> jets =
				deltaClustering[result](particles);
			unsigned int nJets = 0;
			for(std::vector<Jet>::const_iterator iter =
				jets.begin(); iter != jets.end(); ++iter)
				if ((useEt ? iter->et() : iter->pt())
							> defaultPtCut)
					nJets++;

//std::cout << "\t---(" << *djr << ")--> bin " << result << ": " << nJets << " jets" << std::endl;

			if (nJets < minDJR)
				nJets = minDJR;
			else if (nJets > maxDJR)
				nJets = maxDJR;

			for(int j = nJets; j >= (int)minDJR; j--) {
				if (deltaJets[j - minDJR].first < 0 ||
				    result > deltaJets[j - minDJR].first)
					deltaJets[j - minDJR].first = result;
			}
			for(int j = nJets; j <= (int)maxDJR; j++) {
				if (deltaJets[j - minDJR].second <
							(int)binsDelta ||
				    result < deltaJets[j - minDJR].second)
					deltaJets[j - minDJR].second = result;
			}
		}

//std::cout << "final " << *djr << ": " << result << std::endl;
		TH1 *h = histoDelta[*djr - minDJR];
		h->Fill(h->GetBinCenter(result + 1));

		h = histoPt[*djr - minDJR];
		if (*djr >= ptJets.size())
			h->Fill(-999.0);
		else if (useEt)
			h->Fill(std::log10(ptJets[*djr].et()));
		else
			h->Fill(std::log10(ptJets[*djr].pt()));
	}

	if (minDJR <= 0) {
		TH1 *h = histoPt[0];
		if (minDJR >= ptJets.size())
			h->Fill(-999.0);
		else if (useEt)
			h->Fill(std::log10(ptJets[minDJR].et()));
		else
			h->Fill(std::log10(ptJets[minDJR].pt()));
	}
}

DEFINE_FWK_MODULE(LHEAnalyzer);
