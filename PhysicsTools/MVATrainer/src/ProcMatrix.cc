#include <cstring>
#include <vector>
#include <memory>

#include <xercesc/dom/DOM.hpp>

#include <TMatrixD.h>
#include <TMatrixF.h>
#include <TH2.h>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"

#include "PhysicsTools/MVATrainer/interface/XMLDocument.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainer.h"
#include "PhysicsTools/MVATrainer/interface/TrainProcessor.h"
#include "PhysicsTools/MVATrainer/interface/LeastSquares.h"

XERCES_CPP_NAMESPACE_USE

using namespace PhysicsTools;

namespace { // anonymous

class ProcMatrix : public TrainProcessor {
    public:
	typedef TrainProcessor::Registry<ProcMatrix>::Type Registry;

	ProcMatrix(const char *name, const AtomicId *id,
	           MVATrainer *trainer);
	virtual ~ProcMatrix();

	virtual void configure(DOMElement *elem);
	virtual Calibration::VarProcessor *getCalibration() const;

	virtual void trainBegin();
	virtual void trainData(const std::vector<double> *values,
	                       bool target, double weight);
	virtual void trainEnd();

	virtual bool load();
	virtual void save();

    protected:
	virtual void *requestObject(const std::string &name) const;

    private:
	enum Iteration {
		ITER_FILL,
		ITER_DONE
	} iteration;

	typedef std::pair<unsigned int, double> Rank;

	std::vector<Rank> ranking() const;

	std::auto_ptr<LeastSquares>	lsSignal, lsBackground;
	std::auto_ptr<LeastSquares>	ls;
	std::vector<double>		vars;
	bool				fillSignal;
	bool				fillBackground;
	bool				doNormalization;
	bool				doRanking;
};

static ProcMatrix::Registry registry("ProcMatrix");

ProcMatrix::ProcMatrix(const char *name, const AtomicId *id,
                             MVATrainer *trainer) :
	TrainProcessor(name, id, trainer),
	iteration(ITER_FILL), fillSignal(true), fillBackground(true),
	doRanking(false)
{
}

ProcMatrix::~ProcMatrix()
{
}

void ProcMatrix::configure(DOMElement *elem)
{
	ls.reset(new LeastSquares(getInputs().size()));

	DOMNode *node = elem->getFirstChild();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (!node)
		return;

	if (std::strcmp(XMLSimpleStr(node->getNodeName()), "fill") != 0)
		throw cms::Exception("ProcMatrix")
				<< "Expected fill tag in config section."
				<< std::endl;

	elem = static_cast<DOMElement*>(node);

	fillSignal =
		XMLDocument::readAttribute<bool>(elem, "signal", false);
	fillBackground =
		XMLDocument::readAttribute<bool>(elem, "background", false);
	doNormalization =
		XMLDocument::readAttribute<bool>(elem, "normalize", false);

	doRanking = XMLDocument::readAttribute<bool>(elem, "ranking", false);
	if (doRanking)
		fillSignal = fillBackground = doNormalization = true;

	if (doNormalization && fillSignal && fillBackground) {
		lsSignal.reset(new LeastSquares(getInputs().size()));
		lsBackground.reset(new LeastSquares(getInputs().size()));
	}

	node = node->getNextSibling();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (node)
		throw cms::Exception("ProcMatrix")
			<< "Superfluous tags in config section."
			<< std::endl;

	if (!fillSignal && !fillBackground)
		throw cms::Exception("ProcMatrix")
			<< "Filling neither background nor signal in config."
			<< std::endl;
}

Calibration::VarProcessor *ProcMatrix::getCalibration() const
{
	if (doRanking)
		return 0;

	Calibration::ProcMatrix *calib = new Calibration::ProcMatrix;

	unsigned int n = ls->getSize();
	const TMatrixD &rotation = ls->getRotation();

	calib->matrix.rows = n;
	calib->matrix.columns = n;

	for(unsigned int i = 0; i < n; i++)
		for(unsigned int j = 0; j < n; j++)
			calib->matrix.elements.push_back(rotation(j, i));

	return calib;
}

void ProcMatrix::trainBegin()
{
	if (iteration == ITER_FILL)
		vars.resize(ls->getSize());
}

void ProcMatrix::trainData(const std::vector<double> *values,
                           bool target, double weight)
{
	if (iteration != ITER_FILL)
		return;

	if (!(target ? fillSignal : fillBackground))
		return;

	LeastSquares *ls = target ? lsSignal.get() : lsBackground.get();
	if (!ls)
		ls = this->ls.get();

	for(unsigned int i = 0; i < ls->getSize(); i++, values++) {
		if (values->empty())
			throw cms::Exception("ProcMatrix")
				<< "Variable \""
				<< (const char*)getInputs().get()[i]->getName()
				<< "\" is not set in ProcMatrix trainer."
				<< std::endl;
		vars[i] = values->front();
	}

	ls->add(vars, target, weight);
}

void ProcMatrix::trainEnd()
{
	switch(iteration) {
	    case ITER_FILL:
		vars.clear();
		if (lsSignal.get()) {
			unsigned int n = ls->getSize();
			double weight = lsSignal->getCoefficients()
								(n + 1, n + 1);
			if (weight > 1.0e-9)
				ls->add(*lsSignal, 1.0 / weight);
			lsSignal.reset();
		}
		if (lsBackground.get()) {
			unsigned int n = ls->getSize();
			double weight = lsBackground->getCoefficients()
								(n + 1, n + 1);
			if (weight > 1.0e-9)
				ls->add(*lsBackground, 1.0 / weight);
			lsBackground.reset();
		}
		ls->calculate();

		iteration = ITER_DONE;
		trained = true;
		break;

	    default:
		/* shut up */;
	}

	if (iteration == ITER_DONE && monitoring) {
		TMatrixF matrix(ls->getCorrelations());
		TH2F *histo = monitoring->book<TH2F>("CorrMatrix", matrix);
		histo->SetNameTitle("CorrMatrix",
			(fillSignal && fillBackground)
			? "correlation matrix (signal + background)"
			: (fillSignal ? "correlation matrix (signal)"
			              : "correlation matrix (background)"));

		std::vector<SourceVariable*> inputs = getInputs().get();
		for(std::vector<SourceVariable*>::const_iterator iter =
			inputs.begin(); iter != inputs.end(); ++iter) {

			unsigned int idx = iter - inputs.begin();
			SourceVariable *var = *iter;
			std::string name =
				(const char*)var->getSource()->getName()
				+ std::string("_")
				+ (const char*)var->getName();

			histo->GetXaxis()->SetBinLabel(idx + 1, name.c_str());
			histo->GetYaxis()->SetBinLabel(idx + 1, name.c_str());
			histo->GetXaxis()->SetBinLabel(inputs.size() + 1,
			                               "target");
			histo->GetYaxis()->SetBinLabel(inputs.size() + 1,
			                               "target");
		}
		histo->LabelsOption("d");
		histo->SetMinimum(-1.0);
		histo->SetMaximum(+1.0);

		if (!doRanking)
			return;

		std::vector<Rank> ranks = ranking();
		TVectorD rankVector(ranks.size());
		for(unsigned int i = 0; i < ranks.size(); i++)
			rankVector[i] = ranks[i].second;
		TH1F *rank = monitoring->book<TH1F>("Ranking", rankVector);
		rank->SetNameTitle("Ranking", "variable ranking");
		rank->SetYTitle("correlation to target");
		for(unsigned int i = 0; i < ranks.size(); i++) {
			unsigned int v = ranks[i].first;
			std::string name;
			SourceVariable *var = inputs[v];
			name = (const char*)var->getSource()->getName()
			       + std::string("_")
			       + (const char*)var->getName();
			rank->GetXaxis()->SetBinLabel(i + 1, name.c_str());
		}
	}
}

void *ProcMatrix::requestObject(const std::string &name) const
{
	if (name == "linearAnalyzer")
		return static_cast<void*>(ls.get());

	return 0;
}

bool ProcMatrix::load()
{
	std::string filename = trainer->trainFileName(this, "xml");
	if (!exists(filename))
		return false;

	XMLDocument xml(filename);
	DOMElement *elem = xml.getRootNode();
	if (std::strcmp(XMLSimpleStr(elem->getNodeName()), "ProcMatrix") != 0)
		throw cms::Exception("ProcMatrix")
			<< "XML training data file has bad root node."
			<< std::endl;

	DOMNode *node = elem->getFirstChild();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (!node)
		throw cms::Exception("ProcMatrix")
			<< "Train data file empty." << std::endl;

	ls->load(static_cast<DOMElement*>(node));

	node = elem->getNextSibling();
	while(node && node->getNodeType() != DOMNode::ELEMENT_NODE)
		node = node->getNextSibling();

	if (node)
		throw cms::Exception("ProcMatrix")
			<< "Train data file contains superfluous tags."
			<< std::endl;

	iteration = ITER_DONE;
	trained = true;
	return true;
}

void ProcMatrix::save()
{
	XMLDocument xml(trainer->trainFileName(this, "xml"), true);
	DOMDocument *doc = xml.createDocument("ProcMatrix");

	xml.getRootNode()->appendChild(ls->save(doc));
}

static void maskLine(TMatrixDSym &m, unsigned int line)
{
	unsigned int n = m.GetNrows();
	for(unsigned int i = 0; i < n; i++)
		m(i, line) = m(line, i) = 0.;
	m(line, line) = 1.;
}

static void restoreLine(TMatrixDSym &m, TMatrixDSym &o, unsigned int line)
{
	unsigned int n = m.GetNrows();
	for(unsigned int i = 0; i < n; i++) {
		m(i, line) = o(i, line);
		m(line, i) = o(line, i);
	}
}

static double targetCorrelation(const TMatrixDSym &coeffs,
                                const std::vector<bool> &use)
{
	unsigned int n = coeffs.GetNrows() - 2;
	
	TVectorD weights = LeastSquares::solveFisher(coeffs);
	weights.ResizeTo(n + 2);
	weights[n + 1] = weights[n];
	weights[n] = 0.;

	double v1 = 0.;
	double v2 = 0.;
	double v3 = coeffs(n, n);
	double N = coeffs(n + 1, n + 1);
	double M = 0.;
	for(unsigned int i = 0; i < n + 2; i++) {
		if (i < n && !use[n])
			continue;
		double w = weights[i];
		for(unsigned int j = 0; j < n + 2; j++) {
			if (i < n && !use[n])
				continue;
			v1 += w * weights[j] * coeffs(i, j);
		}
		v2 += w * coeffs(i, n);
		M += w * coeffs(i, n + 1);
	}

	double c1 = v1 * N - M * M;
	double c2 = v2 * N - M * coeffs(n + 1, n);
	double c3 = v3 * N - coeffs(n + 1, n) * coeffs(n + 1, n);

	double c = c1 * c3;
	return (c > 1.0e-9) ? c2 / std::sqrt(c) : 0.0;
}

std::vector<ProcMatrix::Rank> ProcMatrix::ranking() const
{
	TMatrixDSym coeffs = ls->getCoefficients();
	unsigned int n = coeffs.GetNrows() - 2;

	typedef std::pair<unsigned int, double> Rank;
	std::vector<Rank> ranking;
	std::vector<bool> use(n, true);

	double corr = targetCorrelation(coeffs, use);

	for(unsigned int nVars = n; nVars > 1; nVars--) {
		double bestCorr = -99999.0;
		unsigned int bestIdx = n;
		TMatrixDSym origCoeffs = coeffs;

		for(unsigned int i = 0; i < n; i++) {
			if (!use[i])
				continue;

			use[i] = false;
			maskLine(coeffs, i);
			double newCorr = targetCorrelation(coeffs, use);
			use[i] = true;
			restoreLine(coeffs, origCoeffs, i);

			if (newCorr > bestCorr) {
				bestCorr = newCorr;
				bestIdx = i;
			}
		}

		ranking.push_back(Rank(bestIdx, corr));
		corr = bestCorr;
		use[bestIdx] = false;
		maskLine(coeffs, bestIdx);
	}

	for(unsigned int i = 0; i < n; i++)
		if (use[i])
			ranking.push_back(Rank(i, corr));

	return ranking;
}

} // anonymous namespace
