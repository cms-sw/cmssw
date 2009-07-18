#include <iostream>
#include <iomanip>
#include <cstring>
#include <vector>
#include <cmath>

#include <TMatrixD.h>
#include <TVectorD.h>
#include <TDecompSVD.h>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVATrainer/interface/XMLDocument.h"
#include "PhysicsTools/MVATrainer/interface/XMLSimpleStr.h"
#include "PhysicsTools/MVATrainer/interface/XMLUniStr.h"
#include "PhysicsTools/MVATrainer/interface/LeastSquares.h"

XERCES_CPP_NAMESPACE_USE

namespace PhysicsTools {

LeastSquares::LeastSquares(unsigned int n) :
	coeffs(n + 1), covar(n), corr(n), rotation(n, n),
	sums(n + 1), weights(n + 1), variance(n), trace(n), n(n)
{
}

LeastSquares::~LeastSquares()
{
}

void LeastSquares::add(const std::vector<double> &values, double dest,
                       double weight)
{
	if (values.size() != n)
		throw cms::Exception("LeastSquares")
			<< "add(): invalid array size!" << std::endl;

	for(unsigned int i = 0; i < n; i++) {
		for(unsigned int j = 0; j < n; j++)
			coeffs(i, j) += values[i] * values[j] * weight;

		coeffs(i, n) += values[i] * weight;
		coeffs(n, i) += values[i] * weight;
		sums[i] += values[i] * dest * weight;
	} 

	coeffs(n, n) += weight;
	sums[n] += dest * weight;
}

void LeastSquares::calculate()
{
	double N = coeffs(n, n);

	for(unsigned int i = 0; i < n; i++)
		for(unsigned int j = 0; j < n; j++)
			covar(i, j) = coeffs(i, j) * N - coeffs(i, n) * coeffs(n, j);

	for(unsigned int i = 0; i < n; i++) {
		double c = covar(i, i);
		variance[i] = c > 0.0 ? std::sqrt(c) : c;
	}

	for(unsigned int i = 0; i < n; i++) {
		for(unsigned int j = 0; j < n; j++) {
			double v = variance[i] * variance[j];

			corr(i, j) = (v >= 1.0e-9) ? (covar(i, j) / v) : (i == j ? 1.0 : 0.0);
		}
	}

	TDecompSVD decCoeffs(coeffs);
	bool ok;
	weights = decCoeffs.Solve(sums, ok);

	TDecompSVD decCovar(covar);
	rotation = decCovar.GetU();
	trace = decCovar.GetSig();
}

std::vector<double> LeastSquares::getWeights() const
{
	std::vector<double> results;
	
	for(unsigned int i = 0; i < n; i++)
		results.push_back(weights[i]);
	
	return results;
}

std::vector<double> LeastSquares::getMeans() const
{
	std::vector<double> results;

	double N = coeffs[n][n];
	for(unsigned int i = 0; i < n; i++)
		results.push_back(coeffs[i][n] / N);
	
	return results;
}

double LeastSquares::getConstant() const
{
	return weights[n];
}

static void loadMatrix(DOMElement *elem, unsigned int n, TMatrixDBase &matrix)
{
	if (std::strcmp(XMLSimpleStr(elem->getNodeName()),
	                             "matrix") != 0)
		throw cms::Exception("LeastSquares")
			<< "Expected matrix in data file."
			<< std::endl;

	unsigned int row = 0;
	for(DOMNode *node = elem->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		if (std::strcmp(XMLSimpleStr(node->getNodeName()),
		                             "row") != 0)
			throw cms::Exception("LeastSquares")
				<< "Expected row tag in data file."
				<< std::endl;

		if (row >= n)
			throw cms::Exception("LeastSquares")
				<< "Too many rows in data file." << std::endl;

		elem = static_cast<DOMElement*>(node);

		unsigned int col = 0;
		for(DOMNode *subNode = elem->getFirstChild();
		    subNode; subNode = subNode->getNextSibling()) {
			if (subNode->getNodeType() != DOMNode::ELEMENT_NODE)
				continue;

			if (std::strcmp(XMLSimpleStr(subNode->getNodeName()),
			                             "value") != 0)
			throw cms::Exception("LeastSquares")
				<< "Expected value tag in data file."
				<< std::endl;

			if (col >= n)
				throw cms::Exception("LeastSquares")
					<< "Too many columns in data file."
					<< std::endl;

			matrix(row, col) =
				XMLDocument::readContent<double>(subNode);
			col++;
		}

		if (col != n)
			throw cms::Exception("LeastSquares")
				<< "Missing columns in data file."
				<< std::endl;
		row++;
	}

	if (row != n)
		throw cms::Exception("LeastSquares")
			<< "Missing rows in data file."
			<< std::endl;
}

static void loadVector(DOMElement *elem, unsigned int n, TVectorD &vector)
{
	if (std::strcmp(XMLSimpleStr(elem->getNodeName()),
	                             "vector") != 0)
		throw cms::Exception("LeastSquares")
			<< "Expected matrix in data file."
			<< std::endl;

	unsigned int col = 0;
	for(DOMNode *node = elem->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		if (std::strcmp(XMLSimpleStr(node->getNodeName()),
		                             "value") != 0)
		throw cms::Exception("LeastSquares")
			<< "Expected value tag in data file."
			<< std::endl;

		if (col >= n)
			throw cms::Exception("LeastSquares")
				<< "Too many columns in data file."
				<< std::endl;

		vector(col) = XMLDocument::readContent<double>(node);
		col++;
	}

	if (col != n)
		throw cms::Exception("LeastSquares")
			<< "Missing columns in data file."
			<< std::endl;
}

static DOMElement *saveMatrix(DOMDocument *doc, unsigned int n,
                              const TMatrixDBase &matrix)
{
	DOMElement *root = doc->createElement(XMLUniStr("matrix"));
	XMLDocument::writeAttribute<unsigned int>(root, "size", n);

	for(unsigned int i = 0; i < n; i++) {
		DOMElement *row = doc->createElement(XMLUniStr("row"));
		root->appendChild(row);

		for(unsigned int j = 0; j < n; j++) {
			DOMElement *value =
				doc->createElement(XMLUniStr("value"));
			row->appendChild(value);

			XMLDocument::writeContent<double>(value, doc,
			                                  matrix(i, j));
		}
	}

	return root;
}

static DOMElement *saveVector(DOMDocument *doc, unsigned int n,
                              const TVectorD &vector)
{
	DOMElement *root = doc->createElement(XMLUniStr("vector"));
	XMLDocument::writeAttribute<unsigned int>(root, "size", n);

	for(unsigned int i = 0; i < n; i++) {
		DOMElement *value =
			doc->createElement(XMLUniStr("value"));
		root->appendChild(value);

		XMLDocument::writeContent<double>(value, doc, vector(i));
	}

	return root;
}

void LeastSquares::load(DOMElement *elem)
{
	if (std::strcmp(XMLSimpleStr(elem->getNodeName()),
	                             "LinearAnalysis") != 0)
		throw cms::Exception("LeastSquares")
			<< "Expected LinearAnalysis in data file."
			<< std::endl;

	enum Position {
		POS_COEFFS, POS_COVAR, POS_CORR, POS_ROTATION,
		POS_SUMS, POS_WEIGHTS, POS_VARIANCE, POS_TRACE, POS_DONE
	} pos = POS_COEFFS;


	for(DOMNode *node = elem->getFirstChild();
	    node; node = node->getNextSibling()) {
		if (node->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;

		DOMElement *elem = static_cast<DOMElement*>(node);

		switch(pos) {
		    case POS_COEFFS:
			loadMatrix(elem, n + 1, coeffs);
			break;
		    case POS_COVAR:
			loadMatrix(elem, n, covar);
			break;
		    case POS_CORR:
			loadMatrix(elem, n, corr);
			break;
		    case POS_ROTATION:
			loadMatrix(elem, n, rotation);
			break;
		    case POS_SUMS:
			loadVector(elem, n + 1, sums);
			break;
		    case POS_WEIGHTS:
			loadVector(elem, n + 1, weights);
			break;
		    case POS_VARIANCE:
			loadVector(elem, n, variance);
			break;
		    case POS_TRACE:
			loadVector(elem, n, trace);
			break;
		    default:
			throw cms::Exception("LeastSquares")
				<< "Superfluous content in data file."
				<< std::endl;
		}

		pos = (Position)(pos + 1);
	}

	if (pos != POS_DONE)
		throw cms::Exception("LeastSquares")
			<< "Missing objects in data file."
			<< std::endl;
}

DOMElement *LeastSquares::save(DOMDocument *doc) const
{
	DOMElement *root = doc->createElement(XMLUniStr("LinearAnalysis"));
	XMLDocument::writeAttribute<unsigned int>(root, "size", n);

	root->appendChild(saveMatrix(doc, n + 1, coeffs));
	root->appendChild(saveMatrix(doc, n, covar));
	root->appendChild(saveMatrix(doc, n, corr));
	root->appendChild(saveMatrix(doc, n, rotation));
	root->appendChild(saveVector(doc, n + 1, sums));
	root->appendChild(saveVector(doc, n + 1, weights));
	root->appendChild(saveVector(doc, n, variance));
	root->appendChild(saveVector(doc, n, trace));

	return root;
}

} // namespace PhysicsTools
