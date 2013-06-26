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
	coeffs(n + 2), covar(n + 1), corr(n + 1), rotation(n, n),
	weights(n + 1), variance(n + 1), trace(n), n(n)
{
}

LeastSquares::~LeastSquares()
{
}

void LeastSquares::add(const std::vector<double> &values,
                       double dest, double weight)
{
	if (values.size() != n)
		throw cms::Exception("LeastSquares")
			<< "add(): invalid array size!" << std::endl;

	for(unsigned int i = 0; i < n; i++) {
		for(unsigned int j = 0; j < n; j++)
			coeffs(i, j) += values[i] * values[j] * weight;

		coeffs(n, i) += values[i] * dest * weight;
		coeffs(i, n) += values[i] * dest * weight;
		coeffs(n + 1, i) += values[i] * weight;
		coeffs(i, n + 1) += values[i] * weight;
	} 

	coeffs(n, n) += dest * dest * weight;
	coeffs(n + 1, n) += dest * weight;
	coeffs(n, n + 1) += dest * weight;
	coeffs(n + 1, n + 1) += weight;
}

void LeastSquares::add(const LeastSquares &other, double weight)
{
	if (other.getSize() != n)
		throw cms::Exception("LeastSquares")
			<< "add(): invalid array size!" << std::endl;

	coeffs += weight * other.coeffs;
}

TVectorD LeastSquares::solveFisher(const TMatrixDSym &coeffs)
{
	unsigned int n = coeffs.GetNrows() - 2;

	TMatrixDSym tmp;
	coeffs.GetSub(0, n, tmp);
	tmp[n] = TVectorD(n + 1, coeffs[n + 1].GetPtr());
	tmp(n, n) = coeffs(n + 1, n + 1);

	TDecompSVD decCoeffs(tmp);
	bool ok;
	return decCoeffs.Solve(TVectorD(n + 1, coeffs[n].GetPtr()), ok);
}

TMatrixD LeastSquares::solveRotation(const TMatrixDSym &covar, TVectorD &trace)
{
	TMatrixDSym tmp;
	covar.GetSub(0, covar.GetNrows() - 2, tmp);
	TDecompSVD decCovar(tmp);
	trace = decCovar.GetSig();
	return decCovar.GetU();
}

void LeastSquares::calculate()
{
	double N = coeffs(n + 1, n + 1);

	for(unsigned int i = 0; i <= n; i++) {
		double M = coeffs(n + 1, i);
		for(unsigned int j = 0; j <= n; j++)
			covar(i, j) = coeffs(i, j) * N - M * coeffs(n + 1, j);
	}

	for(unsigned int i = 0; i <= n; i++) {
		double c = covar(i, i);
		variance[i] = c > 0.0 ? std::sqrt(c) : 0.0;
	}

	for(unsigned int i = 0; i <= n; i++) {
		double M = variance[i];
		for(unsigned int j = 0; j <= n; j++) {
			double v = M * variance[j];
			double w = covar(i, j);

			corr(i, j) = (v >= 1.0e-9) ? (w / v) : (i == j);
		}
	}

	weights = solveFisher(coeffs);
	rotation = solveRotation(covar, trace);
}

std::vector<double> LeastSquares::getWeights() const
{
	std::vector<double> results;
	results.reserve(n);
	
	for(unsigned int i = 0; i < n; i++)
		results.push_back(weights[i]);
	
	return results;
}

std::vector<double> LeastSquares::getMeans() const
{
	std::vector<double> results;
	results.reserve(n);

	double N = coeffs(n + 1, n + 1);
	for(unsigned int i = 0; i < n; i++)
		results.push_back(coeffs(n + 1, i) / N);

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

	unsigned int version = XMLDocument::readAttribute<unsigned int>(
							elem, "version", 1);

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
			if (version < 2) {
				loadMatrix(elem, n + 1, coeffs);
				coeffs.ResizeTo(n + 2, n + 2);
				for(unsigned int i = 0; i <= n; i++) {
					coeffs(n + 1, i) = coeffs(n, i);
					coeffs(i, n + 1) = coeffs(i, n);
				}
				coeffs(n + 1, n + 1) = coeffs(n + 1, n);
			} else
				loadMatrix(elem, n + 2, coeffs);
			break;
		    case POS_COVAR:
			if (version < 2)
				loadMatrix(elem, n, covar);
			else
				loadMatrix(elem, n + 1, covar);
			break;
		    case POS_CORR:
			if (version < 2)
				loadMatrix(elem, n, corr);
			else
				loadMatrix(elem, n + 1, corr);
			break;
		    case POS_ROTATION:
			loadMatrix(elem, n, rotation);
			break;
		    case POS_SUMS:
			if (version < 2) {
				TVectorD tmp(n + 1);
				loadVector(elem, n + 1, tmp);

				double M = coeffs(n + 1, n);
				double N = coeffs(n + 1, n + 1);

				for(unsigned int i = 0; i <= n; i++) {
					double v = coeffs(n, i) * N -
					           M * coeffs(n + 1, i);
					double w = coeffs(n, i) * N - v;

					covar(n, i) = w;
					covar(i, n) = w;
				}

				break;
			} else
				pos = (Position)(pos + 1);
		    case POS_WEIGHTS:
			loadVector(elem, n + 1, weights);
			break;
		    case POS_VARIANCE:
			if (version < 2) {
				loadVector(elem, n, variance);

				double M = covar(n, n);
				M = M > 0.0 ? std::sqrt(M) : 0.0;
				variance[n] = M;

				for(unsigned int i = 0; i <= n; i++) {
					double v = M * variance[i];
					double w = covar(n, i);
					double c = (v >= 1.0e-9)
							? (w / v) : (i == n);

					corr(n, i) = c;
					corr(i, n) = c;
				}
			} else
				loadVector(elem, n + 1, variance);
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
	XMLDocument::writeAttribute<unsigned int>(root, "version", 2);
	XMLDocument::writeAttribute<unsigned int>(root, "size", n);

	root->appendChild(saveMatrix(doc, n + 2, coeffs));
	root->appendChild(saveMatrix(doc, n + 1, covar));
	root->appendChild(saveMatrix(doc, n + 1, corr));
	root->appendChild(saveMatrix(doc, n, rotation));
	root->appendChild(saveVector(doc, n + 1, weights));
	root->appendChild(saveVector(doc, n + 1, variance));
	root->appendChild(saveVector(doc, n, trace));

	return root;
}

} // namespace PhysicsTools
