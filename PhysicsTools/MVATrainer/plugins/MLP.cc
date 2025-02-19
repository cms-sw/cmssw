#include <iostream>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "FWCore/Utilities/interface/Exception.h"

#include "MLP.h"

#include "mlp_gen.h"

namespace PhysicsTools {

bool MLP::inUse = false;

static std::vector<std::string> split(const std::string line, char delim)
{
	const char *str = line.c_str();
	const char *p = str;

	std::vector<std::string> tokens;

	if (line[0] == '\0')
		return tokens;

	while(p) {
		const char *q = std::strchr(p, delim);

		if (!q) {
			tokens.push_back(std::string(p));
			p = 0;
		} else {
			tokens.push_back(std::string(p, q - p));
			p = q + 1;
		}
	}

	return tokens;
}

MLP::MLP(unsigned int nIn, unsigned int nOut, const std::string layout_) :
	initialized(false), layers(0), layout(0), epoch(0)
{
	if (inUse)
		throw cms::Exception("MLP")
			<< "mlpfit doesn't support more than one instance."
			<< std::endl;

	std::vector<std::string> parsed = split(layout_, ':');
	if (parsed.size() < 1)
		throw cms::Exception("MLP")
			<< "Invalid layout." << std::endl;

	layout = new int[parsed.size() + 2];

	layers = parsed.size();
	layout[0] = (int)nIn;
	for(int i = 0; i < layers; i++) {
		std::istringstream ss(parsed[i]);
		int nodes;
		ss >> nodes;
		if (nodes < 1)
			throw cms::Exception("MLP")
				<< "Invalid layout." << std::endl;

		layout[i + 1] = nodes;
	}
	layout[layers + 1] = (int)nOut;
	layers += 2;

	inUse = true;

	MLP_SetNet(&layers, layout);
	setLearn();
	LearnAlloc();
	InitWeights();
}

MLP::~MLP()
{
	clear();

	LearnFree();
	inUse = false;
	delete[] layout;
}

void MLP::clear()
{
	if (!initialized)
		return;
	initialized = false;

	FreePatterns(0);
	free(PAT.Rin);
	free(PAT.Rans);
	free(PAT.Pond);
}

void MLP::setLearn(void)
{
	LEARN.Meth = 7;
	LEARN.Nreset = 50;
	LEARN.Tau = 1.5;
	LEARN.Decay = 1.0;
	LEARN.eta = 0.1;
	LEARN.Lambda = 1.0;
	LEARN.delta = 0.0;
	LEARN.epsilon = 0.2;
}

void MLP::setNPattern(unsigned int size)
{
	PAT.Npat[0] = (int)size;
	PAT.Npat[1] = 0;
	PAT.Nin = layout[0];
	PAT.Nout = layout[layers - 1];
}

void MLP::init(unsigned int rows)
{
	setNPattern(rows);
	AllocPatterns(0, rows, layout[0], layout[layers - 1], 0);
	initialized = true;
}

void MLP::set(unsigned int row, double *data, double *target, double weight)
{
	int nIn = layout[0];
	int nOut = layout[layers - 1];

	std::memcpy(&PAT.vRin[0][row*(nIn + 1) + 1], data, sizeof(double) * nIn);
	std::memcpy(&PAT.Rans[0][row][0], target, sizeof(double) * nOut);
	PAT.Pond[0][row] = weight;
}

double MLP::train()
{
	double alpMin;
	int nTest;

	return MLP_Epoch(++epoch, &alpMin, &nTest);
}

const double *MLP::eval(double *data) const
{
	MLP_Out_T(data);

	return &NET.Outn[layers - 1][0];
}

void MLP::save(const std::string file) const
{
	if (SaveWeights(const_cast<char*>(file.c_str()), (int)epoch) < 0)
		throw cms::Exception("MLP")
			<< "Error opening \"" << file << "\"." << std::endl;
}

void MLP::load(const std::string file)
{
	int epoch_ = 0;
	if (LoadWeights(const_cast<char*>(file.c_str()), &epoch_) < 0)
		throw cms::Exception("MLP")
			<< "Error opening \"" << file << "\"." << std::endl;
	epoch = (unsigned int)epoch_;
}

} // namespace PhysicsTools
