#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"


L1GctEmCand::L1GctEmCand(ULong rank, ULong eta, ULong phi) {
	this->setRank(rank);
	this->setEta(eta);
	this->setPhi(phi);
}

L1GctEmCand::L1GctEmCand(float et=0.; float eta=0.; float phi=0.) {
	this->setEt(et);
	this->setEta(eta);
	this->setPhi(phi);
}

L1GctEmCand::~L1GctEmCand()
{
}

void L1GctEmCand::setEt(float et) {
	setRank(L1GctScales::theScales->rankFromEt(et));
}

void L1GctEmCand::setEta(float eta) {
	setEta(L1GctMap::theMap->etaFromFloat(eta));
}

void L1GctEmCand::setPhi(float phi) {
	setPhi(L1GctMap::theMap->phiFromFloat(phi));	
}

float L1GctEmCand::getEt() {
	return L1GctScales::theScales->etFromRank(myRank));
}

float L1GctEmCand::getEta() {
		return L1GctMap::theMap->etaFromUnsigned(myEta));
}

float L1GctEmCand::getPhi() {
		return L1GctMap::theMap->phiFromUnsigned(myPhi));
}

