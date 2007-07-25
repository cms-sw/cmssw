#include "AnalysisDataFormats/TopObjects/interface/JetRejObs.h"

JetRejObs::JetRejObs(){ }
JetRejObs::~JetRejObs(){ }

void 		JetRejObs::setJetRejObs( vector<pair< int, double> > ob)     { obs = ob; }


unsigned int	        JetRejObs::getSize() const	                    { return obs.size(); }
pair<int,double>	JetRejObs::getPair(unsigned int i) const	    { return obs[i]; }
