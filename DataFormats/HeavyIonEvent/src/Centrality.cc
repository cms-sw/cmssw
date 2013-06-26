//
// $Id: Centrality.cc,v 1.14 2010/10/16 17:09:34 yilmaz Exp $
//

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include <iostream>
using namespace std;
using namespace reco;

Centrality::Centrality(double d, std::string label)
  : 
value_(d),
label_(label),
etHFhitSumPlus_(0),
etHFtowerSumPlus_(0),
etHFtruncatedPlus_(0),
etHFhitSumMinus_(0),
etHFtowerSumMinus_(0),
etHFtruncatedMinus_(0),
etEESumPlus_(0),
etEEtruncatedPlus_(0),
etEESumMinus_(0),
etEEtruncatedMinus_(0),
etEBSum_(0),
etEBtruncated_(0),
pixelMultiplicity_(0),
trackMultiplicity_(0),
zdcSumPlus_(0),
zdcSumMinus_(0),
etMidRapiditySum_(0),
ntracksPtCut_(0),
ntracksEtaCut_(0),
ntracksEtaPtCut_(0),
nPixelTracks_(0)
{
}


Centrality::~Centrality()
{
}

#include "CondFormats/HIObjects/interface/CentralityTable.h"
#include "CondFormats/DataRecord/interface/HeavyIonRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

const CentralityBins* getCentralityBinsFromDB(const edm::EventSetup& iSetup){

  string centralityLabel = "";
  string centralityMC = "";
  
  const edm::ParameterSet &thepset = edm::getProcessParameterSet();
  if(thepset.exists("HeavyIonGlobalParameters")){
    edm::ParameterSet hiPset = thepset.getParameter<edm::ParameterSet>("HeavyIonGlobalParameters");
    centralityLabel = hiPset.getParameter<string>("centralityVariable");
    if(hiPset.exists("nonDefaultGlauberModel")){
      centralityMC = hiPset.getParameter<string>("nonDefaultGlauberModel");
      centralityLabel += centralityMC;
    }
  }
  
  edm::ESHandle<CentralityTable> inputDB_;
  iSetup.get<HeavyIonRcd>().get(centralityLabel,inputDB_);
  int nbinsMax = inputDB_->m_table.size();
  //cout<<"nbinsMax "<<nbinsMax<<endl;
   CentralityBins* CB = new CentralityBins("ctemp","",nbinsMax);
   for(int j=0; j<nbinsMax; j++){

      const CentralityTable::CBin* thisBin;
      thisBin = &(inputDB_->m_table[j]);
      CB->table_[j].bin_edge = thisBin->bin_edge;
      CB->table_[j].n_part_mean = thisBin->n_part.mean;
      CB->table_[j].n_part_var  = thisBin->n_part.var;
      CB->table_[j].n_coll_mean = thisBin->n_coll.mean;
      CB->table_[j].n_coll_var  = thisBin->n_coll.var;
      CB->table_[j].n_hard_mean = thisBin->n_hard.mean;
      CB->table_[j].n_hard_var  = thisBin->n_hard.var;
      CB->table_[j].b_mean = thisBin->b.mean;
      CB->table_[j].b_var = thisBin->b.var;

   }

   return CB;
}



