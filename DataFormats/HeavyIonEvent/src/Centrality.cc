//
// $Id: Centrality.cc,v 1.4 2010/02/23 13:35:38 yilmaz Exp $
//

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"

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
zdcSumPlus_(0),
zdcSumMinus_(0)
{
}


Centrality::~Centrality()
{
}

#include "CondFormats/HIObjects/interface/CentralityTable.h"
#include "CondFormats/DataRecord/interface/HeavyIonRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

const CentralityBins* getCentralityBinsFromDB(const edm::EventSetup& iSetup, int nbins){

   CentralityBins* CB = new CentralityBins("ctemp","",nbins);
   edm::ESHandle<CentralityTable> inputDB_;
   iSetup.get<HeavyIonRcd>().get(inputDB_);

   for(int j=0; j<nbins; j++){

      const CentralityTable::CBin* thisBin;
      thisBin = &(inputDB_->m_table[j]);
      CB->table_[j].bin_edge = thisBin->bin_edge;
      CB->table_[j].n_part_mean = thisBin->n_part_mean;
      CB->table_[j].n_part_var  = thisBin->n_part_var;
      CB->table_[j].n_coll_mean = thisBin->n_coll_mean;
      CB->table_[j].n_coll_var  = thisBin->n_coll_var;
      CB->table_[j].n_hard_mean = thisBin->n_hard_mean;
      CB->table_[j].n_hard_var  = thisBin->n_hard_var;
      CB->table_[j].b_mean = thisBin->b_mean;
      CB->table_[j].b_var = thisBin->b_var;

   }
   return CB;
}



