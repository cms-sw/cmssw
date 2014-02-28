#include "RecoHI/HiCentralityAlgos/plugins/CentralityTableHandler.h"
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <TFile.h>
#include <iostream>
using namespace std;
using namespace popcon;

void CentralityTableHandler::getNewObjects(){
  //  cond::TagInfo const & tagInfo_ = tagInfo();
  //  cond::LogDBEntry const & logDBEntry_ = logDBEntry();     
  Ref payload = lastPayload();
  cond::Time_t snc = 1;
  TFile* inputTFile_ = new TFile(inputTFileName_.data(),"read");
  runnum_ = 1;
  CentralityBins* CB = (CentralityBins*) inputTFile_->Get(Form("%s/run%d",centralityTag_.data(),runnum_));
  cout<<centralityTag_.data()<<endl;
  CentralityTable* CT = new CentralityTable();
  CT->m_table.reserve(CB->getNbins());

  for(int j=0; j<CB->getNbins(); j++){
    CentralityTable::CBin* thisBin = new CentralityTable::CBin();
    thisBin->bin_edge = CB->lowEdgeOfBin(j);
    thisBin->n_part.mean = CB->NpartMeanOfBin(j);
    thisBin->n_part.var  = CB->NpartSigmaOfBin(j);
    thisBin->n_coll.mean = CB->NcollMeanOfBin(j);
    thisBin->n_coll.var  = CB->NcollSigmaOfBin(j);
    thisBin->n_hard.mean = CB->NhardMeanOfBin(j);
    thisBin->n_hard.var  = CB->NhardSigmaOfBin(j);
    thisBin->b.mean = CB->bMeanOfBin(j);
    thisBin->b.var = CB->bSigmaOfBin(j);

    CT->m_table.push_back(*thisBin);
  }

  m_to_transfer.push_back(std::make_pair(CT,snc));

}

typedef  popcon::PopConAnalyzer<CentralityTableHandler> CentralityPopConProducer;
DEFINE_FWK_MODULE(CentralityPopConProducer);




