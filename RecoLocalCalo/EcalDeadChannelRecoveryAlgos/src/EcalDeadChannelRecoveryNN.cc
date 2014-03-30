#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/EcalDeadChannelRecoveryNN.h"
#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/EcalCrystalMatrixProbality.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <iostream>
#include <TMath.h>

template <typename T>
EcalDeadChannelRecoveryNN<T>::EcalDeadChannelRecoveryNN() {
  for (int id = 0; id < 9; ++id) {
    ctx_[id].mlp = NULL;
  }

  this->load();
}

template <typename T>
EcalDeadChannelRecoveryNN<T>::~EcalDeadChannelRecoveryNN() {
  for (int id = 0; id < 9; ++id) {
    if (ctx_[id].mlp) {
      // @TODO segfaults for an uknown reason
      // delete ctx[id].mlp;
      // delete ctx[id].tree;
    }
  }
}

template <>
void EcalDeadChannelRecoveryNN<EBDetId>::setCaloTopology(const CaloTopology  *topo)
{
  topology_ = topo->getSubdetectorTopology(DetId::Ecal, EcalBarrel);
}

template <>
void EcalDeadChannelRecoveryNN<EEDetId>::setCaloTopology(const CaloTopology  *topo)
{
  topology_ = topo->getSubdetectorTopology(DetId::Ecal, EcalEndcap);
}

template <typename T>
void EcalDeadChannelRecoveryNN<T>::load_file(MultiLayerPerceptronContext& ctx, std::string fn) {
  std::string path = edm::FileInPath(fn).fullPath();

  TTree *t = new TTree("t", "dummy MLP tree");
  t->SetDirectory(0);

  t->Branch("z1", &(ctx.tmp[0]), "z1/D");
  t->Branch("z2", &(ctx.tmp[1]), "z2/D");
  t->Branch("z3", &(ctx.tmp[2]), "z3/D");
  t->Branch("z4", &(ctx.tmp[3]), "z4/D");
  t->Branch("z5", &(ctx.tmp[4]), "z5/D");
  t->Branch("z6", &(ctx.tmp[5]), "z6/D");
  t->Branch("z7", &(ctx.tmp[6]), "z7/D");
  t->Branch("z8", &(ctx.tmp[7]), "z8/D");
  t->Branch("zf", &(ctx.tmp[8]), "zf/D");

  ctx.tree = t;
  ctx.mlp =
      new TMultiLayerPerceptron("@z1,@z2,@z3,@z4,@z5,@z6,@z7,@z8:10:5:zf", t);
  ctx.mlp->LoadWeights(path.c_str());
}

template <>
void EcalDeadChannelRecoveryNN<EBDetId>::load() {
  std::string p = "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/data/NNWeights/";

  this->load_file(ctx_[CellID::CC], p + "EB_ccNNWeights.txt");
  this->load_file(ctx_[CellID::RR], p + "EB_rrNNWeights.txt");
  this->load_file(ctx_[CellID::LL], p + "EB_llNNWeights.txt");
  this->load_file(ctx_[CellID::UU], p + "EB_uuNNWeights.txt");
  this->load_file(ctx_[CellID::DD], p + "EB_ddNNWeights.txt");
  this->load_file(ctx_[CellID::RU], p + "EB_ruNNWeights.txt");
  this->load_file(ctx_[CellID::RD], p + "EB_rdNNWeights.txt");
  this->load_file(ctx_[CellID::LU], p + "EB_luNNWeights.txt");
  this->load_file(ctx_[CellID::LD], p + "EB_ldNNWeights.txt");
}

template <>
void EcalDeadChannelRecoveryNN<EEDetId>::load() {
  std::string p = "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/data/NNWeights/";

  this->load_file(ctx_[CellID::CC], p + "EE_ccNNWeights.txt");
  this->load_file(ctx_[CellID::RR], p + "EE_rrNNWeights.txt");
  this->load_file(ctx_[CellID::LL], p + "EE_llNNWeights.txt");
  this->load_file(ctx_[CellID::UU], p + "EE_uuNNWeights.txt");
  this->load_file(ctx_[CellID::DD], p + "EE_ddNNWeights.txt");
  this->load_file(ctx_[CellID::RU], p + "EE_ruNNWeights.txt");
  this->load_file(ctx_[CellID::RD], p + "EE_rdNNWeights.txt");
  this->load_file(ctx_[CellID::LU], p + "EE_luNNWeights.txt");
  this->load_file(ctx_[CellID::LD], p + "EE_ldNNWeights.txt");
}

template <typename T>
double EcalDeadChannelRecoveryNN<T>::recover(const T id, const EcalRecHitCollection& hit_collection, double Sum8Cut, bool *AcceptFlag)
{
  // use the correct probability matrix
  typedef class EcalCrystalMatrixProbality<T> P;

  double NewEnergy = 0.0;

  double NewEnergy_RelMC = 0.0;
  double NewEnergy_RelDC = 0.0;

  double MNxN_RelMC[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } ;
  double MNxN_RelDC[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } ;

  double sum8 = 0.0;

  double sum8_RelMC = makeNxNMatrice_RelMC(id, hit_collection, MNxN_RelMC,AcceptFlag);
  double sum8_RelDC = makeNxNMatrice_RelDC(id, hit_collection, MNxN_RelDC,AcceptFlag);

  //  Only if "AcceptFlag" is true call the ANN
  if ( *AcceptFlag ) {
    if (sum8_RelDC > Sum8Cut && sum8_RelMC > Sum8Cut) {
      NewEnergy_RelMC = estimateEnergy(MNxN_RelMC);
      NewEnergy_RelDC = estimateEnergy(MNxN_RelDC);
      
      //  Matrices "MNxN_RelMC" and "MNxN_RelDC" have now the full set of energies, the original ones plus 
      //  whatever "estimates" by the ANN for the "dead" xtal. Use those full matrices and calculate probabilities.
      //  
      double SumMNxN_RelMC =  MNxN_RelMC[LU] + MNxN_RelMC[UU] + MNxN_RelMC[RU] + 
                              MNxN_RelMC[LL] + MNxN_RelMC[CC] + MNxN_RelMC[RR] + 
                              MNxN_RelMC[LD] + MNxN_RelMC[DD] + MNxN_RelMC[RD] ;
      
      double frMNxN_RelMC[9];  for (int i=0; i<9; i++) { frMNxN_RelMC[i] = MNxN_RelMC[i] / SumMNxN_RelMC ; }
      
      double prMNxN_RelMC  =  P::Diagonal(  frMNxN_RelMC[LU] ) * P::UpDown(  frMNxN_RelMC[UU] ) * P::Diagonal(  frMNxN_RelMC[RU] ) * 
                              P::ReftRight( frMNxN_RelMC[LL] ) * P::Central( frMNxN_RelMC[CC] ) * P::ReftRight( frMNxN_RelMC[RR] ) * 
                              P::Diagonal(  frMNxN_RelMC[LD] ) * P::UpDown(  frMNxN_RelMC[DD] ) * P::Diagonal(  frMNxN_RelMC[RD] ) ;
      
      double SumMNxN_RelDC =  MNxN_RelDC[LU] + MNxN_RelDC[UU] + MNxN_RelDC[RU] + 
                              MNxN_RelDC[LL] + MNxN_RelDC[CC] + MNxN_RelDC[RR] + 
                              MNxN_RelDC[LD] + MNxN_RelDC[DD] + MNxN_RelDC[RD] ;
      
      double frMNxN_RelDC[9];  for (int i=0; i<9; i++) { frMNxN_RelDC[i] = MNxN_RelDC[i] / SumMNxN_RelDC ; }
      
      double prMNxN_RelDC  =  P::Diagonal(  frMNxN_RelDC[LU] ) * P::UpDown(  frMNxN_RelDC[UU] ) * P::Diagonal(  frMNxN_RelDC[RU] ) * 
                              P::ReftRight( frMNxN_RelDC[LL] ) * P::Central( frMNxN_RelDC[CC] ) * P::ReftRight( frMNxN_RelDC[RR] ) * 
                              P::Diagonal(  frMNxN_RelDC[LD] ) * P::UpDown(  frMNxN_RelDC[DD] ) * P::Diagonal(  frMNxN_RelDC[RD] ) ;
      
      if ( prMNxN_RelDC > prMNxN_RelMC )  { NewEnergy = NewEnergy_RelDC ; sum8 = sum8_RelDC ; } 
      if ( prMNxN_RelDC <= prMNxN_RelMC ) { NewEnergy = NewEnergy_RelMC ; sum8 = sum8_RelMC ; } 
      
      
      //  If the return value of "CorrectDeadChannelsNN" is one of the followin negative values then
      //  it corresponds to an error condition. See "CorrectDeadChannelsNN.cc" for possible values.
      if ( NewEnergy == -1000000.0 ||
           NewEnergy == -1000001.0 ||
           NewEnergy == -2000000.0 ||
           NewEnergy == -3000000.0 ||
           NewEnergy == -3000001.0 ) { *AcceptFlag=false ; NewEnergy = 0.0 ; }             
    }
  }
  
  // Protect against non physical high values
  // From the distribution of (max.cont.xtal / Sum8) we get as limit 5 (hard) and 10 (softer)
  // Choose 10 as highest possible energy to be assigned to the dead channel under any scenario.
  if ( NewEnergy > 10.0 * sum8 ) { *AcceptFlag=false ; NewEnergy = 0.0 ; }
  
  return NewEnergy;
}

template <typename T>
double EcalDeadChannelRecoveryNN<T>::estimateEnergy(double *M3x3Input, double epsilon) {
  int missing[9];
  int missing_index = 0;

  for (int i = 0; i < 9; i++) {
    if (TMath::Abs(M3x3Input[i]) < epsilon) {
      missing[missing_index++] = i;
    } else {
      //  Generally the "dead" cells are allowed to have negative energies (since they will be estimated by the ANN anyway).
      //  But all the remaining "live" ones must have positive values otherwise the logarithm fails.

      if (M3x3Input[i] < 0.0) { return -2000000.0; }
    }
  }

  //  Currently EXACTLY ONE AND ONLY ONE dead cell is corrected. Return -1000000.0 if zero DC's detected and -101.0 if more than one DC's exist.
  int idxDC = -1 ;
  if (missing_index == 0) { return -1000000.0; }    //  Zero DC's were detected
  if (missing_index  > 1) { return -1000001.0; }    //  More than one DC's detected.
  if (missing_index == 1) { idxDC = missing[0]; } 

  // Arrange inputs into an array of 8, excluding the dead cell;
  int input_order[9] = { CC, RR, LL, UU, DD, RU, RD, LU, LD };
  int input_index = 0;
  Double_t input[8];

  for (int id : input_order) {
    if (id == idxDC)
      continue;

    input[input_index++] = TMath::Log(M3x3Input[id]);
  }

  //  Select the case to apply the appropriate NN and return the result.
  M3x3Input[idxDC] = TMath::Exp(ctx_[idxDC].mlp->Evaluate(0, input));
  return M3x3Input[idxDC];
}

template <typename DetIdT>
double EcalDeadChannelRecoveryNN<DetIdT>::makeNxNMatrice_RelMC(DetIdT itID, const EcalRecHitCollection& hit_collection, double *MNxN_RelMC, bool* AcceptFlag) {
  //  Since ANN corrects within a 3x3 window, the possible candidate 3x3 windows that contain 
  //  the "dead" crystal form a 5x5 window around it (totaly eight 3x3 windows overlapping).
  //  Get this 5x5 and locate the Max.Contain.Crystal within.

  //  Get the 5x5 window around the "dead" crystal -> vector "NxNaroundDC"
  std::vector<DetId> NxNaroundDC = topology_->getWindow(itID, 5, 5);

  DetIdT CellMax ;     //  Create a null DetId
  double EnergyMax = 0.0;

  //  Loop over all cells in the vector "NxNaroundDC", and for each cell find it's energy
  //  (from the EcalRecHits collection). Use this energy to detect the Max.Cont.Crystal.
  std::vector<DetId>::const_iterator theCells;

  for (theCells = NxNaroundDC.begin(); theCells != NxNaroundDC.end(); ++theCells) {
    DetIdT cell = DetIdT(*theCells);

    if (! cell.null()) {
      EcalRecHitCollection::const_iterator goS_it = hit_collection.find(cell);
      
      if ( goS_it !=  hit_collection.end() && goS_it->energy() >= EnergyMax ) {
        EnergyMax = goS_it->energy();
        CellMax = cell;
      }
    }
  }

  //  No Max.Cont.Crystal found, return back with no changes.
  if ( CellMax.null() ) { *AcceptFlag=false ; return 0.0 ; }

#if 1
  // Not a smart check, because why not just get 4x4 matrix and have a guaranteed hit?

  //  Check that the "dead" crystal belongs to the 3x3 around  Max.Cont.Crystal
  bool dcIn3x3 = false ;
  
  std::vector<DetId> NxNaroundMaxCont = topology_->getWindow(CellMax,3,3);
  std::vector<DetId>::const_iterator testCell;
  for (testCell = NxNaroundMaxCont.begin(); testCell != NxNaroundMaxCont.end(); ++testCell) {
    if ( itID == DetIdT(*testCell) ) { dcIn3x3 = true ; } 
  }
  
  //  If the "dead" crystal is outside the 3x3 then do nothing.
  if (!dcIn3x3) { *AcceptFlag=false ; return 0.0 ; }
#endif
  
  return makeNxNMatrice_RelDC(CellMax, hit_collection, MNxN_RelMC, AcceptFlag);
}

template <>
double EcalDeadChannelRecoveryNN<EBDetId>::makeNxNMatrice_RelDC(EBDetId id, const EcalRecHitCollection& hit_collection, double *MNxN, bool* AcceptFlag) {
  //  Make an ANN 3x3 energy matrix around the crystal.
  //  If the "dead" crystal is at the EB boundary (+/- 85) do nothing since we need a full 3x3 around it.
  if ( id.ieta() == 85 || id.ieta() == -85 ) { *AcceptFlag=false ; return 0.0 ; }

  //  Define the ieta and iphi steps (zero, plus, minus)
  int ietaZ = id.ieta() ;
  int ietaP = ( ietaZ == -1 ) ?  1 : ietaZ + 1 ;
  int ietaN = ( ietaZ ==  1 ) ? -1 : ietaZ - 1 ;

  int iphiZ = id.iphi() ;
  int iphiP = ( iphiZ == 360 ) ?   1 : iphiZ + 1 ;
  int iphiN = ( iphiZ ==   1 ) ? 360 : iphiZ - 1 ;
  
  for (int i=0; i<9; i++) { MNxN[i] = 0.0 ; }
  
  //  Loop over all cells in the vector "window", and fill the MNxN matrix
  //  to be passed to the ANN for prediction.
  std::vector<DetId> window = topology_->getWindow(id, 3, 3);

  std::vector<DetId>::const_iterator itCells;
  for (itCells = window.begin(); itCells != window.end(); ++itCells) {
    EBDetId cell = EBDetId(*itCells);

    if (! cell.null()) {
      EcalRecHitCollection::const_iterator goS_it = hit_collection.find(cell);

      if (goS_it !=  hit_collection.end()) {
        double energy = goS_it->energy();

        if       ( cell.ieta() == ietaP && cell.iphi() == iphiP ) { MNxN[RU] = energy; }
        else if  ( cell.ieta() == ietaP && cell.iphi() == iphiZ ) { MNxN[RR] = energy; }
        else if  ( cell.ieta() == ietaP && cell.iphi() == iphiN ) { MNxN[RD] = energy; }
        
        else if  ( cell.ieta() == ietaZ && cell.iphi() == iphiP ) { MNxN[UU] = energy; }
        else if  ( cell.ieta() == ietaZ && cell.iphi() == iphiZ ) { MNxN[CC] = energy; }
        else if  ( cell.ieta() == ietaZ && cell.iphi() == iphiN ) { MNxN[DD] = energy; }
        
        else if  ( cell.ieta() == ietaN && cell.iphi() == iphiP ) { MNxN[LU] = energy; }
        else if  ( cell.ieta() == ietaN && cell.iphi() == iphiZ ) { MNxN[LL] = energy; }
        else if  ( cell.ieta() == ietaN && cell.iphi() == iphiN ) { MNxN[LD] = energy; }

        else { *AcceptFlag=false ; return 0.0 ;}
      }
    }
  }

  //  Get the sum of 8
  double ESUMis = 0.0 ; 
  for (int i=0; i<9; i++) { ESUMis = ESUMis + MNxN[i] ; }

  *AcceptFlag=true ;
  return ESUMis;
}

template <>
double EcalDeadChannelRecoveryNN<EEDetId>::makeNxNMatrice_RelDC(EEDetId itID,const EcalRecHitCollection& hit_collection, double *MNxN, bool* AcceptFlag) {
  //  Make an ANN 3x3 energy matrix around the crystal.
  //  If the "dead" crystal is at the EB boundary (+/- 85) do nothing since we need a full 3x3 around it.

  //  If the "dead" crystal is at the EE boundary (inner or outer ring) do nothing since we need a full 3x3 around it.
  if ( EEDetId::isNextToRingBoundary(itID) ) { *AcceptFlag=false ; return 0.0 ; }

  //  Define the ix and iy steps (zero, plus, minus)
  int ixZ = itID.ix() ;
  int ixP = ixZ + 1 ;
  int ixN = ixZ - 1 ;

  int iyZ = itID.iy() ;
  int iyP = iyZ + 1 ;
  int iyN = iyZ - 1 ;
  
  for (int i=0; i<9; i++) { MNxN[i] = 0.0 ; }

  //  Take the "dead" crystal as reference and get the 3x3 around it.
  std::vector<DetId> NxNaroundRefXtal = topology_->getWindow(itID,3,3);

  //  Loop over all cells in the vector "NxNaroundRefXtal", and fill the MNxN matrix
  //  to be passed to the ANN for prediction.
  std::vector<DetId>::const_iterator itCells;

  for (itCells = NxNaroundRefXtal.begin(); itCells != NxNaroundRefXtal.end(); ++itCells) {
    EEDetId cell = EEDetId(*itCells);

    if (!cell.null()) {
      EcalRecHitCollection::const_iterator goS_it = hit_collection.find(cell);
      
      if ( goS_it !=  hit_collection.end() ) { 
        double energy = goS_it->energy();

        if       ( cell.ix() == ixP && cell.iy() == iyP ) { MNxN[RU] = energy; }
        else if  ( cell.ix() == ixP && cell.iy() == iyZ ) { MNxN[RR] = energy; }
        else if  ( cell.ix() == ixP && cell.iy() == iyN ) { MNxN[RD] = energy; }
        
        else if  ( cell.ix() == ixZ && cell.iy() == iyP ) { MNxN[UU] = energy; }
        else if  ( cell.ix() == ixZ && cell.iy() == iyZ ) { MNxN[CC] = energy; }
        else if  ( cell.ix() == ixZ && cell.iy() == iyN ) { MNxN[DD] = energy; }
        
        else if  ( cell.ix() == ixN && cell.iy() == iyP ) { MNxN[LU] = energy; }
        else if  ( cell.ix() == ixN && cell.iy() == iyZ ) { MNxN[LL] = energy; }
        else if  ( cell.ix() == ixN && cell.iy() == iyN ) { MNxN[LD] = energy; }

        else { *AcceptFlag=false ; return 0.0 ;}
      }
    }
  }

  //  Get the sum of 8
  double ESUMis = 0.0 ; 
  for (int i=0; i<9; i++) { ESUMis = ESUMis + MNxN[i] ; }

  *AcceptFlag=true ;
  return ESUMis;
}


template class EcalDeadChannelRecoveryNN<EBDetId>;
template class EcalDeadChannelRecoveryNN<EEDetId>;
