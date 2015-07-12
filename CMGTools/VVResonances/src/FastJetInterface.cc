#include "CMGTools/VVResonances/interface/FastJetInterface.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "fastjet/tools/Pruner.hh"
#include "fastjet/tools/MassDropTagger.hh"
#include "fastjet/contrib/Njettiness.hh"
#include "fastjet/contrib/SoftDrop.hh"


using namespace std;
//using namespace std;
using namespace fastjet;

namespace cmg{


FastJetInterface::FastJetInterface(const std::vector<LorentzVector> & objects, double ktpower, double rparam)
{
  // define jet inputs
  input_.clear();
  int index=0;
  for (const LorentzVector &o : objects) {
    fastjet::PseudoJet j(o.Px(),o.Py(),o.Pz(),o.E());
    j.set_user_index(index); index++; // in case we want to know which piece ended where
    input_.push_back(j);
  }
  // choose a jet definition
  fastjet::JetDefinition jet_def;

  // prepare jet def 
  if (ktpower == 1.0) {
    jet_def = JetDefinition(kt_algorithm, rparam);
  }  else if (ktpower == 0.0) {
    jet_def = JetDefinition(cambridge_algorithm, rparam);
  }  else if (ktpower == -1.0) {
    jet_def = JetDefinition(antikt_algorithm, rparam);
  }  else {
    throw cms::Exception("InvalidArgument", "Unsupported ktpower value");
  }
  clusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequence( input_, jet_def)); 
}


FastJetInterface::FastJetInterface(const std::vector<LorentzVector> & objects, double ktpower, double rparam,int active_area_repeats,double ghost_area,double ghost_eta_max,double rho_eta_max)
{
  // define jet inputs
  input_.clear();
  int index=0;
  for (const LorentzVector &o : objects) {
    fastjet::PseudoJet j(o.Px(),o.Py(),o.Pz(),o.E());
    j.set_user_index(index); index++; // in case we want to know which piece ended where
    input_.push_back(j);
  }

  // choose a jet definition
  fastjet::JetDefinition jet_def;
  fastjet::GhostedAreaSpec ghosted_spec(ghost_eta_max,active_area_repeats,ghost_area);
  ghosted_spec.set_fj2_placement(true);
  fastjet::AreaDefinition area_def(fastjet::active_area_explicit_ghosts,ghosted_spec);

  // prepare jet def 
  if (ktpower == 1.0) {
    jet_def = JetDefinition(kt_algorithm, rparam);
  }  else if (ktpower == 0.0) {
    jet_def = JetDefinition(cambridge_algorithm, rparam);
  }  else if (ktpower == -1.0) {
    jet_def = JetDefinition(antikt_algorithm, rparam);
  }  else {
    throw cms::Exception("InvalidArgument", "Unsupported ktpower value");
  }
  clusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequenceArea( input_, jet_def,area_def)); 
}



std::vector<math::XYZTLorentzVector> FastJetInterface::makeP4s(const std::vector<fastjet::PseudoJet> &jets) {
  std::vector<math::XYZTLorentzVector> JetObjectsAll;
  for (const fastjet::PseudoJet & pj : jets) {
    JetObjectsAll.push_back( LorentzVector( pj.px(), pj.py(), pj.pz(), pj.e() ) );
  }
  return JetObjectsAll;
}


void  FastJetInterface::makeInclusiveJets( double ptMin ) {
  jets_ = sorted_by_pt(clusterSeq_->inclusive_jets(ptMin));
}

void  FastJetInterface::makeExclusiveJets( double dcut ) {
  jets_ = sorted_by_pt(clusterSeq_->exclusive_jets(dcut));
}

void  FastJetInterface::makeExclusiveJets(  int njets ) {
  jets_ = sorted_by_pt(clusterSeq_->exclusive_jets(njets));

}

void  FastJetInterface::makeExclusiveJetsUpTo( int njets ) {
  jets_ = sorted_by_pt(clusterSeq_->exclusive_jets_up_to(njets));
}


void FastJetInterface::makeSubJets( unsigned int i, double dcut) {

  std::vector<fastjet::PseudoJet> empty;
  if (i>jets_.size()-1) {
    printf("Make Subjets(dcut)Collection size smaller than the requested jet\n");
    subjets_ = empty;
    return;
  }
  
  subjets_ = sorted_by_pt(jets_[i].exclusive_subjets(dcut));
}


void FastJetInterface::makeSubJets( unsigned int i) {

  std::vector<fastjet::PseudoJet> empty;
  if (i>jets_.size()-1) {
    printf("MakeSubjets(pieces)Collection size smaller than the requested jet\n");
    subjets_ = empty;
    return;
  }
  if(jets_[i].has_pieces())
    subjets_ = sorted_by_pt(jets_[i].pieces());
  else
    subjets_=empty;
}


void FastJetInterface::makeSubJets( unsigned int i, int N) {
  std::vector<fastjet::PseudoJet> empty;
  if (i>jets_.size()-1) {
    printf("MakeSubJets(N): Collection size smaller than the requested jet\n");
    return;
  }

  if (jets_[i].constituents().size()<unsigned(N))
    subjets_ = empty;
  else
    subjets_ = sorted_by_pt(jets_[i].exclusive_subjets(N));
}



void FastJetInterface::makeSubJetsUpTo( unsigned int i, int njets) {

  if (i>jets_.size()-1) {
    printf("MakeSubJetsUpTo(N) :Collection size smaller than the requested jet\n");
    return;
  }
  subjets_ = sorted_by_pt(jets_[i].exclusive_subjets_up_to(njets));
}



std::vector<math::XYZTLorentzVector> FastJetInterface::get(bool subjets) {
 const std::vector<fastjet::PseudoJet>& output = subjets ? subjets_ : jets_;
 return makeP4s(output);
}


std::vector< unsigned int> FastJetInterface::getConstituents(bool jet, unsigned int i) {
  std::vector<unsigned int> out;

  const std::vector<fastjet::PseudoJet>& output = jet ? jets_ : subjets_;

  if (i>output.size()-1) {
    printf("Constituents: Collection size smaller than the requested jet\n");
    return out;
  }
  
  for (const auto& c : output[i].constituents())
    if (c.pt()>1e-3)
      out.push_back(c.user_index());
  return out;
}

double FastJetInterface::getArea(bool jet, unsigned int i) {

  const std::vector<fastjet::PseudoJet>& output = jet ? jets_ : subjets_;

  if (i>output.size()-1) {
    printf("Area: Collection size smaller than the requested jet\n");
    return 0.0;
  }
  if(output[i].has_area())
    return output[i].area();
  else
    return -1.0;
}




bool FastJetInterface::massDropTag( unsigned int i,double& mu, double& y) {
  if (i>jets_.size()-1) {
    printf("Mass Drop:Collection size smaller than the requested jet\n");
    return false;
  }
  
  MassDropTagger mdtagger(mu,y);
  fastjet::PseudoJet tagged_jet = mdtagger(jets_[i]);

  if( tagged_jet==0)
    return false;

  mu = tagged_jet.structure_of<MassDropTagger>().mu();
  y = tagged_jet.structure_of<MassDropTagger>().y();
  return true;
}



void FastJetInterface::prune(bool jet, double zcut,double rcutfactor ) {

  const std::vector<fastjet::PseudoJet>& input = jet ? jets_ : subjets_;
  fastjet::Pruner pruner(fastjet::cambridge_algorithm, zcut, rcutfactor);

  std::vector<fastjet::PseudoJet> output;
  for (unsigned int i=0;i<input.size();++i) {
    output.push_back(pruner(input[i]));
  }

  if(jet)
    jets_ = output;
  else
    subjets_ = output;
}

void FastJetInterface::softDrop(bool jet, double beta, double zcut,double R0 ) {

  const std::vector<fastjet::PseudoJet>& input = jet ? jets_ : subjets_;
  fastjet::contrib::SoftDrop softdrop(beta, zcut, R0);

  std::vector<fastjet::PseudoJet> output;
  for (unsigned int i=0;i<input.size();++i) {
    output.push_back(softdrop(input[i]));
  }

  if(jet)
    jets_ = output;
  else
    subjets_ = output;
}





std::vector<double> FastJetInterface::nSubJettiness(unsigned int i ,int NMAX,unsigned int measureDefinition,unsigned int axesDefinition, double beta, double R0,double Rcutoff,double akAxesR0,int nPass) {

  enum MeasureDefinition_t {
    NormalizedMeasure=0,       // (beta,R0) 
    UnnormalizedMeasure,       // (beta) 
    GeometricMeasure,          // (beta) 
    NormalizedCutoffMeasure,   // (beta,R0,Rcutoff) 
    UnnormalizedCutoffMeasure, // (beta,Rcutoff) 
    GeometricCutoffMeasure,    // (beta,Rcutoff) 
    N_MEASURE_DEFINITIONS
  };

  enum AxesDefinition_t {
    KT_Axes=0,
    CA_Axes,
    AntiKT_Axes,   // (axAxesR0)
    WTA_KT_Axes,
    WTA_CA_Axes,
    Manual_Axes,
    OnePass_KT_Axes,
    OnePass_CA_Axes,
    OnePass_AntiKT_Axes,   // (axAxesR0)
    OnePass_WTA_KT_Axes,
    OnePass_WTA_CA_Axes,
    OnePass_Manual_Axes,
    MultiPass_Axes,
      N_AXES_DEFINITIONS
  };



  
  fastjet::contrib::NormalizedMeasure          normalizedMeasure        (beta,R0);
  fastjet::contrib::UnnormalizedMeasure        unnormalizedMeasure      (beta);
  fastjet::contrib::GeometricMeasure           geometricMeasure         (beta);
  fastjet::contrib::NormalizedCutoffMeasure    normalizedCutoffMeasure  (beta,R0,Rcutoff);
  fastjet::contrib::UnnormalizedCutoffMeasure  unnormalizedCutoffMeasure(beta,Rcutoff);
  fastjet::contrib::GeometricCutoffMeasure     geometricCutoffMeasure   (beta,Rcutoff);

  fastjet::contrib::MeasureDefinition const * measureDef = 0;
  switch ( measureDefinition ) {
  case UnnormalizedMeasure : measureDef = &unnormalizedMeasure; break;
  case GeometricMeasure    : measureDef = &geometricMeasure; break;
  case NormalizedCutoffMeasure : measureDef = &normalizedCutoffMeasure; break;
  case UnnormalizedCutoffMeasure : measureDef = &unnormalizedCutoffMeasure; break;
  case GeometricCutoffMeasure : measureDef = &geometricCutoffMeasure; break;
  case NormalizedMeasure : default : measureDef = &normalizedMeasure; break;
  } 

  // Get the axes definition
  fastjet::contrib::KT_Axes             kt_axes; 
  fastjet::contrib::CA_Axes             ca_axes; 
  fastjet::contrib::AntiKT_Axes         antikt_axes   (akAxesR0);
  fastjet::contrib::WTA_KT_Axes         wta_kt_axes; 
  fastjet::contrib::WTA_CA_Axes         wta_ca_axes; 
  fastjet::contrib::OnePass_KT_Axes     onepass_kt_axes;
  fastjet::contrib::OnePass_CA_Axes     onepass_ca_axes;
  fastjet::contrib::OnePass_AntiKT_Axes onepass_antikt_axes   (akAxesR0);
  fastjet::contrib::OnePass_WTA_KT_Axes onepass_wta_kt_axes;
  fastjet::contrib::OnePass_WTA_CA_Axes onepass_wta_ca_axes;
  fastjet::contrib::MultiPass_Axes      multipass_axes (nPass);

  fastjet::contrib::AxesDefinition const * axesDef = 0;
  switch ( axesDefinition ) {
  case  KT_Axes : default : axesDef = &kt_axes; break;
  case  CA_Axes : axesDef = &ca_axes; break; 
  case  AntiKT_Axes : axesDef = &antikt_axes; break;
  case  WTA_KT_Axes : axesDef = &wta_kt_axes; break; 
  case  WTA_CA_Axes : axesDef = &wta_ca_axes; break; 
  case  OnePass_KT_Axes : axesDef = &onepass_kt_axes; break;
  case  OnePass_CA_Axes : axesDef = &onepass_ca_axes; break; 
  case  OnePass_AntiKT_Axes : axesDef = &onepass_antikt_axes; break;
  case  OnePass_WTA_KT_Axes : axesDef = &onepass_wta_kt_axes; break; 
  case  OnePass_WTA_CA_Axes : axesDef = &onepass_wta_ca_axes; break; 
  case  MultiPass_Axes : axesDef = &multipass_axes; break;
  };


  std::vector<double> out;
  fastjet::contrib::Njettiness jettiness(*axesDef,*measureDef);
  for ( int N=1;N<=NMAX;++N) {
    out.push_back(jettiness.getTau(N,jets_[i].constituents()));
  }
  return out;
}


}

