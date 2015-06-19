#include "RecoBTag/SecondaryVertex/interface/MvaBoostedDoubleSecondaryVertexEstimator.h"

#include "CommonTools/Utils/interface/TMVAZipReader.h"


MvaBoostedDoubleSecondaryVertexEstimator::MvaBoostedDoubleSecondaryVertexEstimator(const std::string & weightFile)
{
  TMVAReader = new TMVA::Reader("Color:Silent:Error");
  TMVAReader->SetVerbose(false);

  TMVAReader->AddVariable("PFLepton_ptrel",       &mva_PFLepton_ptrel);
  TMVAReader->AddVariable("z_ratio1",             &mva_z_ratio);
  TMVAReader->AddVariable("tau_dot",              &mva_tau_dot);
  TMVAReader->AddVariable("SV_mass_0",            &mva_SV_mass_0);
  TMVAReader->AddVariable("SV_vtx_EnergyRatio_0", &mva_SV_EnergyRatio_0);
  TMVAReader->AddVariable("SV_vtx_EnergyRatio_1", &mva_SV_EnergyRatio_1);
  TMVAReader->AddVariable("PFLepton_IP2D",        &mva_PFLepton_IP2D);
  TMVAReader->AddVariable("tau2/tau1",            &mva_tau21);
  TMVAReader->AddVariable("nSL",                  &mva_nSL);
  TMVAReader->AddVariable("jetNTracksEtaRel",     &mva_jetNTracksEtaRel);

  TMVAReader->AddSpectator("massGroomed", &mva_massGroomed);
  TMVAReader->AddSpectator("flavour",     &mva_flavour);
  TMVAReader->AddSpectator("nbHadrons",   &mva_nbHadrons);
  TMVAReader->AddSpectator("ptGroomed",   &mva_ptGroomed);
  TMVAReader->AddSpectator("etaGroomed",  &mva_etaGroomed);

  reco::details::loadTMVAWeights(TMVAReader, "BDTG", weightFile.c_str());
}


MvaBoostedDoubleSecondaryVertexEstimator::~MvaBoostedDoubleSecondaryVertexEstimator()
{
  delete TMVAReader;
}


float MvaBoostedDoubleSecondaryVertexEstimator::mvaValue(float PFLepton_ptrel, float z_ratio, float tau_dot, float SV_mass_0,
                                                         float SV_EnergyRatio_0, float SV_EnergyRatio_1, float PFLepton_IP2D,
                                                         float tau21, float nSL, float jetNTracksEtaRel)
{
  mva_PFLepton_ptrel   = PFLepton_ptrel;
  mva_z_ratio          = z_ratio;
  mva_tau_dot          = tau_dot;
  mva_SV_mass_0        = SV_mass_0;
  mva_SV_EnergyRatio_0 = SV_EnergyRatio_0;
  mva_SV_EnergyRatio_1 = SV_EnergyRatio_1;
  mva_PFLepton_IP2D    = PFLepton_IP2D;
  mva_tau21            = tau21;
  mva_nSL              = nSL;
  mva_jetNTracksEtaRel = jetNTracksEtaRel;

  // evaluate the MVA
  float value = TMVAReader->EvaluateMVA("BDTG");

  return value;
}
