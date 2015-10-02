#include "RecoBTag/SecondaryVertex/interface/CandidateBoostedDoubleSecondaryVertexComputer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSoftLeptonTagInfo.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"


CandidateBoostedDoubleSecondaryVertexComputer::CandidateBoostedDoubleSecondaryVertexComputer(const edm::ParameterSet & parameters) :
  beta_(parameters.getParameter<double>("beta")),
  R0_(parameters.getParameter<double>("R0")),
  njettiness_(fastjet::contrib::OnePass_KT_Axes(), fastjet::contrib::NormalizedMeasure(beta_,R0_)),
  maxSVDeltaRToJet_(parameters.getParameter<double>("maxSVDeltaRToJet")),
  weightFile_(parameters.getParameter<edm::FileInPath>("weightFile"))
{
  uses(0, "ipTagInfos");
  uses(1, "svTagInfos");
  uses(2, "muonTagInfos");
  uses(3, "elecTagInfos");

  mvaID.reset(new TMVAEvaluator());

  // variable order needs to be the same as in the training
  std::vector<std::string> variables({"PFLepton_ptrel", "z_ratio1", "tau_dot", "SV_mass_0", "SV_vtx_EnergyRatio_0",
                                      "SV_vtx_EnergyRatio_1","PFLepton_IP2D", "tau2/tau1", "nSL", "jetNTracksEtaRel"});
  std::vector<std::string> spectators({"massGroomed", "flavour", "nbHadrons", "ptGroomed", "etaGroomed"});

  mvaID->initialize("Color:Silent:Error", "BDTG", weightFile_.fullPath(), variables, spectators,true,false);
}


float CandidateBoostedDoubleSecondaryVertexComputer::discriminator(const TagInfoHelper & tagInfo) const
{
  // get TagInfos
  const reco::CandIPTagInfo              & ipTagInfo = tagInfo.get<reco::CandIPTagInfo>(0);
  const reco::CandSecondaryVertexTagInfo & svTagInfo = tagInfo.get<reco::CandSecondaryVertexTagInfo>(1);
  const reco::CandSoftLeptonTagInfo      & muonTagInfo = tagInfo.get<reco::CandSoftLeptonTagInfo>(2);
  const reco::CandSoftLeptonTagInfo      & elecTagInfo = tagInfo.get<reco::CandSoftLeptonTagInfo>(3);

  // default discriminator value
  float value = -10.;

  // default variable values
  float z_ratio = -1. , tau_dot = -1., SV_pt_0 = -1., SV_mass_0 = -1., SV_EnergyRatio_0 = -1., SV_EnergyRatio_1 = -1., tau21 = -1.;
  int contSV = 0, vertexNTracks = 0;
  int nSL = 0, nSM = 0, nSE = 0;

  // get the jet reference
  const reco::JetBaseRef jet = svTagInfo.jet();

  std::vector<fastjet::PseudoJet> currentAxes;
  float tau2, tau1;
  // calculate N-subjettiness
  calcNsubjettiness(jet, tau1, tau2, currentAxes);
  if (tau1 != 0.) tau21 = tau2/tau1;

  const std::vector<reco::CandidatePtr> & selectedTracks( ipTagInfo.selectedTracks() );
  size_t trackSize = selectedTracks.size();
  const reco::VertexRef & vertexRef = ipTagInfo.primaryVertex();
  reco::TrackKinematics allKinematics;

  for (size_t itt=0; itt < trackSize; ++itt)
  {
    const reco::Track & ptrack = *(reco::btag::toTrack(selectedTracks[itt]));
    const reco::CandidatePtr ptrackRef = selectedTracks[itt];

    float track_PVweight = 0.;
    setTracksPV(ptrackRef, vertexRef, track_PVweight);
    if (track_PVweight>0.) { allKinematics.add(ptrack, track_PVweight); }
  }

  math::XYZVector jetDir = jet->momentum().Unit();

  std::map<double, size_t> VTXmass;
  for (size_t vtx = 0; vtx < svTagInfo.nVertices(); ++vtx)
  {
    vertexNTracks += (svTagInfo.secondaryVertex(vtx)).numberOfSourceCandidatePtrs();
    GlobalVector flightDir = svTagInfo.flightDirection(vtx);
    if (reco::deltaR2(flightDir, jetDir)<(maxSVDeltaRToJet_*maxSVDeltaRToJet_))
    {
      ++contSV;
      VTXmass[svTagInfo.secondaryVertex(vtx).p4().mass()]=vtx;
    }
  }

  int cont=0;
  GlobalVector flightDir_0, flightDir_1;
  reco::Candidate::LorentzVector SV_p4_0 , SV_p4_1;
  for ( std::map<double, size_t>::reverse_iterator iVtx=VTXmass.rbegin(); iVtx!=VTXmass.rend(); ++iVtx)
  {
    ++cont;
    const reco::VertexCompositePtrCandidate &vertex = svTagInfo.secondaryVertex(iVtx->second);
    reco::TrackKinematics vtxKinematics;
    vertexKinematics(vertex, vtxKinematics);
    math::XYZTLorentzVector allSum = allKinematics.weightedVectorSum();
    math::XYZTLorentzVector vertexSum = vtxKinematics.weightedVectorSum();
    if (cont==1)
    {
      SV_mass_0 = vertex.p4().mass()  ;
      SV_EnergyRatio_0 = vertexSum.E() / allSum.E();
      SV_pt_0 = vertex.p4().pt();
      flightDir_0 = svTagInfo.flightDirection(iVtx->second);
      SV_p4_0 = vertex.p4();

      if (reco::deltaR2(flightDir_0,currentAxes[1])<reco::deltaR2(flightDir_0,currentAxes[0]))
        tau_dot = (currentAxes[1].px()*flightDir_0.x()+currentAxes[1].py()*flightDir_0.y()+currentAxes[1].pz()*flightDir_0.z())/(sqrt(currentAxes[1].modp2())*flightDir_0.mag());
      else
        tau_dot = (currentAxes[0].px()*flightDir_0.x()+currentAxes[0].py()*flightDir_0.y()+currentAxes[0].pz()*flightDir_0.z())/(sqrt(currentAxes[0].modp2())*flightDir_0.mag());
    }
    if (cont==2)
    {
      SV_EnergyRatio_1= vertexSum.E() / allSum.E();
      flightDir_1 = svTagInfo.flightDirection(iVtx->second);
      SV_p4_1 = vertex.p4();
      z_ratio = reco::deltaR(flightDir_0,flightDir_1)*SV_pt_0/(SV_p4_0+SV_p4_1).mass();
      break;
    }
  }

  nSM = muonTagInfo.leptons();
  nSE = elecTagInfo.leptons();
  nSL = nSM + nSE;

  float PFLepton_ptrel = -1., PFLepton_IP2D = -1.;

  // PFMuon information
  for (size_t leptIdx = 0; leptIdx < muonTagInfo.leptons() ; ++leptIdx)
  {
    float PFMuon_ptrel = (muonTagInfo.properties(leptIdx).ptRel);
    if (PFMuon_ptrel > PFLepton_ptrel )
    {
      PFLepton_ptrel = PFMuon_ptrel;
      PFLepton_IP2D  = (muonTagInfo.properties(leptIdx).sip2d);
    }
  }

  // PFElectron information
  for (size_t leptIdx = 0; leptIdx <  elecTagInfo.leptons() ; ++leptIdx)
  {
    float PFElectron_ptrel = (elecTagInfo.properties(leptIdx).ptRel);
    if (PFElectron_ptrel > PFLepton_ptrel )
    {
      PFLepton_ptrel = PFElectron_ptrel;
      PFLepton_IP2D  = (elecTagInfo.properties(leptIdx).sip2d);
    }
  }

  std::map<std::string,float> inputs;
  inputs["z_ratio1"] = z_ratio;
  inputs["tau_dot"] = tau_dot;
  inputs["SV_mass_0"] = SV_mass_0;
  inputs["SV_vtx_EnergyRatio_0"] = SV_EnergyRatio_0;
  inputs["SV_vtx_EnergyRatio_1"] = SV_EnergyRatio_1;
  inputs["jetNTracksEtaRel"] = vertexNTracks;
  inputs["PFLepton_ptrel"] = PFLepton_ptrel;
  inputs["PFLepton_IP2D"] = PFLepton_IP2D;
  inputs["nSL"] = nSL;
  inputs["tau2/tau1"] = tau21;
  
  // evaluate the MVA
  value = mvaID->evaluate(inputs);

  // return the final discriminator value
  return value;
}


void CandidateBoostedDoubleSecondaryVertexComputer::calcNsubjettiness(const reco::JetBaseRef & jet, float & tau1, float & tau2, std::vector<fastjet::PseudoJet> & currentAxes) const
{
  std::vector<fastjet::PseudoJet> fjParticles;

  // loop over jet constituents and push them in the vector of FastJet constituents
  for(const reco::CandidatePtr & daughter : jet->daughterPtrVector())
  {
    if ( daughter.isNonnull() && daughter.isAvailable() )
      fjParticles.push_back( fastjet::PseudoJet( daughter->px(), daughter->py(), daughter->pz(), daughter->energy() ) );
    else
      edm::LogWarning("MissingJetConstituent") << "Jet constituent required for N-subjettiness computation is missing!";
  }

  // calculate N-subjettiness
  tau1 = njettiness_.getTau(1, fjParticles);
  tau2 = njettiness_.getTau(2, fjParticles);
  currentAxes = njettiness_.currentAxes();
}


void CandidateBoostedDoubleSecondaryVertexComputer::setTracksPVBase(const reco::TrackRef & trackRef, const reco::VertexRef & vertexRef, float & PVweight) const
{
  PVweight = 0.;

  const reco::TrackBaseRef trackBaseRef( trackRef );

  typedef reco::Vertex::trackRef_iterator IT;

  const reco::Vertex & vtx = *(vertexRef);
  // loop over tracks in vertices
  for(IT it=vtx.tracks_begin(); it!=vtx.tracks_end(); ++it)
  {
    const reco::TrackBaseRef & baseRef = *it;
    // one of the tracks in the vertex is the same as the track considered in the function
    if( baseRef == trackBaseRef )
    {
      PVweight = vtx.trackWeight(baseRef);
      break;
    }
  }
}


void CandidateBoostedDoubleSecondaryVertexComputer::setTracksPV(const reco::CandidatePtr & trackRef, const reco::VertexRef & vertexRef, float & PVweight) const
{
  PVweight = 0.;

  const pat::PackedCandidate * pcand = dynamic_cast<const pat::PackedCandidate *>(trackRef.get());

  if(pcand) // MiniAOD case
  {
    if( pcand->fromPV() == pat::PackedCandidate::PVUsedInFit )
    {
      PVweight = 1.;
    }
  }
  else
  {
    const reco::PFCandidate * pfcand = dynamic_cast<const reco::PFCandidate *>(trackRef.get());

    setTracksPVBase(pfcand->trackRef(), vertexRef, PVweight);
  }
}


void CandidateBoostedDoubleSecondaryVertexComputer::vertexKinematics(const reco::VertexCompositePtrCandidate & vertex, reco::TrackKinematics & vtxKinematics) const
{
  const std::vector<reco::CandidatePtr> & tracks = vertex.daughterPtrVector();

  for(std::vector<reco::CandidatePtr>::const_iterator track = tracks.begin(); track != tracks.end(); ++track) {
    const reco::Track& mytrack = *(*track)->bestTrack();
    vtxKinematics.add(mytrack, 1.0);
  }
}
