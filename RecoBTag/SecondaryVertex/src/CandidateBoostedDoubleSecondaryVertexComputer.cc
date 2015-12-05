#include "RecoBTag/SecondaryVertex/interface/CandidateBoostedDoubleSecondaryVertexComputer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSecondaryVertexTagInfo.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "TrackingTools/IPTools/interface/IPTools.h"

#include "fastjet/contrib/Njettiness.hh"

CandidateBoostedDoubleSecondaryVertexComputer::CandidateBoostedDoubleSecondaryVertexComputer(const edm::ParameterSet & parameters) :
  beta_(parameters.getParameter<double>("beta")),
  R0_(parameters.getParameter<double>("R0")),
  maxSVDeltaRToJet_(parameters.getParameter<double>("maxSVDeltaRToJet")),
  useCondDB_(parameters.getParameter<bool>("useCondDB")),
  gbrForestLabel_(parameters.existsAs<std::string>("gbrForestLabel") ? parameters.getParameter<std::string>("gbrForestLabel") : ""),
  weightFile_(parameters.existsAs<edm::FileInPath>("weightFile") ? parameters.getParameter<edm::FileInPath>("weightFile") : edm::FileInPath()),
  useGBRForest_(parameters.existsAs<bool>("useGBRForest") ? parameters.getParameter<bool>("useGBRForest") : false),
  useAdaBoost_(parameters.existsAs<bool>("useAdaBoost") ? parameters.getParameter<bool>("useAdaBoost") : false),
  maxDistToAxis_(parameters.getParameter<edm::ParameterSet>("trackSelection").getParameter<double>("maxDistToAxis")),
  maxDecayLen_(parameters.getParameter<edm::ParameterSet>("trackSelection").getParameter<double>("maxDecayLen")),
  trackPairV0Filter(parameters.getParameter<edm::ParameterSet>("trackPairV0Filter")),
  trackSelector(parameters.getParameter<edm::ParameterSet>("trackSelection"))
{
  uses(0, "ipTagInfos");
  uses(1, "svTagInfos");

  mvaID.reset(new TMVAEvaluator());
}

void CandidateBoostedDoubleSecondaryVertexComputer::initialize(const JetTagComputerRecord & record)
{
  // variable names and order need to be the same as in the training
  std::vector<std::string> variables({"z_ratio",
                                      "trackSipdSig_3","trackSipdSig_2","trackSipdSig_1","trackSipdSig_0",
                                      "trackSipdSig_1_0","trackSipdSig_0_0","trackSipdSig_1_1","trackSipdSig_0_1",
                                      "trackSip2dSigAboveCharm_0","trackSip2dSigAboveBottom_0","trackSip2dSigAboveBottom_1",
                                      "tau0_trackEtaRel_0","tau0_trackEtaRel_1","tau0_trackEtaRel_2",
                                      "tau1_trackEtaRel_0","tau1_trackEtaRel_1","tau1_trackEtaRel_2",
                                      "tau_vertexMass_0","tau_vertexEnergyRatio_0","tau_vertexDeltaR_0","tau_flightDistance2dSig_0",
                                      "tau_vertexMass_1","tau_vertexEnergyRatio_1","tau_flightDistance2dSig_1",
                                      "jetNTracks","nSV"});
  // book TMVA readers
  std::vector<std::string> spectators({"massPruned", "flavour", "nbHadrons", "ptPruned", "etaPruned"});

  if (useCondDB_)
  {
     const GBRWrapperRcd & gbrWrapperRecord = record.getRecord<GBRWrapperRcd>();

     edm::ESHandle<GBRForest> gbrForestHandle;
     gbrWrapperRecord.get(gbrForestLabel_.c_str(), gbrForestHandle);

     mvaID->initializeGBRForest(gbrForestHandle.product(), variables, spectators, useAdaBoost_);
  }
  else
    mvaID->initialize("Color:Silent:Error", "BDT", weightFile_.fullPath(), variables, spectators, useGBRForest_, useAdaBoost_);

  // get TransientTrackBuilder
  const TransientTrackRecord & transientTrackRcd = record.getRecord<TransientTrackRecord>();
  transientTrackRcd.get("TransientTrackBuilder", trackBuilder);
}

float CandidateBoostedDoubleSecondaryVertexComputer::discriminator(const TagInfoHelper & tagInfo) const
{
  // get TagInfos
  const reco::CandIPTagInfo              & ipTagInfo = tagInfo.get<reco::CandIPTagInfo>(0);
  const reco::CandSecondaryVertexTagInfo & svTagInfo = tagInfo.get<reco::CandSecondaryVertexTagInfo>(1);

  // default discriminator value
  float value = -10.;

  // default variable values
  float z_ratio = dummyZ_ratio;
  float trackSip3dSig_3 = dummyTrackSip3dSig, trackSip3dSig_2 = dummyTrackSip3dSig, trackSip3dSig_1 = dummyTrackSip3dSig, trackSip3dSig_0 = dummyTrackSip3dSig;
  float tau2_trackSip3dSig_0 = dummyTrackSip3dSig, tau1_trackSip3dSig_0 = dummyTrackSip3dSig, tau2_trackSip3dSig_1 = dummyTrackSip3dSig, tau1_trackSip3dSig_1 = dummyTrackSip3dSig;
  float trackSip2dSigAboveCharm_0 = dummyTrackSip2dSigAbove, trackSip2dSigAboveBottom_0 = dummyTrackSip2dSigAbove, trackSip2dSigAboveBottom_1 = dummyTrackSip2dSigAbove;
  float tau1_trackEtaRel_0 = dummyTrackEtaRel, tau1_trackEtaRel_1 = dummyTrackEtaRel, tau1_trackEtaRel_2 = dummyTrackEtaRel;
  float tau2_trackEtaRel_0 = dummyTrackEtaRel, tau2_trackEtaRel_1 = dummyTrackEtaRel, tau2_trackEtaRel_2 = dummyTrackEtaRel;
  float tau1_vertexMass = dummyVertexMass, tau1_vertexEnergyRatio = dummyVertexEnergyRatio, tau1_vertexDeltaR = dummyVertexDeltaR, tau1_flightDistance2dSig = dummyFlightDistance2dSig;
  float tau2_vertexMass = dummyVertexMass, tau2_vertexEnergyRatio = dummyVertexEnergyRatio, tau2_vertexDeltaR = dummyVertexDeltaR, tau2_flightDistance2dSig = dummyFlightDistance2dSig;
  float jetNTracks = 0, nSV = 0, tau1_nSecondaryVertices = 0, tau2_nSecondaryVertices = 0;

  // get the jet reference
  const reco::JetBaseRef jet = svTagInfo.jet();

  std::vector<fastjet::PseudoJet> currentAxes;
  float tau2, tau1;
  // calculate N-subjettiness
  calcNsubjettiness(jet, tau1, tau2, currentAxes);

  const reco::VertexRef & vertexRef = ipTagInfo.primaryVertex();
  GlobalPoint pv(0.,0.,0.);
  if ( ipTagInfo.primaryVertex().isNonnull() )
    pv = GlobalPoint(vertexRef->x(),vertexRef->y(),vertexRef->z());

  const std::vector<reco::CandidatePtr> & selectedTracks = ipTagInfo.selectedTracks();
  const std::vector<reco::btag::TrackIPData> & ipData = ipTagInfo.impactParameterData();
  size_t trackSize = selectedTracks.size();


  reco::TrackKinematics allKinematics;
  std::vector<float> IP3Ds, IP3Ds_1, IP3Ds_2;
  int contTrk=0;

  // loop over tracks associated to the jet
  for (size_t itt=0; itt < trackSize; ++itt)
  {
    const reco::CandidatePtr ptrackRef = selectedTracks[itt];
    const reco::Track * ptrackPtr = reco::btag::toTrack(ptrackRef);
    const reco::Track & ptrack = *ptrackPtr;

    float track_PVweight = 0.;
    setTracksPV(ptrackRef, vertexRef, track_PVweight);
    if (track_PVweight>0.5) allKinematics.add(ptrack, track_PVweight);

    const reco::btag::TrackIPData &data = ipData[itt];
    bool isSelected = false;
    if (trackSelector(ptrack, data, *jet, pv)) isSelected = true;

    // check if the track is from V0
    bool isfromV0 = false, isfromV0Tight = false;
    const reco::Track * trackPairV0Test[2];

    trackPairV0Test[0] = ptrackPtr;

    for (size_t jtt=0; jtt < trackSize; ++jtt)
    {
      if (itt == jtt) continue;

      const reco::btag::TrackIPData & pairTrackData = ipData[jtt];
      const reco::CandidatePtr pairTrackRef = selectedTracks[jtt];
      const reco::Track * pairTrackPtr = reco::btag::toTrack(pairTrackRef);
      const reco::Track & pairTrack = *pairTrackPtr;

      trackPairV0Test[1] = pairTrackPtr;

      if (!trackPairV0Filter(trackPairV0Test, 2))
      {
        isfromV0 = true;

        if ( trackSelector(pairTrack, pairTrackData, *jet, pv) )
          isfromV0Tight = true;
      }

      if (isfromV0 && isfromV0Tight)
        break;
    }

    if( isSelected && !isfromV0Tight ) jetNTracks += 1.;

    reco::TransientTrack transientTrack = trackBuilder->build(ptrack);
    GlobalVector direction(jet->px(), jet->py(), jet->pz());

    int index = 0;
    if (currentAxes.size() > 1 && reco::deltaR2(ptrack,currentAxes[1]) < reco::deltaR2(ptrack,currentAxes[0]))
        index = 1;
    direction = GlobalVector(currentAxes[index].px(), currentAxes[index].py(), currentAxes[index].pz());

    // decay distance and track distance wrt to the closest tau axis
    float decayLengthTau=-1;
    float distTauAxis=-1;

    TrajectoryStateOnSurface closest = IPTools::closestApproachToJet(transientTrack.impactPointState(), *vertexRef , direction, transientTrack.field());
    if (closest.isValid())
      decayLengthTau =  (closest.globalPosition() - RecoVertex::convertPos(vertexRef->position())).mag();

    distTauAxis = std::abs(IPTools::jetTrackDistance(transientTrack, direction, *vertexRef ).second.value());

    float IP3Dsig = ipTagInfo.impactParameterData()[itt].ip3d.significance();

    if( !isfromV0 && decayLengthTau<maxDecayLen_ && distTauAxis<maxDistToAxis_ )
    {
      IP3Ds.push_back( IP3Dsig<-50. ? -50. : IP3Dsig );
      ++contTrk;
      if (currentAxes.size() > 1)
      {
        if (reco::deltaR2(ptrack,currentAxes[0]) < reco::deltaR2(ptrack,currentAxes[1]))
          IP3Ds_1.push_back( IP3Dsig<-50. ? -50. : IP3Dsig );
        else
          IP3Ds_2.push_back( IP3Dsig<-50. ? -50. : IP3Dsig );
      }
      else
        IP3Ds_1.push_back( IP3Dsig<-50. ? -50. : IP3Dsig );
    }
  }

  std::vector<size_t> indices = ipTagInfo.sortedIndexes(reco::btag::IP2DSig);
  bool charmThreshSet = false;

  reco::TrackKinematics kin;
  for (size_t i=0; i<indices.size(); ++i)
  {
    size_t idx = indices[i];
    const reco::btag::TrackIPData & data = ipData[idx];
    const reco::CandidatePtr ptrackRef = selectedTracks[idx];
    const reco::Track * ptrackPtr = reco::btag::toTrack(ptrackRef);
    const reco::Track & track = (*ptrackPtr);

    kin.add(track);

    if ( kin.vectorSum().M() > charmThreshold // charm cut
         && !charmThreshSet )
    {
      trackSip2dSigAboveCharm_0 = data.ip2d.significance();

      charmThreshSet = true;
    }

    if ( kin.vectorSum().M() > bottomThreshold ) // bottom cut
    {
      trackSip2dSigAboveBottom_0 = data.ip2d.significance();
      if ( (i+1)<indices.size() ) trackSip2dSigAboveBottom_1 = (ipData[indices[i+1]]).ip2d.significance();

      break;
    }
  }

  float dummyTrack = -50.;

  std::sort( IP3Ds.begin(),IP3Ds.end(),std::greater<float>() );
  std::sort( IP3Ds_1.begin(),IP3Ds_1.end(),std::greater<float>() );
  std::sort( IP3Ds_2.begin(),IP3Ds_2.end(),std::greater<float>() );
  int num_1 = IP3Ds_1.size();
  int num_2 = IP3Ds_2.size();

  switch(contTrk){
          case 0:

                  trackSip3dSig_0 = dummyTrack;
                  trackSip3dSig_1 = dummyTrack;
                  trackSip3dSig_2 = dummyTrack;
                  trackSip3dSig_3 = dummyTrack;

                  break;

          case 1:

                  trackSip3dSig_0 = IP3Ds.at(0);
                  trackSip3dSig_1 = dummyTrack;
                  trackSip3dSig_2 = dummyTrack;
                  trackSip3dSig_3 = dummyTrack;

                  break;

          case 2:

                  trackSip3dSig_0 = IP3Ds.at(0);
                  trackSip3dSig_1 = IP3Ds.at(1);
                  trackSip3dSig_2 = dummyTrack;
                  trackSip3dSig_3 = dummyTrack;

                  break;

          case 3:

                  trackSip3dSig_0 = IP3Ds.at(0);
                  trackSip3dSig_1 = IP3Ds.at(1);
                  trackSip3dSig_2 = IP3Ds.at(2);
                  trackSip3dSig_3 = dummyTrack;

                  break;

          default:

                  trackSip3dSig_0 = IP3Ds.at(0);
                  trackSip3dSig_1 = IP3Ds.at(1);
                  trackSip3dSig_2 = IP3Ds.at(2);
                  trackSip3dSig_3 = IP3Ds.at(3);

  }

  switch(num_1){
          case 0:

                  tau1_trackSip3dSig_0 = dummyTrack;
                  tau1_trackSip3dSig_1 = dummyTrack;

                  break;

          case 1:

                  tau1_trackSip3dSig_0 = IP3Ds_1.at(0);
                  tau1_trackSip3dSig_1 = dummyTrack;

                  break;

          default:

                  tau1_trackSip3dSig_0 = IP3Ds_1.at(0);
                  tau1_trackSip3dSig_1 = IP3Ds_1.at(1);

  }

  switch(num_2){
          case 0:

                   tau2_trackSip3dSig_0 = dummyTrack;
                   tau2_trackSip3dSig_1 = dummyTrack;

                   break;

          case 1:
                   tau2_trackSip3dSig_0 = IP3Ds_2.at(0);
                   tau2_trackSip3dSig_1 = dummyTrack;

                   break;

          default:

                   tau2_trackSip3dSig_0 = IP3Ds_2.at(0);
                   tau2_trackSip3dSig_1 = IP3Ds_2.at(1);

  }

  math::XYZVector jetDir = jet->momentum().Unit();
  reco::TrackKinematics tau1Kinematics;
  reco::TrackKinematics tau2Kinematics;
  std::vector<float> tau1_trackEtaRels, tau2_trackEtaRels;

  std::map<double, size_t> VTXmap;
  for (size_t vtx = 0; vtx < svTagInfo.nVertices(); ++vtx)
  {
    reco::TrackKinematics vertexKinematic;

    // get the vertex kinematics
    const reco::VertexCompositePtrCandidate vertex = svTagInfo.secondaryVertex(vtx);
    vertexKinematics(vertex, vertexKinematic);

    if (currentAxes.size() > 1)
    {
            if (reco::deltaR2(svTagInfo.flightDirection(vtx),currentAxes[1]) < reco::deltaR2(svTagInfo.flightDirection(vtx),currentAxes[0]))
            {
                    tau2Kinematics = tau2Kinematics + vertexKinematic;
                    if( tau2_flightDistance2dSig < 0 )
                    {
                      tau2_flightDistance2dSig = svTagInfo.flightDistance(vtx,true).significance();
                      tau2_vertexDeltaR = reco::deltaR(svTagInfo.flightDirection(vtx),currentAxes[1]);
                    }
                    etaRelToTauAxis(vertex, currentAxes[1], tau2_trackEtaRels);
                    tau2_nSecondaryVertices += 1.;
            }
            else
            {
                    tau1Kinematics = tau1Kinematics + vertexKinematic;
                    if( tau1_flightDistance2dSig < 0 )
                    {
                      tau1_flightDistance2dSig =svTagInfo.flightDistance(vtx,true).significance();
                      tau1_vertexDeltaR = reco::deltaR(svTagInfo.flightDirection(vtx),currentAxes[0]);
                    }
                    etaRelToTauAxis(vertex, currentAxes[0], tau1_trackEtaRels);
                    tau1_nSecondaryVertices += 1.;
            }

    }
    else if (currentAxes.size() > 0)
    {
            tau1Kinematics = tau1Kinematics + vertexKinematic;
            if( tau1_flightDistance2dSig < 0 )
            {
              tau1_flightDistance2dSig =svTagInfo.flightDistance(vtx,true).significance();
              tau1_vertexDeltaR = reco::deltaR(svTagInfo.flightDirection(vtx),currentAxes[0]);
            }
            etaRelToTauAxis(vertex, currentAxes[1], tau1_trackEtaRels);
            tau1_nSecondaryVertices += 1.;
    }

    GlobalVector flightDir = svTagInfo.flightDirection(vtx);
    if (reco::deltaR2(flightDir, jetDir)<(maxSVDeltaRToJet_*maxSVDeltaRToJet_))
      VTXmap[svTagInfo.flightDistance(vtx).error()]=vtx;
  }
  nSV = VTXmap.size();


  math::XYZTLorentzVector allSum = allKinematics.weightedVectorSum() ;
  if ( tau1_nSecondaryVertices > 0. )
  {
    math::XYZTLorentzVector tau1_vertexSum = tau1Kinematics.weightedVectorSum();
    tau1_vertexEnergyRatio = tau1_vertexSum.E() / allSum.E();
    if ( tau1_vertexEnergyRatio > 50. ) tau1_vertexEnergyRatio = 50.;

    tau1_vertexMass = tau1_vertexSum.M();
  }

  if ( tau2_nSecondaryVertices > 0. )
  {
    math::XYZTLorentzVector tau2_vertexSum = tau2Kinematics.weightedVectorSum();
    tau2_vertexEnergyRatio = tau2_vertexSum.E() / allSum.E();
    if ( tau2_vertexEnergyRatio > 50. ) tau2_vertexEnergyRatio = 50.;

    tau2_vertexMass= tau2_vertexSum.M();
  }


  float dummyEtaRel = -1.;

  std::sort( tau1_trackEtaRels.begin(),tau1_trackEtaRels.end() );
  std::sort( tau2_trackEtaRels.begin(),tau2_trackEtaRels.end() );

  switch(tau2_trackEtaRels.size()){
          case 0:

                  tau2_trackEtaRel_0 = dummyEtaRel;
                  tau2_trackEtaRel_1 = dummyEtaRel;
                  tau2_trackEtaRel_2 = dummyEtaRel;

                  break;

          case 1:

                  tau2_trackEtaRel_0 = tau2_trackEtaRels.at(0);
                  tau2_trackEtaRel_1 = dummyEtaRel;
                  tau2_trackEtaRel_2 = dummyEtaRel;

                  break;

          case 2:

                  tau2_trackEtaRel_0 = tau2_trackEtaRels.at(0);
                  tau2_trackEtaRel_1 = tau2_trackEtaRels.at(1);
                  tau2_trackEtaRel_2 = dummyEtaRel;

                  break;

          default:

                  tau2_trackEtaRel_0 = tau2_trackEtaRels.at(0);
                  tau2_trackEtaRel_1 = tau2_trackEtaRels.at(1);
                  tau2_trackEtaRel_2 = tau2_trackEtaRels.at(2);

  }

  switch(tau1_trackEtaRels.size()){
          case 0:

                  tau1_trackEtaRel_0 = dummyEtaRel;
                  tau1_trackEtaRel_1 = dummyEtaRel;
                  tau1_trackEtaRel_2 = dummyEtaRel;

                  break;

          case 1:

                  tau1_trackEtaRel_0 = tau1_trackEtaRels.at(0);
                  tau1_trackEtaRel_1 = dummyEtaRel;
                  tau1_trackEtaRel_2 = dummyEtaRel;

                  break;

          case 2:

                  tau1_trackEtaRel_0 = tau1_trackEtaRels.at(0);
                  tau1_trackEtaRel_1 = tau1_trackEtaRels.at(1);
                  tau1_trackEtaRel_2 = dummyEtaRel;

                  break;

          default:

                  tau1_trackEtaRel_0 = tau1_trackEtaRels.at(0);
                  tau1_trackEtaRel_1 = tau1_trackEtaRels.at(1);
                  tau1_trackEtaRel_2 = tau1_trackEtaRels.at(2);

  }

  int cont=0;
  GlobalVector flightDir_0, flightDir_1;
  reco::Candidate::LorentzVector SV_p4_0 , SV_p4_1;
  double vtxMass = 0.;

  for ( std::map<double, size_t>::iterator iVtx=VTXmap.begin(); iVtx!=VTXmap.end(); ++iVtx)
  {
    ++cont;
    const reco::VertexCompositePtrCandidate &vertex = svTagInfo.secondaryVertex(iVtx->second);
    if (cont==1)
    {
      flightDir_0 = svTagInfo.flightDirection(iVtx->second);
      SV_p4_0 = vertex.p4();
      vtxMass = SV_p4_0.mass();

      if(vtxMass > 0.)
        z_ratio = reco::deltaR(currentAxes[1],currentAxes[0])*SV_p4_0.pt()/vtxMass;
    }
    if (cont==2)
    {
      flightDir_1 = svTagInfo.flightDirection(iVtx->second);
      SV_p4_1 = vertex.p4();
      vtxMass = (SV_p4_1+SV_p4_0).mass();

      if(vtxMass > 0.)
        z_ratio = reco::deltaR(flightDir_0,flightDir_1)*SV_p4_1.pt()/vtxMass;

      break;
    }
  }

  // when only one tau axis has SVs assigned, they are all assigned to the 1st tau axis
  // in the special case below need to swap values
  if( (tau1_vertexMass<0 && tau2_vertexMass>0) )
  {
    float temp = tau1_trackEtaRel_0;
    tau1_trackEtaRel_0= tau2_trackEtaRel_0;
    tau2_trackEtaRel_0= temp;

    temp = tau1_trackEtaRel_1;
    tau1_trackEtaRel_1= tau2_trackEtaRel_1;
    tau2_trackEtaRel_1= temp;

    temp = tau1_trackEtaRel_2;
    tau1_trackEtaRel_2= tau2_trackEtaRel_2;
    tau2_trackEtaRel_2= temp;

    temp = tau1_flightDistance2dSig;
    tau1_flightDistance2dSig= tau2_flightDistance2dSig;
    tau2_flightDistance2dSig= temp;

    tau1_vertexDeltaR= tau2_vertexDeltaR;

    temp = tau1_vertexEnergyRatio;
    tau1_vertexEnergyRatio= tau2_vertexEnergyRatio;
    tau2_vertexEnergyRatio= temp;

    temp = tau1_vertexMass;
    tau1_vertexMass= tau2_vertexMass;
    tau2_vertexMass= temp;
  }


  std::map<std::string,float> inputs;
  inputs["z_ratio"] = z_ratio;
  inputs["trackSipdSig_3"] = trackSip3dSig_3;
  inputs["trackSipdSig_2"] = trackSip3dSig_2;
  inputs["trackSipdSig_1"] = trackSip3dSig_1;
  inputs["trackSipdSig_0"] = trackSip3dSig_0;
  inputs["trackSipdSig_1_0"] = tau2_trackSip3dSig_0;
  inputs["trackSipdSig_0_0"] = tau1_trackSip3dSig_0;
  inputs["trackSipdSig_1_1"] = tau2_trackSip3dSig_1;
  inputs["trackSipdSig_0_1"] = tau1_trackSip3dSig_1;
  inputs["trackSip2dSigAboveCharm_0"] = trackSip2dSigAboveCharm_0;
  inputs["trackSip2dSigAboveBottom_0"] = trackSip2dSigAboveBottom_0;
  inputs["trackSip2dSigAboveBottom_1"] = trackSip2dSigAboveBottom_1;
  inputs["tau1_trackEtaRel_0"] = tau2_trackEtaRel_0;
  inputs["tau1_trackEtaRel_1"] = tau2_trackEtaRel_1;
  inputs["tau1_trackEtaRel_2"] = tau2_trackEtaRel_2;
  inputs["tau0_trackEtaRel_0"] = tau1_trackEtaRel_0;
  inputs["tau0_trackEtaRel_1"] = tau1_trackEtaRel_1;
  inputs["tau0_trackEtaRel_2"] = tau1_trackEtaRel_2;
  inputs["tau_vertexMass_0"] = tau1_vertexMass;
  inputs["tau_vertexEnergyRatio_0"] = tau1_vertexEnergyRatio;
  inputs["tau_vertexDeltaR_0"] = tau1_vertexDeltaR;
  inputs["tau_flightDistance2dSig_0"] = tau1_flightDistance2dSig;
  inputs["tau_vertexMass_1"] = tau2_vertexMass;
  inputs["tau_vertexEnergyRatio_1"] = tau2_vertexEnergyRatio;
  inputs["tau_flightDistance2dSig_1"] = tau2_flightDistance2dSig;
  inputs["jetNTracks"] = jetNTracks;
  inputs["nSV"] = nSV;

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

  // N-subjettiness calculator
  fastjet::contrib::Njettiness njettiness(fastjet::contrib::OnePass_KT_Axes(), fastjet::contrib::NormalizedMeasure(beta_,R0_));

  // calculate N-subjettiness
  tau1 = njettiness.getTau(1, fjParticles);
  tau2 = njettiness.getTau(2, fjParticles);
  currentAxes = njettiness.currentAxes();
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


void CandidateBoostedDoubleSecondaryVertexComputer::etaRelToTauAxis(const reco::VertexCompositePtrCandidate & vertex,
                                                                    fastjet::PseudoJet & tauAxis, std::vector<float> & tau_trackEtaRel) const
{
  math::XYZVector direction(tauAxis.px(), tauAxis.py(), tauAxis.pz());
  const std::vector<reco::CandidatePtr> & tracks = vertex.daughterPtrVector();

  for(std::vector<reco::CandidatePtr>::const_iterator track = tracks.begin(); track != tracks.end(); ++track)
    tau_trackEtaRel.push_back(std::abs(reco::btau::etaRel(direction.Unit(), (*track)->momentum())));
}
