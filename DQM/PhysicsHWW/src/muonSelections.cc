#include <iostream>

#include "Math/VectorUtil.h"

#include "DQM/PhysicsHWW/interface/muonSelections.h"
#include "DQM/PhysicsHWW/interface/trackSelections.h"

using namespace std;

namespace HWWFunctions {

  ////////////////////
  // Identification //
  ////////////////////

  bool muonId(HWW& hww, unsigned int index, SelectionType type){

      float isovalue;
      bool  truncated = true;

      switch(type) {

          ///////////////
          // Higgs, WW //
          ///////////////

          // WW
      case NominalWWV0:
      case NominalWWV1:
          isovalue = 0.15;
          break;
      case muonSelectionFO_mu_wwV1:
      case muonSelectionFO_mu_ww:
          isovalue = 0.40;
          break;
      case muonSelectionFO_mu_smurf_04:
          if (!muonIdNotIsolated(hww, index, type )) return false;
          return muonIsoValuePF(hww, index,0,0.3) < 0.40;
          break;
      case muonSelectionFO_mu_wwV1_iso10_d0:
      case muonSelectionFO_mu_wwV1_iso10:
      case muonSelectionFO_mu_ww_iso10:
          isovalue = 1.0;
          break;

          // SMURF
      case muonSelectionFO_mu_smurf_10:
          if (!muonIdNotIsolated(hww, index, type )) return false;
          return muonIsoValuePF(hww, index,0,0.3) < 1.0;
          break;
      case NominalSmurfV3:
          if (!muonIdNotIsolated(hww, index, type )) return false;
          if (hww.mus_p4().at(index).pt()<20) 
              return muonIsoValue(hww, index,false) < 0.1;
          else
              return muonIsoValue(hww, index,false) < 0.15;
          break;
      case NominalSmurfV4:
          if (!muonIdNotIsolated(hww, index, type )) return false;
          if (hww.mus_p4().at(index).pt()>20) {
              if (TMath::Abs(hww.mus_p4().at(index).eta())<1.479) return muonIsoValuePF(hww, index,0) < 0.22;
              else return muonIsoValuePF(hww, index,0) < 0.20;
          } else {
              return muonIsoValuePF(hww, index,0) < 0.11;
          }
          break;
      case NominalSmurfV5:
      case NominalSmurfV6:
          if (!muonIdNotIsolated(hww, index, type )) return false;
          if (hww.mus_p4().at(index).pt()>20) {
              if (TMath::Abs(hww.mus_p4().at(index).eta())<1.479) return muonIsoValuePF(hww, index,0,0.3) < 0.13;
              else return muonIsoValuePF(hww, index,0,0.3) < 0.09;
          } else {
              if (TMath::Abs(hww.mus_p4().at(index).eta())<1.479) return muonIsoValuePF(hww, index,0,0.3) < 0.06;
              else return muonIsoValuePF(hww, index,0,0.3) < 0.05;
          }
          break;



          /////////////
          // Default //
          /////////////
      default:
          edm::LogError("InvalidInput") << "muonID ERROR: requested muon type is not defined. Abort.";
          exit(1);
          return false;
      } 
      return 
          muonIdNotIsolated(hww, index, type ) &&   // Id
          muonIsoValue(hww, index,truncated) < isovalue;           // Isolation cut
  }


  ////////////////////
  // Identification //
  ////////////////////

  bool muonIdNotIsolated(HWW& hww, unsigned int index, SelectionType type) {

      if ( hww.mus_p4().at(index).pt() < 5.0) {
          return false;
      }

      switch (type) {

      case NominalWWV0:
          if ( TMath::Abs(hww.mus_p4().at(index).eta()) > 2.4)  return false; // eta cut
          if (hww.mus_gfit_chi2().at(index)/hww.mus_gfit_ndof().at(index) >= 10) return false; //glb fit chisq
          if (((hww.mus_type().at(index)) & (1<<1)) == 0)    return false; // global muon
          if (((hww.mus_type().at(index)) & (1<<2)) == 0)    return false; // tracker muon
          if (hww.mus_validHits().at(index) < 11)            return false; // # of tracker hits  
          if (hww.mus_gfit_validSTAHits().at(index)==0 )     return false; // Glb fit must have hits in mu chambers
          if (TMath::Abs(mud0PV(hww, index)) >= 0.02)              return false; // d0 from pvtx
          return true;
          break;

      case muonSelectionFO_mu_wwV1:
      case muonSelectionFO_mu_wwV1_iso10:
      case NominalWWV1:
          if ( TMath::Abs(hww.mus_p4().at(index).eta()) > 2.4)  return false; // eta cut
          if (hww.mus_gfit_chi2().at(index)/hww.mus_gfit_ndof().at(index) >= 10) return false; //glb fit chisq
          if (((hww.mus_type().at(index)) & (1<<1)) == 0)    return false; // global muon
          if (((hww.mus_type().at(index)) & (1<<2)) == 0)    return false; // tracker muon
          if (hww.mus_validHits().at(index) < 11)            return false; // # of tracker hits  
          if (hww.mus_gfit_validSTAHits().at(index)==0 )     return false; // Glb fit must have hits in mu chambers
          if (TMath::Abs(mud0PV_wwV1(hww, index)) >= 0.02)         return false; // d0 from pvtx
          if (TMath::Abs(mudzPV_wwV1(hww, index)) >= 1.0)          return false; // dz from pvtx
          if (hww.mus_ptErr().at(index)/hww.mus_p4().at(index).pt()>0.1) return false;
          if (hww.trks_valid_pixelhits().at(hww.mus_trkidx().at(index))==0) return false;
          if (hww.mus_nmatches().at(index)<2) return false;
          return true;
          break;

      case muonSelectionFO_mu_wwV1_iso10_d0: // same as muonSelectionFO_mu_wwV1_iso10 but with looser d0 cut
          if ( TMath::Abs(hww.mus_p4().at(index).eta()) > 2.4)  return false; // eta cut
          if (hww.mus_gfit_chi2().at(index)/hww.mus_gfit_ndof().at(index) >= 10) return false; //glb fit chisq
          if (((hww.mus_type().at(index)) & (1<<1)) == 0)    return false; // global muon
          if (((hww.mus_type().at(index)) & (1<<2)) == 0)    return false; // tracker muon
          if (hww.mus_validHits().at(index) < 11)            return false; // # of tracker hits  
          if (hww.mus_gfit_validSTAHits().at(index)==0 )     return false; // Glb fit must have hits in mu chambers
          if (TMath::Abs(mud0PV_wwV1(hww, index)) >= 0.2)         return false; // d0 from pvtx
          if (TMath::Abs(mudzPV_wwV1(hww, index)) >= 1.0)          return false; // dz from pvtx
          if (hww.mus_ptErr().at(index)/hww.mus_p4().at(index).pt()>0.1) return false;
          if (hww.trks_valid_pixelhits().at(hww.mus_trkidx().at(index))==0) return false;
          if (hww.mus_nmatches().at(index)<2) return false;
          return true;
          break;

      case muonSelectionFO_mu_ww:
          if ( TMath::Abs(hww.mus_p4().at(index).eta()) > 2.4)  return false; // eta cut
          if (hww.mus_gfit_chi2().at(index)/hww.mus_gfit_ndof().at(index) >= 10) return false; //glb fit chisq
          if (((hww.mus_type().at(index)) & (1<<1)) == 0)    return false; // global muon
          if (((hww.mus_type().at(index)) & (1<<2)) == 0)    return false; // tracker muon
          if (hww.mus_validHits().at(index) < 11)            return false; // # of tracker hits  
          if (hww.mus_gfit_validSTAHits().at(index)==0 )     return false; // Glb fit must have hits in mu chambers
          if (TMath::Abs(mud0PV(hww, index)) >= 0.02)              return false; // d0 from pvtx
          return true;

      case muonSelectionFO_mu_ww_iso10:
          if ( TMath::Abs(hww.mus_p4().at(index).eta()) > 2.4)  return false; // eta cut
          if (hww.mus_gfit_chi2().at(index)/hww.mus_gfit_ndof().at(index) >= 10) return false; //glb fit chisq
          if (((hww.mus_type().at(index)) & (1<<1)) == 0)    return false; // global muon
          if (((hww.mus_type().at(index)) & (1<<2)) == 0)    return false; // tracker muon
          if (hww.mus_validHits().at(index) < 11)            return false; // # of tracker hits  
          if (hww.mus_gfit_validSTAHits().at(index)==0 )     return false; // Glb fit must have hits in mu chambers
          if (TMath::Abs(mud0PV(hww, index)) >= 0.02)              return false; // d0 from pvtx
          return true;

      case NominalSmurfV3:
      case NominalSmurfV4:
      case NominalSmurfV5:
          if (type == NominalSmurfV3 || type == NominalSmurfV4 || type == NominalSmurfV5){
              if (hww.mus_p4().at(index).pt()<20){
                  if (TMath::Abs(mud0PV_smurfV3(hww, index)) >= 0.01)    return false; // d0 from pvtx
              } else {
                  if (TMath::Abs(mud0PV_smurfV3(hww, index)) >= 0.02)    return false; // d0 from pvtx
              }
          } else {
              if (TMath::Abs(mud0PV_smurfV3(hww, index)) >= 0.2)    return false; // d0 from pvtx
          }
          if ( TMath::Abs(hww.mus_p4().at(index).eta()) > 2.4)  return false; // eta cut
          if (hww.mus_gfit_chi2().at(index)/hww.mus_gfit_ndof().at(index) >= 10) return false; //glb fit chisq
          if (((hww.mus_type().at(index)) & (1<<1)) == 0)    return false; // global muon
          if (((hww.mus_type().at(index)) & (1<<2)) == 0)    return false; // tracker muon
          if (hww.mus_validHits().at(index) < 11)            return false; // # of tracker hits  
          if (hww.mus_gfit_validSTAHits().at(index)==0 )     return false; // Glb fit must have hits in mu chambers
          if (TMath::Abs(mudzPV_smurfV3(hww, index)) >= 0.1)       return false; // dz from pvtx
          if (hww.mus_ptErr().at(index)/hww.mus_p4().at(index).pt()>0.1) return false;
          if (hww.trks_valid_pixelhits().at(hww.mus_trkidx().at(index))==0) return false;
          if (hww.mus_nmatches().at(index)<2) return false;
          return true;
          break;


      case muonSelectionFO_mu_smurf_04:
      case muonSelectionFO_mu_smurf_10:
      case NominalSmurfV6:
      {
          if (type == NominalSmurfV6){
              if (hww.mus_p4().at(index).pt()<20){
                  if (TMath::Abs(mud0PV_smurfV3(hww, index)) >= 0.01)    return false; // d0 from pvtx
              } else {
                  if (TMath::Abs(mud0PV_smurfV3(hww, index)) >= 0.02)    return false; // d0 from pvtx
              }
          } else {
              if (TMath::Abs(mud0PV_smurfV3(hww, index)) >= 0.2)    return false; // d0 from pvtx
          }
          if ( TMath::Abs(hww.mus_p4().at(index).eta()) > 2.4)  return false; // eta cut
          if (hww.mus_validHits().at(index) < 11)            return false; // # of tracker hits  
          if (TMath::Abs(mudzPV_smurfV3(hww, index)) >= 0.1)       return false; // dz from pvtx
          if (hww.mus_ptErr().at(index)/hww.mus_p4().at(index).pt()>0.1) return false;
          if (hww.trks_valid_pixelhits().at(hww.mus_trkidx().at(index))==0) return false;
          bool goodMuonGlobalMuon = false;
          if (((hww.mus_type().at(index)) & (1<<1)) != 0) { // global muon
              goodMuonGlobalMuon = true;
              if (hww.mus_nmatches().at(index)<2) goodMuonGlobalMuon = false;
              if (hww.mus_gfit_chi2().at(index)/hww.mus_gfit_ndof().at(index) >= 10) goodMuonGlobalMuon = false; //glb fit chisq
              if (hww.mus_gfit_validSTAHits().at(index)==0 ) goodMuonGlobalMuon = false; // Glb fit must have hits in mu chambers
          } 
          bool goodMuonTrackerMuon = false;
          if (((hww.mus_type().at(index)) & (1<<2)) != 0) { // tracker muon
              goodMuonTrackerMuon = true;
              if (hww.mus_pid_TMLastStationTight().at(index) == 0 ) goodMuonTrackerMuon = false; // last station tight
          }
          return goodMuonGlobalMuon || goodMuonTrackerMuon;
          break;
      }
      default:
          edm::LogError("InvalidInput") << "muonID ERROR: requested muon type is not defined. Abort.";
          return false;
      }
  }


  ////////////////////////////
  // Isolation Calculations //
  ////////////////////////////

  double muonIsoValue(HWW& hww, unsigned int index, bool truncated ){
      return ( muonIsoValue_TRK(hww, index, truncated ) + muonIsoValue_ECAL(hww, index, truncated ) + muonIsoValue_HCAL(hww, index, truncated ) );
  }
  double muonIsoValue_TRK(HWW& hww, unsigned int index, bool truncated ){
      double pt        = hww.mus_p4().at(index).pt();
      if(truncated) pt = max( pt, 20.0 );
      return hww.mus_iso03_sumPt().at(index) / pt;
  }
  double muonIsoValue_ECAL(HWW& hww, unsigned int index, bool truncated ){
      double pt  = hww.mus_p4().at(index).pt();
      if(truncated) pt = max( pt, 20.0 );
      return hww.mus_iso03_emEt().at(index) / pt;
  }
  double muonIsoValue_HCAL(HWW& hww, unsigned int index, bool truncated){
      double pt  = hww.mus_p4().at(index).pt();
      if(truncated) pt = max( pt, 20.0 );
      return hww.mus_iso03_hadEt().at(index) / pt;
  }

  #ifdef PFISOFROMNTUPLE
  double muonIsoValuePF(HWW& hww, unsigned int imu, unsigned int ivtx, float coner, float minptn, float dzcut, int filterId){
      if (fabs(coner-0.3)<0.0001) {
          if (hww.mus_iso03_pf().at(imu)<-99.) return 9999.;
          return hww.mus_iso03_pf().at(imu)/hww.mus_p4().at(imu).pt();
      } else if (fabs(coner-0.4)<0.0001) {
          if (hww.mus_iso04_pf().at(imu)<-99.) return 9999.;
          return hww.mus_iso04_pf().at(imu)/hww.mus_p4().at(imu).pt();
      } else {
          edm::LogWarning("InvalidInput") << "muonIsoValuePF: CONE SIZE NOT SUPPORTED";
          return 9999.;
      }
  }
  #else
  double muonIsoValuePF(HWW& hww, unsigned int imu, unsigned int ivtx, float coner, float minptn, float dzcut, int filterId){
      float pfciso = 0;
      float pfniso = 0;
      int mutkid = hww.mus_trkidx().at(imu);
      float mudz = mutkid>=0 ? trks_dz_pv(hww, mutkid,ivtx).first : hww.mus_sta_z0corr().at(imu);
      for (unsigned int ipf=0; ipf<hww.pfcands_p4().size(); ++ipf){
          float dR = ROOT::Math::VectorUtil::DeltaR( hww.pfcands_p4().at(ipf), hww.mus_p4().at(imu) );
          if (dR>coner) continue;
          float pfpt = hww.pfcands_p4().at(ipf).pt();
          int pfid = abs(hww.pfcands_particleId().at(ipf));
          if (filterId!=0 && filterId!=pfid) continue;
          if (hww.pfcands_charge().at(ipf)==0) {
              //neutrals
              if (pfpt>minptn) pfniso+=pfpt;
          } else {
              //charged
              //avoid double counting of muon itself
              int pftkid = hww.pfcands_trkidx().at(ipf);
              if (mutkid>=0 && pftkid>=0 && mutkid==pftkid) continue;
              //first check electrons with gsf track
              if (abs(hww.pfcands_particleId().at(ipf))==11 && hww.pfcands_pfelsidx().at(ipf)>=0 && hww.pfels_elsidx().at(hww.pfcands_pfelsidx().at(ipf))>=0) {
                  int gsfid = hww.els_gsftrkidx().at(hww.pfels_elsidx().at(hww.pfcands_pfelsidx().at(ipf))); 
                  if (gsfid>=0) { 
                      if(fabs(gsftrks_dz_pv(hww, gsfid,ivtx ).first - mudz )<dzcut) {//dz cut
                          pfciso+=pfpt;
                      }   
                      continue;//and avoid double counting
                  }
              }
              //then check anything that has a ctf track
              if (hww.pfcands_trkidx().at(ipf)>=0) {//charged (with a ctf track)
                  if(fabs( trks_dz_pv(hww, hww.pfcands_trkidx().at(ipf),ivtx).first - mudz )<dzcut) {//dz cut
                      pfciso+=pfpt;
                  }
              } 
          }
      } 
      return (pfciso+pfniso)/hww.mus_p4().at(imu).pt();
  }
  #endif


  /////////////////////////////
  // Muon d0 corrected by PV //
  ////////////////////////////

  double mud0PV(HWW& hww, unsigned int index){
      if ( hww.vtxs_sumpt().empty() ) return 9999.;
      unsigned int iMax = 0;
      double sumPtMax = hww.vtxs_sumpt().at(0);
      for ( unsigned int i = iMax+1; i < hww.vtxs_sumpt().size(); ++i )
          if ( hww.vtxs_sumpt().at(i) > sumPtMax ){
              iMax = i;
              sumPtMax = hww.vtxs_sumpt().at(i);
          }
      double dxyPV = hww.mus_d0().at(index)-
          hww.vtxs_position().at(iMax).x()*sin(hww.mus_trk_p4().at(index).phi())+
          hww.vtxs_position().at(iMax).y()*cos(hww.mus_trk_p4().at(index).phi());
      return dxyPV;
  }

  double mud0PV_wwV1(HWW& hww, unsigned int index){
      if ( hww.vtxs_sumpt().empty() ) return 9999.;
      double sumPtMax = -1;
      int iMax = -1;
      for ( unsigned int i = 0; i < hww.vtxs_sumpt().size(); ++i ){
          if (hww.vtxs_isFake().at(i)) continue;
          if (hww.vtxs_ndof().at(i) < 4.) continue;
          if (hww.vtxs_position().at(i).Rho() > 2.0) continue;
          if (fabs(hww.vtxs_position().at(i).Z()) > 24.0) continue;
          if ( hww.vtxs_sumpt().at(i) > sumPtMax ){
              iMax = i;
              sumPtMax = hww.vtxs_sumpt().at(i);
          }
      }
      if (iMax<0) return 9999.;
      double dxyPV = hww.mus_d0().at(index)-
          hww.vtxs_position().at(iMax).x()*sin(hww.mus_trk_p4().at(index).phi())+
          hww.vtxs_position().at(iMax).y()*cos(hww.mus_trk_p4().at(index).phi());
      return dxyPV;
  }

  double mud0PV_smurfV3(HWW& hww, unsigned int index){
      int vtxIndex = 0;
      double dxyPV = hww.mus_d0().at(index)-
          hww.vtxs_position().at(vtxIndex).x()*sin(hww.mus_trk_p4().at(index).phi())+
          hww.vtxs_position().at(vtxIndex).y()*cos(hww.mus_trk_p4().at(index).phi());
      return dxyPV;
  }

  double dzPV_mu(HWW& hww, const LorentzVector& vtx, const LorentzVector& p4, const LorentzVector& pv){
      return (vtx.z()-pv.z()) - ((vtx.x()-pv.x())*p4.x()+(vtx.y()-pv.y())*p4.y())/p4.pt() * p4.z()/p4.pt();
  }

  double mudzPV_smurfV3(HWW& hww, unsigned int index){
      int vtxIndex = 0;
      double dzpv = dzPV_mu(hww, hww.mus_vertex_p4().at(index), hww.mus_trk_p4().at(index), hww.vtxs_position().at(vtxIndex));
      return dzpv;
  }

  double mudzPV_wwV1(HWW& hww, unsigned int index){
      if ( hww.vtxs_sumpt().empty() ) return 9999.;
      double sumPtMax = -1;
      int iMax = -1;
      for ( unsigned int i = 0; i < hww.vtxs_sumpt().size(); ++i ){
          if (hww.vtxs_isFake().at(i)) continue;
          if (hww.vtxs_ndof().at(i) < 4.) continue;
          if (hww.vtxs_position().at(i).Rho() > 2.0) continue;
          if (fabs(hww.vtxs_position().at(i).Z()) > 24.0) continue;
          if ( hww.vtxs_sumpt().at(i) > sumPtMax ){
              iMax = i;
              sumPtMax = hww.vtxs_sumpt().at(i);
          }
      }
      if (iMax<0) return 9999.;
      const LorentzVector& vtx = hww.mus_vertex_p4().at(index);
      const LorentzVector& p4 = hww.mus_trk_p4().at(index);
      const LorentzVector& pv = hww.vtxs_position().at(iMax);
      return (vtx.z()-pv.z()) - ((vtx.x()-pv.x())*p4.x()+(vtx.y()-pv.y())*p4.y())/p4.pt() * p4.z()/p4.pt(); 
  }

  bool isPFMuon(HWW& hww, int index , bool requireSamePt , float dpt_max ){

      int ipf = hww.mus_pfmusidx().at( index );

      //--------------------------
      // require matched pfmuon
      //--------------------------

      if( ipf >= int(hww.pfmus_p4().size()) || ipf < 0 ) return false;

      //----------------------------------------------------
      // require PFMuon pt = reco muon pt (within dpt_max)
      //----------------------------------------------------

      if( requireSamePt ){

          float pt_pf = hww.pfmus_p4().at(ipf).pt();
          float pt    = hww.mus_p4().at(index).pt();

          if( fabs( pt_pf - pt ) > dpt_max ) return false;

      }

      return true;

  }

}
