#!/bin/csh

# This script can be used to generate a web page to compare histograms from 
# two input root files produced using the EDAnalyzers in RecoEgamma/Examples.

#============= Configuration =================
# This script behavior is tuned by few unix variables and command-line
# arguments. You can use Oval as execution manager, whose configuration is
# within OvalFile. Oval will set the relevant variables for you.
# If you prefer to set the variables and run this script directly,
# here they are :
#
# $1 : eventual first command-line argument, immediatly duplicated into VAL_ENV,
#   is the name of the current context, used to build some default value
#   for other variables, especially VAL_NEW_FILE and VAL_REF_FILE.
# $2 : eventual second command-line argument, immediatly duplicated into VAL_OUTPUT_FILE,
#   is the default base name of the files containing the histograms ; it is
#   also used to build some default value for other variables.
# $3 : eventual third command-line argument, immediatly duplicated into VAL_WEB_SUB_DIR,
#   it is the name of the web subdirectory. Default is ${DBS_SAMPLE}_Ideal
#
# VAL_NEW_FILE : complete path of the file containing the new histograms.
# VAL_REF_FILE : complete path of the file containing the old histograms to compare with.
# 
# VAL_ANALYZER : name of the analyzer used to do the histograms ; it is used to know
#   which histograms must be searched for. The value must be one of GsfElectronMCAnalyzer,
#   GsfElectronDataAnalyzer,  GsfElectronFakeAnalyzer or SimplePhotonAnalyzer.
#
# VAL_NEW_RELEASE : chosen name for the new release to validate ; used in web pages
#   and used to build the path where the web pages will be stored.
# VAL_REF_RELEASE : chosen name of the old release to compare with ; used in web pages
#   and used to build the path where the web pages will be stored.
# DBS_SAMPLE : short chosen name for the current dataset ; used in web pages
#   and used to build the path where the web pages will be stored.
#=========================================================================================


#============== Core config ==================

setenv VAL_ENV $1
setenv VAL_OUTPUT_FILE $2
setenv VAL_WEB_SUB_DIR $3
setenv VAL_ORIGINAL_DIR $cwd
setenv VAL_WEB "/afs/cern.ch/cms/Physics/egamma/www/validation"
setenv VAL_WEB_URL "http://cmsdoc.cern.ch/Physics/egamma/www/validation"

# those must have a value
#setenv VAL_NEW_RELEASE ...
#setenv VAL_REF_RELEASE ...
# those either have a value, or will receive a default below
#setenv VAL_NEW_FILE ...
#setenv VAL_REF_FILE ...

#============== Find and prepare main output directory ==================

echo "VAL_WEB = ${VAL_WEB}"
cd $VAL_WEB

if (! -d $VAL_NEW_RELEASE) then
  mkdir $VAL_NEW_RELEASE
endif

echo "VAL_NEW_RELEASE = ${VAL_NEW_RELEASE}"
cd $VAL_NEW_RELEASE

#============== Find and archive new log and data files ==================

if ( ${?VAL_NEW_FILE} == "0" ) setenv VAL_NEW_FILE ""

if ( ${VAL_NEW_FILE} == "" ) then
  setenv VAL_NEW_FILE ${VAL_ORIGINAL_DIR}/cmsRun.${VAL_ENV}.olog.${VAL_OUTPUT_FILE}
endif

if (! -d data) then
  mkdir data
endif

if ( "${VAL_NEW_FILE}" != "" ) then
  if ( -r "$VAL_NEW_FILE" ) then
    cp -f $VAL_NEW_FILE data
  endif
endif

if ( -e "${VAL_ORIGINAL_DIR}/cmsRun.${VAL_ENV}.olog" ) then
  cp -f ${VAL_ORIGINAL_DIR}/cmsRun.${VAL_ENV}.olog data
endif

if ( -e "${VAL_ORIGINAL_DIR}/dbs_discovery.py.${VAL_ENV}.olog" ) then
  cp -f ${VAL_ORIGINAL_DIR}/dbs_discovery.py.${VAL_ENV}.olog data
endif

echo "VAL_NEW_FILE = ${VAL_NEW_FILE}"

#============== Find reference data file (eventually the freshly copied new data) ==================

if ( ${?VAL_REF_FILE} == "0" ) setenv VAL_REF_FILE ""

if ( ${VAL_REF_FILE} == "" ) then
  if ( -e "${VAL_WEB}/${VAL_REF_RELEASE}/data/cmsRun.${VAL_ENV}.olog.${VAL_OUTPUT_FILE}" ) then
    setenv VAL_REF_FILE ${VAL_WEB}/${VAL_REF_RELEASE}/data/cmsRun.${VAL_ENV}.olog.${VAL_OUTPUT_FILE}
  endif
endif

if ( ${VAL_REF_FILE} == "" ) then
  if ( -e "${VAL_ORIGINAL_DIR}/cmsRun.${VAL_ENV}.oref.${VAL_OUTPUT_FILE}" ) then
    setenv VAL_REF_FILE ${VAL_ORIGINAL_DIR}/cmsRun.${VAL_ENV}.oref.${VAL_OUTPUT_FILE}
  endif
endif
 
echo "VAL_REF_FILE = ${VAL_REF_FILE}"

 
#============== Prepare output subdirectories ==================

if (! -d "vs${VAL_REF_RELEASE}") then
  mkdir "vs${VAL_REF_RELEASE}"
endif

echo "VAL_REF_RELEASE = ${VAL_REF_RELEASE}"
cd "vs${VAL_REF_RELEASE}"

if ( ${VAL_WEB_SUB_DIR} == "" ) then
  if ( "${DBS_COND}" =~ *MC* ) then
    setenv VAL_WEB_SUB_DIR ${DBS_SAMPLE}_Mc
  else if ( "${DBS_COND}" =~ *IDEAL* ) then
    setenv VAL_WEB_SUB_DIR ${DBS_SAMPLE}_Ideal
  else if ( "${DBS_COND}" =~ *STARTUP* ) then
    setenv VAL_WEB_SUB_DIR ${DBS_SAMPLE}_Startup
  else
    setenv VAL_WEB_SUB_DIR ${DBS_SAMPLE}
  endif
endif

if (! -d ${VAL_WEB_SUB_DIR}) then
  mkdir ${VAL_WEB_SUB_DIR}
endif

echo "VAL_WEB_SUB_DIR = ${VAL_WEB_SUB_DIR}"
cd ${VAL_WEB_SUB_DIR}

if (! -d gifs) then
  mkdir gifs
endif

cp -f ${VAL_ORIGINAL_DIR}/newvalidation.C .


#============== Prepare the list of histograms ==================
# The second argument is 1 if the histogram is scaled, 0 otherwise
# The third argument is 1 if the histogram is in log scale, 0 otherwise
# The fourth argument is 1 if the histogram is drawn with errors, 0 otherwise

if ( $VAL_ANALYZER == GsfElectronMCAnalyzer ) then

cat >! histos.txt <<EOF
h_ele_charge 1 0 1 0 0 0 
h_ele_vertexX 1 0 1 0 0 0
h_ele_vertexY 1	0 1 0 0 0	
h_ele_vertexZ 1	0 1 0 0 0
h_ele_vertexTIP 1 1 1 0 0 0 
h_ele_vertexP 1	0 1 0 0 0
h_ele_vertexPt 1 0 1 0 0 0
h_ele_Et 1 0 1 0 0 0
h_ele_outerP_mode 1 0 1 0 0 0
h_ele_outerPt_mode 1 0 1 0 0 0
h_ele_ChargeMnChargeTrue 1 0 1  0 0 0
h_ele_PoPtrue 1 0 1 0 0 0
h_ele_PoPtrue_barrel 1 0 1 0 0 0
h_ele_PoPtrue_endcaps 1	0 1 0 0 0
h_ele_PoPtrue_golden_barrel 1 0 1 0 0 0
h_ele_PoPtrue_golden_endcaps 1	0 1 0 0 0
h_ele_PoPtrue_showering_barrel 1 0 1 0 0 0
h_ele_PoPtrue_showering_endcaps 1 0 1 0 0 0
h_ele_PtoPttrue 1 0 1 0 0 0
h_ele_PtoPttrue_barrel 1 0 1 0 0 0
h_ele_PtoPttrue_endcaps 1 0 1 0 0 0
h_scl_EoEtrue_barrel_new 1 0 1 0 0 0
h_scl_EoEtrue_endcaps_new 1 0 1 0 0 0
h_scl_EoEtrue_barrel_etagap_new 1 0 1  0 0 0
h_scl_EoEtrue_barrel_phigap_new 1 0 1 0 0 0  
h_scl_EoEtrue_ebeegap_new 1 0 1 0 0 0    
h_scl_EoEtrue_endcaps_deegap_new 1 0 1 0 0 0
h_scl_EoEtrue_endcaps_ringgap_new 1 0 1 0 0 0
h_ele_EtaMnEtaTrue 1 0 1 0 0 0
h_ele_PhiMnPhiTrue 1 0 1 0 0 0
h_ele_EoP 1 1 1 0 0 0
h_ele_EoPout 1 1 1 0 0 0
h_ele_EeleOPout 1 1 1 0 0 0
h_ele_EseedOP 1 1 1 0 0 0
h_ele_dEtaCl_propOut 1 1 1 0 0 0
h_ele_dEtaEleCl_propOut 1 1 1 0 0 0
h_ele_dEtaSc_propVtx 1 1 1 0 0 0
h_ele_dPhiCl_propOut 1 1 1 0 0 0
h_ele_dPhiEleCl_propOut 1 1 1 0 0 0
h_ele_dPhiSc_propVtx 1 1 1 0 0 0
h_ele_HoE 1 1 1 0 0 0
h_ele_HoE_fiducial 1 1 1 0 0 0
h_scl_sigetaeta 1 1 1 0 0 0
h_scl_sigietaieta_barrel 1 1 1 0 0 0
h_scl_sigietaieta_endcaps 1 1 1  0 0 0
h_scl_E1x5 1 1 1 0 0 0
h_scl_E2x5max 1 1 1 0 0 0
h_scl_E5x5 1 1 1 0 0 0
h_ele_chi2 1 1 1 0 0 0
h_ele_foundHits 1 0 1 0 0 0
h_ele_lostHits 1 0 1 0 0 0
h_ele_ambiguousTracks 1 1 1 0 0 0
h_ele_seedDphi2 1 0 1 0 0 0
h_ele_seedDrz2 1 0 1 0 0 0
h_ele_seedSubdet2 1 0 1 0 0 0
h_ele_PinMnPout_mode 1 0 1 0 0 0
h_ele_fbrem 1 0 1 0 0 0
h_ele_classes 1	0 1 0 0 0
h_ele_mva 1 0 1 0 0 0
h_ele_tkSumPt_dr03 1 1 1 0 0 0
h_ele_ecalRecHitSumEt_dr03 1 1 1  0 0 0
h_ele_hcalDepth1TowerSumEt_dr03 1 1 1  0 0 0
h_ele_hcalDepth2TowerSumEt_dr03 1 1 1 0 0 0 
h_ele_tkSumPt_dr04 1 1 1 0 0 0 
h_ele_ecalRecHitSumEt_dr04 1 1 1 0 0 0 
h_ele_hcalDepth1TowerSumEt_dr04 1 1 1 0 0 0 
h_ele_hcalDepth2TowerSumEt_dr04 1 1 1 0 0 0 
h_recEleNum 1 0 1 0 0 0
h_ele_EoverP_all 1 1 1  0 0 0
h_ele_EseedOP_all 1 1 1 0 0 0 
h_ele_EoPout_all 1 1 1 0 0 0 
h_ele_EeleOPout_all 1 1 1 0 0 0 
h_ele_TIP_all 1 1 1 0 0 0 
h_ele_dEtaSc_propVtx_all 1 1 1 0 0 0 
h_ele_dPhiSc_propVtx_all 1 1 1 0 0 0 
h_ele_dEtaCl_propOut_all 1 1 1 0 0 0 
h_ele_dPhiCl_propOut_all 1 1 1 0 0 0 
h_ele_HoE_all 1 1 1 0 0 0 
h_ele_mee_all 1 1 1 0 0 0 
h_ele_mee_os 1 1 1 0 0 0 
EOF

cat >> histos.txt <<EOF
h_ele_absetaEff	0 0 1 1 h_ele_simAbsEta_matched h_mc_abseta
h_ele_etaEff 0 0 1 1 h_ele_simEta_matched h_mc_eta
h_ele_ptEff 0 0 1 1 h_ele_simPt_matched h_mc_Pt
h_ele_phiEff 0 0 1 1 h_ele_simPhi_matched h_mc_phi
h_ele_zEff 0 0 1 1 h_ele_simZ_matched h_mc_z
h_ele_etaEff_all 0 0 1 1 h_ele_vertexEta_all h_mc_eta
h_ele_ptEff_all	0 0 1 1 h_ele_vertexPt_all h_mc_Pt
h_ele_absetaQmisid 0 0 1 1 h_ele_abseta_matched_qmisid h_mc_abseta
h_ele_etaQmisid	0 0 1 1 h_ele_eta_matched_qmisid h_mc_eta
h_ele_ptQmisid 0 0 1 1 h_ele_Pt_matched_qmisid h_mc_Pt
h_ele_zQmisid 0 0 1 1 h_ele_z_matched_qmisid h_mc_z
h_ele_vertexPtVsEta_pfx 0 0 1 0 0 0
h_ele_PoPtrueVsEta_pfx 0 0 1 0 0 0 
h_ele_PoPtrueVsPhi_pfx 0 0 1 0 0 0
h_scl_EoEtruePfVseg_pfy 0 0 1 0 0 0
h_ele_EtaMnEtaTrueVsEta_pfx 0 0 1 0 0 0
h_ele_PhiMnPhiTrueVsEta_pfx 0 0 1 0 0 0
h_ele_EoPVsEta_pfx 0 0 1 0 0 0
h_ele_EoPoutVsEta_pfx 0 0 1 0 0 0
h_ele_EeleOPoutVsEta_pfx 0 0 1 0 0 0
h_ele_HoEVsEta_pfx 0 0 1 0 0 0
h_ele_chi2VsEta_pfx 0 0 1 0 0 0
h_ele_foundHitsVsEta_pfx 0 0 1 0 0 0
h_ele_ambiguousTracksVsEta_pfx 0 0 1 0 0 0
h_ele_seedDphi2VsEta_pfx 0 0 1 0 0 0
h_ele_seedDphi2VsPt_pfx 0 0 1 0 0 0
h_ele_seedDrz2VsEta_pfx 0 0 1 0 0 0
h_ele_seedDrz2VsPt_pfx 0 0 1 0 0 0
h_ele_fbremvsEtamean 0 0 1 0 0 0
h_ele_fbremvsEtamode 0 0 1 0 0 0
h_ele_eta_bbremFrac 0 0 0 1 h_ele_eta_golden h_ele_eta
h_ele_eta_goldenFrac 0 0 0 1 h_ele_eta_bbrem h_ele_eta
h_ele_eta_narrowFrac 0 0 0 1 h_ele_eta_narrow h_ele_eta
h_ele_eta_showerFrac 0 0 0 1 h_ele_eta_show h_ele_eta
EOF

else if ($VAL_ANALYZER == GsfElectronDataAnalyzer ) then

cat >! histos.txt <<EOF
h_ele_charge 1 0 1 0 0 0
h_ele_vertexX 1 0 1 0 0 0
h_ele_vertexY 1 0 1 0 0 0
h_ele_vertexZ 1 0 1 0 0 0
h_ele_vertexTIP 1 1 1 0 0 0
h_ele_vertexP 1 0 1 0 0 0
h_ele_vertexPt 1 0 1 0 0 0
h_ele_Et 1 0 1 0 0 0
h_ele_outerP_mode 1 0 1 0 0 0
h_ele_outerPt_mode 1 0 1 0 0 0
h_ele_PoPmatchingObject 1 0 1 0 0 0  
h_ele_PoPmatchingObject_barrel 1 0 1 0 0 0   
h_ele_PoPmatchingObject_endcaps 1 0 1 0 0 0   
h_ele_PtoPtmatchingObject 1 0 1 0 0 0  
h_ele_PtoPtmatchingObject_barrel 1 0 1 0 0 0   
h_ele_PtoPtmatchingObject_endcaps 1 0 1 0 0 0   
h_scl_EoEmatchingObject_barrel 1 0 1 0 0 0
h_scl_EoEmatchingObject_endcaps 1 0 1 0 0 0   
h_ele_EtaMnEtamatchingObject 1 0 1 0 0 0  
h_ele_PhiMnPhimatchingObject 1 0 1 0 0 0
h_ele_EoP 1 1 1  0 0 0 	
h_ele_EoPout 1 1 1 0 0 0
h_ele_EeleOPout 1 1 1 0 0 0
h_ele_EseedOP 1 1 1 0 0 0
h_ele_dEtaCl_propOut 1 1 1 0 0 0
h_ele_dEtaEleCl_propOut 1 1 1 0 0 0
h_ele_dEtaSc_propVtx 1 1 1 0 0 0
h_ele_dPhiCl_propOut 1 1 1 0 0 0
h_ele_dPhiEleCl_propOut 1 1 1 0 0 0
h_ele_dPhiSc_propVtx 1 1 1 0 0 0
h_ele_HoE 1 1 1 0 0 0
h_scl_sigetaeta 1 1 1 0 0 0 
h_scl_sigietaieta_barrel 1 1 1 0 0 0 
h_scl_sigietaieta_endcaps 1 1 1 0 0 0
h_scl_E1x5 1 1 1 0 0 0 
h_scl_E2x5max 1 1 1 0 0 0   
h_scl_E5x5 1 1 1 0 0 0   
h_ele_chi2 1 1 1 0 0 0
h_ele_foundHits 1 0 1 0 0 0
h_ele_lostHits 1 0 1 0 0 0
h_ele_ambiguousTracks 1 1 1 0 0 0
h_ele_PinMnPout_mode 1 0 1 0 0 0 
h_ele_fbrem 1 0 1 0 0 0
h_ele_classes 1 0 1 0 0 0
h_ele_mva 1 0 1 0 0 0
h_ele_tkSumPt_dr03 1 1 1 0 0 0
h_ele_ecalRecHitSumEt_dr03 1 1 1 0 0 0
h_ele_hcalDepth1TowerSumEt_dr03 1 1 1 0 0 0
h_ele_hcalDepth2TowerSumEt_dr03 1 1 1 0 0 0
h_ele_tkSumPt_dr04 1 1 1 0 0 0
h_ele_ecalRecHitSumEt_dr04 1 1 1 0 0 0
h_ele_hcalDepth1TowerSumEt_dr04 1 1 1 0 0 0
h_ele_hcalDepth2TowerSumEt_dr04 1 1 1 0 0 0
h_recEleNum 1 0 1 0 0 0
h_ele_mee_all 1 1 1 0 0 0
h_ele_mee 1 1 1 0 0 0
h_ele_mee_os 1 1 1 0 0 0
EOF

cat >> histos.txt <<EOF
h_ele_absetaEff	0 0 1 1 h_ele_matchingObjectAbsEta_matched h_SC_abseta
h_ele_etaEff 0 0 1 1 h_ele_matchingObjectEta_matched h_SC_eta
h_ele_ptEff 0 0 1 1 h_ele_matchingObjectPt_matched h_SC_Pt
h_ele_phiEff 0 0 1 1 h_ele_matchingObjectPhi_matched h_SC_phi
h_ele_zEff 0 0 1 1 h_ele_matchingObjectZ_matched h_SC_z
h_ele_vertexPtVsEta_pfx 0 0 1 0 0 0
h_ele_PoPmatchingObjectVsEta_pfx 0 0 1 0 0 0
h_ele_PoPmatchingObjectVsPhi_pfx 0 0 1 0 0 0   
h_ele_EtaMnEtamatchingObjectVsEta_pfx 0 0 1 0 0 0 
h_ele_PhiMnPhimatchingObjectVsEta_pfx 0 0 1 0 0 0 
h_ele_EoPVsEta_pfx 0 0 1 0 0 0
h_ele_EoPoutVsEta_pfx 0 0 1 0 0 0
h_ele_EeleOPoutVsEta_pfx 0 0 1 0 0 0
h_ele_HoEVsEta_pfx 0 0 1 0 0 0
h_ele_chi2VsEta_pfx 0 0 1 0 0 0
h_ele_foundHitsVsEta_pfx 0 0 1 0 0 0
h_ele_ambiguousTracksVsEta_pfx 0 0 1 0 0 0
h_ele_seedDphi2VsEta_pfx 0 0 1 0 0 0
h_ele_seedDphi2VsPt_pfx 0 0 1 0 0 0
h_ele_seedDrz2VsEta_pfx 0 0 1 0 0 0
h_ele_seedDrz2VsPt_pfx 0 0 1 0 0 0
h_ele_fbremvsEtamean 0 0 1 0 0 0
h_ele_fbremvsEtamode 0 0 1 0 0 0
h_ele_eta_bbremFrac 0 0 0 1 h_ele_eta_golden h_ele_eta
h_ele_eta_goldenFrac 0 0 0 1 h_ele_eta_bbrem h_ele_eta
h_ele_eta_narrowFrac 0 0 0 1 h_ele_eta_narrow h_ele_eta
h_ele_eta_showerFrac 0 0 0 1 h_ele_eta_show h_ele_eta
EOF

else if ($VAL_ANALYZER == GsfElectronFakeAnalyzer ) then

cat >! histos.txt <<EOF
h_ele_charge 1 0 1 0 0 0
h_ele_vertexX 1	0 1 0 0 0
h_ele_vertexY 1	0 1 0 0 0
h_ele_vertexZ 1	0 1 0 0 0
h_ele_vertexTIP 1 1 1 0 0 0
h_ele_vertexP 1	0 1 0 0 0
h_ele_vertexPt 1 0 1 0 0 0
h_ele_outerP_mode 1 0 1 0 0 0
h_ele_outerPt_mode 1 0 1 0 0 0
h_ele_EoP 1 1 1 0 0 0
h_ele_EoPout 1 1 1 0 0 0
h_ele_EeleOPout 1 1 1 0 0 0
h_ele_EseedOP 1 1 1 0 0 0
h_ele_dEtaCl_propOut 1 1 1 0 0 0
h_ele_dEtaEleCl_propOut 1 1 1 0 0 0
h_ele_dEtaSc_propVtx 1 1 1 0 0 0
h_ele_dPhiCl_propOut 1 1 1 0 0 0
h_ele_dPhiEleCl_propOut 1 1 1 0 0 0
h_ele_dPhiSc_propVtx 1 1 1 0 0 0
h_ele_HoE 1 1 1 0 0 0
h_scl_sigetaeta 1 1 1 0 0 0 
h_scl_sigietaieta_barrel 1 1 1 0 0 0
h_scl_sigietaieta_endcaps 1 1 1 0 0 0
h_scl_E1x5 1 1 1 0 0 0
h_scl_E2x5max 1 1 1 0 0 0
h_scl_E5x5 1 1 1 0 0 0
h_ele_chi2 1 1 1 0 0 0
h_ele_foundHits 1 0 1 0 0 0
h_ele_lostHits 1 0 1 0 0 0
h_ele_ambiguousTracks 1 1 1 0 0 0
h_ele_seedDphi2 1 0 1 0 0 0
h_ele_seedDrz2 1 0 1 0 0 0
h_ele_seedSubdet2 1 0 1 0 0 0
h_ele_fbrem 1 0 1 0 0 0
h_ele_classes 1	0 1 0 0 0
h_ele_mva 1 0 1 0 0 0
h_ele_tkSumPt_dr03 1 1 1 0 0 0
h_ele_ecalRecHitSumEt_dr03 1 1 1 0 0 0
h_ele_hcalDepth1TowerSumEt_dr03 1 1 1 0 0 0
h_ele_hcalDepth2TowerSumEt_dr03 1 1 1 0 0 0
h_ele_tkSumPt_dr04 1 1 1 0 0 0
h_ele_ecalRecHitSumEt_dr04 1 1 1 0 0 0
h_ele_hcalDepth1TowerSumEt_dr04 1 1 1 0 0 0
h_ele_hcalDepth2TowerSumEt_dr04	1 1 1 0 0 0
h_recEleNum 1 0 1 0 0 0
h_ele_EoverP_all 1 1 1 0 0 0
h_ele_EseedOP_all 1 1 1 0 0 0
h_ele_EoPout_all 1 1 1 0 0 0
h_ele_EeleOPout_all 1 1 1 0 0 0
h_ele_TIP_all 1	1 1 0 0 0
h_ele_dEtaSc_propVtx_all 1 1 1 0 0 0
h_ele_dPhiSc_propVtx_all 1 1 1 0 0 0
h_ele_dEtaCl_propOut_all 1 1 1 0 0 0
h_ele_dPhiCl_propOut_all 1 1 1 0 0 0
h_ele_HoE_all 1	1 1 0 0 0
h_ele_mee_all 1	0 1 0 0 0
h_ele_mee_os 1	0 1 0 0 0
EOF

cat >> histos.txt <<EOF
h_ele_absetaEff 0 0 1 1 h_ele_matchingObjectAbsEta_matched h_CaloJet_abseta
h_ele_etaEff 0 0 1 1 h_ele_matchingObjectEta_matched h_CaloJet_eta
h_ele_ptEff 0 0 1 1 h_ele_matchingObjectPt_matched h_CaloJet_Pt
h_ele_phiEff 0 0 1 1 h_ele_matchingObjectPhi_matched h_CaloJet_phi
h_ele_zEff 0 0 1 1 h_ele_matchingObjectZ_matched h_CaloJet_z
h_ele_etaEff_all 0 0 1 1 h_ele_vertexEta_all h_CaloJet_eta
h_ele_ptEff_all 0 0 1 1 h_ele_vertexPt_all h_CaloJet_Pt
h_ele_vertexPtVsEta_pfx 0 0 1 0 0 0
h_ele_EoPVsEta_pfx 0 0 1 0 0 0
h_ele_EoPoutVsEta_pfx 0	0 1 0 0 0
h_ele_EeleOPoutVsEta_pfx 0 0 1 0 0 0
h_ele_HoEVsEta_pfx 0 0 1 0 0 0
h_ele_chi2VsEta_pfx 0 0 1 0 0 0
h_ele_foundHitsVsEta_pfx 0 0 1 0 0 0
h_ele_seedDphi2VsEta_pfx 0 0 1 0 0 0	
h_ele_seedDphi2VsPt_pfx 0 0 1 0 0 0
h_ele_seedDrz2VsEta_pfx 0 0 1 0 0 0
h_ele_seedDrz2VsPt_pfx 0 0 1 0 0 0
h_ele_fbremvsEtamean 0 0 1 0 0 0
h_ele_fbremvsEtamode 0 0 1 0 0 0
h_ele_eta_bbremFrac 0 0 0 1 h_ele_eta_golden h_ele_eta
h_ele_eta_goldenFrac 0 0 0 1 h_ele_eta_bbrem h_ele_eta
h_ele_eta_narrowFrac 0 0 0 1 h_ele_eta_narrow h_ele_eta
h_ele_eta_showerFrac 0 0 0 1 h_ele_eta_show h_ele_eta
EOF

else if ( $VAL_ANALYZER == SimplePhotonAnalyzer ) then

cat >! histos.txt <<EOF
scE 1 0 0
scEt 1 0 0
scEta 1 0 0
scPhi 1 0 0
deltaEtaSC 1 0 0
deltaPhiSC 1 0 0
phoE 1 0 0
phoEta 1 0 0
phoPhi 1 0 0
phoR9Barrel 1 0 0
phoR9Endcap 1 0 0
recEoverTrueEBarrel 1 0 0
recEoverTrueEEndcap 1 0 0
recESCoverTrueEBarrel 1 0 0
recESCoverTrueEEndcap 1 0 0
e5x5_unconvBarrelOverEtrue 1 0 0
e5x5_unconvEndcapOverEtrue 1 0 0
ePho_convBarrelOverEtrue 1 0 0
ePho_convEndcapOverEtrue 1 0 0
deltaEta 1 0 0
deltaPhi 1 0 0
EOF

else if ( $VAL_ANALYZER == SimpleConvertedPhotonAnalyzer ) then

cat >! histos.txt <<EOF
deltaE 1 0 0
deltaPhi 1 0 0
deltaEta 1 0 0
MCphoE 1 0 0
MCphoPhi 1 0 0
MCphoEta 1 0 0
MCConvE 1 0 0
MCConvPt 1 0 0
MCConvEta 1 0 0
scE 1 0 0
scEta 1 0 0
scPhi 1 0 0
phoE 1 0 0
phoEta 1 0 0
phoPhi 1 0 0
EOF

endif


#================= Generate the gifs and validation.html =====================

root -b -l -q newvalidation.C
echo "You can view your validation plots here:"
echo "${VAL_WEB_URL}/${VAL_NEW_RELEASE}/vs${VAL_REF_RELEASE}/${VAL_WEB_SUB_DIR}/validation.html"
