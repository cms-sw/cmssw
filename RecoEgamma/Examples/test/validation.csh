#!/bin/csh

#This script can be used to generate a web page to compare histograms from 
#two input root files produced using the EDAnalyzers in RecoEgamma/Examples,
#by running one of:
#
#  RecoEgamma/Examples/test/GsfElectronMCAnalyzer.cfg
#  RecoEgamma/Examples/test/SimplePhotonAnalyzer.cfg
#  RecoEgamma/Examples/test/SimpleConvertedPhotonAnalyzer.cfg
#
# The default list of histograms (configurable) is based on version V00-01-04
# of RecoEgamma/Examples
#
#Two files are created by this script: validation.C and validation.html.
#validation.C should be run inside root to greate a set of gif images
#which can then be viewed in a web browser using validation.html.

#=============BEGIN CONFIGURATION=================

#Input root trees for the two cases to be compared 
setenv NEWFILE ~/scratch0/CMSSW_2_0_0_pre4/src/RecoEgamma/Examples/test/gsfElectronHistos_RelVal200pre4SingleElectronPt10.root 
setenv OLDFILE ~/scratch0/CMSSW_1_6_7/src/RecoEgamma/Examples/test/gsfElectronHistos_RelVal167SingleElectronPt10.root


#Release versions to be compared (affects output directory name and html description only)
setenv NEWRELEASE 200pre4
setenv OLDRELEASE 167
#Name of sample (affects output directory name and html description only)
setenv SAMPLE RelValSingleElectronPt10
#TYPE must be one of GsfElectron, GsfElectronFake, Photon or ConvertedPhoton
setenv TYPE GsfElectron

#==============END BASIC CONFIGURATION==================

#Location of output.  The default will put your output in:
#http://cmsdoc.cern.ch/Physics/egamma/www/validation/

setenv CURRENTDIR $PWD
setenv OUTPATH /afs/cern.ch/cms/Physics/egamma/www/validation
cd $OUTPATH
if (! -d $NEWRELEASE) then
  mkdir $NEWRELEASE
endif
setenv OUTPATH $OUTPATH/$NEWRELEASE

setenv OUTDIR $OUTPATH/${SAMPLE}_${NEWRELEASE}_${OLDRELEASE}
if (! -d $OUTDIR) then
  cd $OUTPATH
  mkdir $OUTDIR
  cd $OUTDIR
  mkdir gifs
endif
cd $OUTDIR

#The list of histograms to be compared for each TYPE can be configured below:

if ( $TYPE == GsfElectron || $TYPE == GsfElectronFake ) then

cat > scaledhistos <<EOF
  h_ele_PoPtrue   
  h_ele_PoPtrue_barrel   
  h_ele_PoPtrue_endcaps   
  h_scl_EoEtrue_barrel   
  h_scl_EoEtrue_endcaps   
  h_ele_EtaMnEtaTrue   
  h_ele_PhiMnPhiTrue 
  h_ele_vertexP 
  h_ele_vertexPt 
  h_ele_outerP_mode 
  h_ele_outerPt_mode 
  h_ele_vertexX 
  h_ele_vertexY 
  h_ele_vertexZ 
  h_ele_EoP 
  h_ele_EoPout 
  h_ele_dEtaCl_propOut 
  h_ele_dEtaSc_propVtx 
  h_ele_dPhiCl_propOut 
  h_ele_dPhiSc_propVtx 
  h_ele_HoE 
  h_ele_chi2 
  h_ele_foundHits 
  h_ele_lostHits 
  h_ele_PinMnPout_mode 
  h_ele_classes 
  h_ele_charge
  h_ele_EoverP_all
  h_recEleNum
EOF

cat > unscaledhistos <<EOF
  h_ele_absetaEff
  h_ele_etaEff
  h_ele_ptEff
  h_ele_phiEff
  h_ele_zEff
  h_ele_etaEff_all
  h_ele_ptEff_all
  h_ele_PoPtrueVsEta_pfx   
  h_ele_PoPtrueVsPhi_pfx   
  h_ele_EtaMnEtaTrueVsEta_pfx  
  h_ele_PhiMnPhiTrueVsEta_pfx 
  h_ele_vertexPtVsEta_pfx 
  h_ele_EoPVsEta_pfx 
  h_ele_EoPoutVsEta_pfx 
  h_ele_HoEVsEta_pfx 
  h_ele_chi2VsEta_pfx 
  h_ele_foundHitsVsEta_pfx 
  h_ele_fbremvsEtamean
  h_ele_fbremvsEtamode
  h_ele_eta_bbremFrac 
  h_ele_eta_goldenFrac 
  h_ele_eta_narrowFrac 
  h_ele_eta_showerFrac 
EOF

else if ( $TYPE == Photon ) then

cat > scaledhistos <<EOF
  scE
  scEt
  scEta
  scPhi
  deltaEtaSC
  deltaPhiSC
  phoE
  phoEta
  phoPhi
  phoR9Barrel
  phoR9Endcap
  recEoverTrueEBarrel
  recEoverTrueEEndcap
  recESCoverTrueEBarrel
  recESCoverTrueEEndcap
  e5x5_unconvBarrelOverEtrue
  e5x5_unconvEndcapOverEtrue
  ePho_convBarrelOverEtrue
  ePho_convEndcapOverEtrue
  deltaEta
  deltaPhi
EOF

cat > unscaledhistos <<EOF
EOF

else if ( $TYPE == ConvertedPhoton ) then

cat > scaledhistos <<EOF
  deltaE
  deltaPhi
  deltaEta
  MCphoE
  MCphoPhi
  MCphoEta
  MCConvE
  MCConvPt
  MCConvEta
  scE
  scEta
  scPhi
  phoE
  phoEta
  phoPhi
EOF

cat > unscaledhistos <<EOF
EOF

endif

#=================END CONFIGURATION=====================

if (-e validation.C) rm validation.C
touch validation.C
cat > begin.C <<EOF
{
TFile *file_old = TFile::Open("$OLDFILE");
TFile *file_new = TFile::Open("$NEWFILE");

EOF
cat begin.C >>& validation.C
rm begin.C

setenv N 1
foreach i (`cat scaledhistos`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd();
$i->SetLineColor(4);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd();
Double_t nnew=$i->GetEntries();
$i->SetLineColor(2);
$i->SetLineWidth(3);
$i->Scale(nold/nnew);
$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end

foreach i (`cat unscaledhistos`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd();
$i->SetLineColor(4);
$i->SetLineWidth(3);
$i->Draw();
file_new->cd();
$i->SetLineColor(2);
$i->SetLineWidth(3);
$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end

setenv NTOT `expr $N - 1`
setenv N 1
while ( $N <= $NTOT )
  cat temp$N.C >>& validation.C
  rm temp$N.C
  setenv N `expr $N + 1`
end

cat > end.C <<EOF
}
EOF
cat end.C >>& validation.C
rm end.C


if ( $TYPE == GsfElectron ) then
  setenv ANALYZER GsfElectronMCAnalyzer
  setenv CFG GsfElectronMCAnalyzer
else if ( $TYPE == GsfElectronFake ) then
  setenv ANALYZER GsfElectronFakeAnalyzer
  setenv CFG GsfElectronFakeAnalyzer
else if ( $TYPE == Photon ) then
  setenv ANALYZER SimplePhotonAnalyzer
  setenv CFG SimplePhotonAnalyzer
else if ( $TYPE == ConvertedPhoton ) then
  setenv ANALYZER SimpleConvertedPhotonAnalyzer
  setenv CFG SimpleConvertedPhotonAnalyzer
endif

if (-e validation.html) rm validation.html
touch validation.html
cat > begin.html <<EOF
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<html>
<head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8" />
<title>$NEWRELEASE vs $OLDRELEASE $TYPE validation</title>
</head>

<h1>$NEWRELEASE vs $OLDRELEASE $TYPE validation</h1>

<p>The following plots were made using <a href="http://cmslxr.fnal.gov/lxr/source/RecoEgamma/Examples/plugins/$ANALYZER.cc">RecoEgamma/Examples/plugins/$ANALYZER</a>, using <a href="http://cmslxr.fnal.gov/lxr/source/RecoEgamma/Examples/test/$CFG.cfg">RecoEgamma/Examples/test/$CFG.cfg</a>, using $SAMPLE as input.

<p>The script used to make the plots is <a href="validation.C">here</a>.

<p>In all plots below, $OLDRELEASE is in blue, $NEWRELEASE in red.

EOF
cat begin.html >>& validation.html
rm begin.html

setenv N 1
foreach i (`cat scaledhistos unscaledhistos`)
  cat > temp$N.html <<EOF
<br>
<p><img class="image" width="500" src="gifs/$i.gif">
EOF
  setenv N `expr $N + 1`
end

setenv N 1
while ( $N <= $NTOT )
  cat temp$N.html >>& validation.html
  rm temp$N.html
  setenv N `expr $N + 1`
end

cat > end.html <<EOF

</html>
EOF
cat end.html >>& validation.html
rm end.html

rm scaledhistos
rm unscaledhistos

echo "Now paste the following into your terminal window:"
echo ""
echo "cd $OUTDIR"
echo "root -b"
echo ".x validation.C"
echo ".q"
echo "cd $CURRENTDIR"
echo ""
echo "Then you can view your valdation plots here:"
echo "http://cmsdoc.cern.ch/Physics/egamma/www/validation/${NEWRELEASE}/${SAMPLE}_${NEWRELEASE}_${OLDRELEASE}/validation.html"
