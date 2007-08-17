#!/bin/csh

#This script can be used to generate a web page to compare histograms from 
#two input root files produced using the EDAnalyzers in RecoEgamma/Examples,
#by running one of:
#
#  RecoEgamma/Examples/test/read_gsfElectrons.cfg
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

#Release versions to be compared
setenv OLDRELEASE CMSSW_1_6_0_pre4
setenv NEWRELEASE CMSSW_1_6_0_pre9
#Name of sample (affects output directory name and html description only)
setenv SAMPLE SingleEPt35
#TYPE must be one of PixelMatchGsfElectron, Photon or ConvertedPhoton
setenv TYPE PixelMatchGsfElectron

#Input root trees for the two cases to be compared 
setenv OLDFILE ~/scratch0/$OLDRELEASE/src/RecoEgamma/Examples/test/gsfElectronHistos.root
setenv NEWFILE ~/scratch0/$NEWRELEASE/src/RecoEgamma/Examples/test/gsfElectronHistos.root

#Location of output
setenv OUTPATH ~/www/egammaValidation
if (! -d $OUTPATH) then
  if (! -d ~www) then
    cd ~/; mkdir www
  endif
  cd ~/www; mkdir egammaValidation
endif

#==============END BASIC CONFIGURATION==================

#The list of histograms to be compared for each TYPE can be configured here:

if ( $TYPE == PixelMatchGsfElectron ) then

cat > scaledhistos <<EOF
  h_ele_PoPtrue   
  h_ele_EtaMnEtaTrue   
  h_ele_PhiMnPhiTrue 
  h_ele_vertexP 
  h_ele_vertexPt 
  h_ele_outerP_mode 
  h_ele_outerPt_mode 
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
EOF

cat > unscaledhistos <<EOF
  h_ele_absetaEff
  h_ele_etaEff
  h_ele_ptEff
  h_ele_eta_bbremFrac 
  h_ele_eta_goldenFrac 
  h_ele_eta_narrowFrac 
  h_ele_eta_showerFrac 
EOF

else if ( $TYPE == Photon ) then

cat > scaledhistos <<EOF
  scE
  scEta
  scPhi
  phoE
  phoEta
  phoPhi
  recEoverTrueE
  deltaEta
  deltaPhi
  corrPhoE
  corrPhoEta
  corrPhoPhi
  corrPhoR9
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


setenv OUTDIR $OUTPATH/${NEWRELEASE}_$SAMPLE
if (! -d $OUTDIR) then
  cd $OUTPATH
  mkdir $OUTDIR
endif
cd $OUTDIR

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
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd();
Double_t nnew=$i->GetEntries();
$i->SetLineColor(2);
$i->Scale(nold/nnew);
$i->Draw("same");
c$i->SaveAs("$i.gif");

EOF
  setenv N `expr $N + 1`
end

foreach i (`cat unscaledhistos`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd();
$i->SetLineColor(4);
$i->Draw();
file_new->cd();
$i->SetLineColor(2);
$i->Draw("same");
c$i->SaveAs("$i.gif");

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


if ( $TYPE == PixelMatchGsfElectron ) then
  setenv ANALYZER PixelMatchGsfElectronAnalyzer
  setenv CFG read_gsfElectrons
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
<title>$NEWRELEASE $TYPE validation</title>
</head>

<h1>$NEWRELEASE $TYPE validation</h1>

<p>The following plots were made using <a href="http://cmslxr.fnal.gov/lxr/source/RecoEgamma/Examples/plugins/$ANALYZER.cc?v=$NEWRELEASE">RecoEgamma/Examples/plugins/$ANALYZER</a>, using <a href="http://cmslxr.fnal.gov/lxr/source/RecoEgamma/Examples/test/$CFG.cfg?v=$NEWRELEASE">RecoEgamma/Examples/test/$CFG.cfg</a>, using $SAMPLE RelVal samples as input.
<p>The script used to make the plots is <a href="validation.C">here</a>.

<p>In all plots below, $OLDRELEASE is in blue, $NEWRELEASE in red.

EOF
cat begin.html >>& validation.html
rm begin.html

setenv N 1
foreach i (`cat scaledhistos unscaledhistos`)
  cat > temp$N.html <<EOF
<br>
<p><img class="image" width="500" src="$i.gif">
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

echo "Valdation plots can be viewed here:"
echo "$OUTDIR/validation.html"
echo "after running the root script validation.C from directory $OUTDIR"
