#!/bin/csh

#This script can be used to generate a web page to compare histograms from 
#two input root files produced using the EDAnalyzers in RecoEgamma/Examples,
#by running one of:
#
#  
#  
#  "Validation/RecoEgamma/test/PhotonValidator_cfg.py
#
# The default list of histograms (configurable) is based on version VXX-XX-XX
# of Validation/RecoEgamma
#
#Two files are created by this script: validation.C and validation.html.
#validation.C should be run inside root to greate a set of gif images
#which can then be viewed in a web browser using validation.html.

#=============BEGIN CONFIGURATION=================

#Input root trees for the two cases to be compared 
setenv OLDFILE /data/test/CMSSW_2_2_3/src/DQMOffline/EGamma/test/Summer08Validation_PYTHIA8PhotonJetPt20to25.root
setenv NEWFILE /data/test/CMSSW_2_2_3/src/DQMOffline/EGamma/test/Summer08Validation_PYTHIA8PhotonJetPt20to25_223Reprocessing.root




setenv OLDRELEASE Summer08
setenv NEWRELEASE Summer08Rreprocessing
#Name of sample (affects output directory name and htmldescription only) 
setenv SAMPLE PYTHIA8PhotonJetPt20to25
#setenv SAMPLE H130GGgluonfusion
#TYPE must be one ofPixelMatchGsfElectron, Photon 
setenv TYPE Photon


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




cat > scaledhistosForAllPhotons <<EOF
 nPhoAllEcal
 phoEAllEcal
 phoEtAllEcal
 phoEta
 phoPhi
 r9AllEcal
 r9Barrel
 r9Endcaps
 hOverEAllEcal

EOF


cat > unscaledhistosForAllPhotons <<EOF

 r9VsEt

EOF


cat > scaledhistosForIsolatedPhotons <<EOF
 nPhoAllEcal
 phoEAllEcal
 phoEtAllEcal
 phoEta
 phoPhi
 r9AllEcal
 r9Barrel
 r9Endcaps
 hOverEAllEcal
EOF


cat > unscaledhistosForIsolatedPhotons <<EOF

 r9VsEt

EOF

cat > eff <<EOF
EfficiencyVsEt
EfficiencyVsEta
EOF

cat > isolationVariables <<EOF


ecalSum
hcalSum
isoPtSumHollow
isoPtSumSolid
nIsoTracksHollow
nIsoTracksSolid

EOF






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
foreach i (`cat unscaledhistosForAllPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/Egamma/PhotonAnalyzer/AllPhotons/Et above 0 GeV");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetLineColor(4);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/Egamma/PhotonAnalyzer/AllPhotons/Et above 0 GeV");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(2);
$i->SetLineWidth(3);
$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end




foreach i (`cat scaledhistosForAllPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/Egamma/PhotonAnalyzer/AllPhotons/Et above 0 GeV");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetLineColor(4);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/Egamma/PhotonAnalyzer/AllPhotons/Et above 0 GeV");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(2);
$i->SetLineWidth(3);
$i->Scale(nold/nnew);
$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end




foreach i (`cat unscaledhistosForIsolatedPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/Egamma/PhotonAnalyzer/IsolatedPhotons/Et above 0 GeV");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetLineColor(4);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/Egamma/PhotonAnalyzer/IsolatedPhotons/Et above 0 GeV");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(2);
$i->SetLineWidth(3);
$i->Draw("same");
c$i->SaveAs("gifs/${i}_isolated.gif");

EOF
  setenv N `expr $N + 1`
end




foreach i (`cat scaledhistosForIsolatedPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/Egamma/PhotonAnalyzer/IsolatedPhotons/Et above 0 GeV");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetLineColor(4);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/Egamma/PhotonAnalyzer/IsolatedPhotons/Et above 0 GeV");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(2);
$i->SetLineWidth(3);
$i->Scale(nold/nnew);
$i->Draw("same");
c$i->SaveAs("gifs/${i}_isolated.gif");

EOF
  setenv N `expr $N + 1`
end



foreach i (`cat eff`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/Egamma/PhotonAnalyzer/IsolationVariables/Et above 0 GeV");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetLineColor(4);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/Egamma/PhotonAnalyzer/IsolationVariables/Et above 0 GeV");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(2);
$i->SetLineWidth(3);
$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end



foreach i (`cat isolationVariables`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("DQMData/Egamma/PhotonAnalyzer/IsolationVariables/Et above 0 GeV");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetMaximum(20.);
$i->SetLineColor(4);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
file_new->cd("DQMData/Egamma/PhotonAnalyzer/IsolationVariables/Et above 0 GeV");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
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


if ( $TYPE == PixelMatchGsfElectron ) then
  setenv ANALYZER PixelMatchGsfElectronAnalyzer
  setenv CFG read_gsfElectrons
else if ( $TYPE == Photon ) then
  setenv ANALYZER PhotonAnalyzer
  setenv CFG EgammaDQMOffline_cfg
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

<p>The following plots were made using <a href="http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/DQMOffline/EGgamma/src/$ANALYZER.cc">DQMOffline/EGamma/src/$ANALYZER</a>, 
using <a href="http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/Validation/RecoEgamma/test/$CFG.py">DQMOffline/EGamma/test/$CFG.py</a>, using $SAMPLE as input.
<p>The script used to make the plots is <a href="validation.C">here</a>.

<p>In all plots below, $OLDRELEASE is in blue, $NEWRELEASE in red.


EOF
cat begin.html >>& validation.html
rm begin.html

setenv N 1
foreach i (`cat scaledhistosForAllPhotons unscaledhistosForAllPhotons`)
  cat > temp$N.html <<EOF
<br>
<p><img class="image" width="500" src="gifs/$i.gif">
EOF
  setenv N `expr $N + 1`
end



foreach i (`cat scaledhistosForIsolatedPhotons unscaledhistosForIsolatedPhotons`)
  cat > temp$N.html <<EOF
<br>
<p><img class="image" width="500" src="gifs/${i}_isolated.gif">
EOF
  setenv N `expr $N + 1`
end

foreach i (`cat eff isolationVariables`)
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

rm scaledhistosForAllPhotons
rm unscaledhistosForAllPhotons
rm scaledhistosForIsolatedPhotons
rm unscaledhistosForIsolatedPhotons
rm isolationVariables
rm eff

echo "Now paste the following into your terminal window:"
echo ""
echo "cd $OUTDIR"
echo " root -b"
echo ".x validation.C"
echo ".q"
echo "cd $CURRENTDIR"
echo ""
echo "Then you can view your valdation plots here:"
echo "http://cmsdoc.cern.ch/Physics/egamma/www/validation/${NEWRELEASE}/${SAMPLE}_${NEWRELEASE}_${OLDRELEASE}/validation.html"
